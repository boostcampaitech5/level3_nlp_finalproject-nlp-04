import requests
import re
import pandas as pd
import time

from pathlib import Path
from selectolax.parser import HTMLParser, Node
from supabase import create_client, Client
from tqdm import tqdm, trange
from datetime import datetime
from .secrets import Secrets
from typing import List, Union

SECRETS = Secrets()

def crawl_finance_news(cur_time: str,
					   stock_nums: int = 30,
					   delay: float = 0.0,
					   threshold_num: int = 10,) -> pd.DataFrame:
	"""
	네이버 금융 뉴스를 크롤링하는 함수입니다.

	Params:
		cur_time (str): 현재 시간
		stock_nums (int): KRX300 기준 크롤링할 종목의 개수
		delay (float): 요청 지연 시간
		threshold_num (int): 일자가 넘어간 기사 용인 갯수

	Returns:
		pandas.DataFrame: 크롤링한 뉴스 데이터프레임
	"""
	assert stock_nums <= 300, "stock_nums를 300개 이내로 입력해주세요."

	BASE_DIR = Path.home().joinpath("level3_nlp_finalproject-nlp-04")
	NEWS_DIR = BASE_DIR.joinpath("Data", "News")
	STOCK_DIR = BASE_DIR.joinpath("Data", "Stock")

	# 현재 날짜
	cur_time = datetime.strptime(cur_time, "%Y-%m-%d")

	stock_detail = pd.read_csv(STOCK_DIR.joinpath("stock_krx300.csv"), encoding="cp949")
	stock_detail = stock_detail.loc[:, ["종목코드", "종목명"]]
	stock_detail["종목코드"] = stock_detail["종목코드"].apply(lambda x: str(x).zfill(6))

	pbar = tqdm(zip(stock_detail["종목코드"][:stock_nums], stock_detail["종목명"][:stock_nums]), desc="종목별 뉴스 크롤링")
	for stock_code, company in pbar:
		pbar.set_description(f"종목: {stock_code}|{company}")
		threshold_cnt = 0
		# 뉴스 페이지의 끝을 찾기 위해 크롤링 수행
		page_max_url = f"https://finance.naver.com/item/news_news.naver?code={stock_code}"
		page_max_html = requests.get(page_max_url).text
		page_max_tree = HTMLParser(page_max_html)
		nodes = page_max_tree.css("body > div > table.Nnavi > tbody > tr > td.pgRR > a")

		try:
			PAGE_MAX = int(re.findall(r"(?<=page=)\d+", nodes[-1].attributes['href'])[0])
		except IndexError:
			print("기사가 존재하지 않습니다.")
			return
		content_df = pd.DataFrame([], columns=['company', 'title', 'link', "writer", 'date', 'content']) if NEWS_DIR.joinpath(stock_code + ".csv").exists() is False else pd.read_csv(NEWS_DIR.joinpath(stock_code + ".csv"), encoding="utf-8")


		# 전체 페이지 순환
		for i in trange(1, PAGE_MAX + 1):
			url = f"https://finance.naver.com/item/news_news.naver?code={stock_code}&page={i}&sm=title_entity_id.basic&clusterId="
			html = requests.get(url).text

			tree = HTMLParser(html)
			nodes = tree.css("body > div.tb_cont > table.type5 > tbody > tr")

			# 연관기사로 묶여있는 것들 제거
			for node in nodes:
				if len(node.attributes) != 0 and "relation_lst" in node.attributes["class"]:
					node.decompose()

			# 해당 페이지의 기사들을 content_df에 추가
			for node in nodes:
				content = []
				tds = node.css("td")

				# company -> title -> link -> writer -> date 순으로 추출
				content.append(company)
				# 연관기사로 묶여있는 리스트가 비어있기 때문에, 아닌 것들만 기사 추출
				if len(tds) != 0:
					for td in tds:
						content.append(td.text(strip=True))
						if len(td.css("a")) != 0:
							content.append(td.css("a")[0].attributes["href"])

					title = content[1]
					link = "https://finance.naver.com" + re.findall(r"\S+(?=\&page=)", content[2])[0] # link의 page앞으로만이 실제 유효한 기사 링크
					content[2] = link
					writer = content[3]
					date = content[4]

					written_date = datetime.strptime(date, '%Y.%m.%d %H:%M')
					if (written_date.date() == cur_time.date()) and not (content_df['link'] == link).any(): # 날짜가 같은 날에 작성된 기사만 크롤링
						# 본문 추출을 위해 위에서 link를 이용해 html을 다시 요청
						content_html = requests.get(link).text
						content_tree = HTMLParser(content_html)

						# 본문이 없는 기사(삭제된 기사)가 종종 존재 -> 예외처리
						try:
							res = content_tree.css(
								"body > div#wrap  > div#middle.new_totalinfo  > div.content_wrap  > div#content > div.section.inner_sub > table.view > tbody > tr > td > div")[0].html
						except IndexError:
							continue

						# 본문 추출
						# 기사본문에 해당하는 부분만 뽑기 위해 <div class="link_news"> 이전까지만 추출
						try:
							body = HTMLParser(res[:re.search(r'(<div class="link_news">)', res).start()].replace("<br>", " ")).text(strip=True) # <br> -> " "로 변경
						except AttributeError:
							body = HTMLParser(res.replace("<br>", " ")).text(strip=True)

						# 기사본문 내용 추가
						content.append(body)
						content = pd.DataFrame([content], columns=['company', 'title', 'link', "writer", 'date', 'content'])

						# 전체 기사에 추가
						content_df = pd.concat([content_df, content], ignore_index=True)
						insert_into_DB(content)

						# Ban되는걸 조심하도록 delay 추가
						time.sleep(delay)
					elif written_date.date() < cur_time.date(): # 작성된 날짜가, 현재 시간보다 이전이라면, threshold_cnt 증가
						if threshold_cnt >= threshold_num: # threshold가 넘어가면 break => 기사 탐색 종료
							break
						threshold_cnt += 1

			if threshold_cnt >= threshold_num:  # threshold가 넘어가면 break => 종목 탐생 종료
				break


		if len(content_df) != 0: # content_df가 비어있지 않을 때(날짜가 같지 않은 애들 같은 경우엔 비어있을 수 있음)
			# link 기준으로 중복 제거
			content_df = content_df.drop_duplicates(['link'], keep='first').reset_index(drop=True)

			# csv로 저장
			content_df.to_csv(NEWS_DIR.joinpath(f"{stock_code}.csv"), encoding="utf-8", index=False)

			print(f"===== Successfully Crawled {stock_code}|{company} News, Numbers = {len(content_df)} =====")

def insert_into_DB(content_df: pd.DataFrame):
	url = SECRETS.url
	key = SECRETS.key
	supabase = create_client(url, key)

	_, _ = supabase.table("news").insert(content_df.iloc[0, :].to_dict()).execute()