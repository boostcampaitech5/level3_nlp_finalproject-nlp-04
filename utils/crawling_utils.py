import requests
import json
import os
import re
import pandas as pd
import pathlib
import time
from selectolax.parser import HTMLParser, Node
from supabase import create_client, Client
from tqdm.notebook import tqdm, trange
from datetime import datetime


# TODO: stock_code list로 들어왔을 때 여러개 한번에 처리해서, 종목별로 반환 가능하도록 구현
def crawl_finance_news(stock_code: str = None, delay: float = 0.0, num_limit: int = 3000) -> pd.DataFrame:
	"""
	네이버 금융 뉴스를 크롤링하는 함수입니다.

	Params:
		stock_code (str): 종목 코드
		delay (float): 요청 지연 시간
		num_limit (int): 크롤링할 뉴스 기사의 최대 개수

	Returns:
		pandas.DataFrame: 크롤링한 뉴스 데이터프레임
	"""
	assert stock_code is not None, "종목코드를 입력해주세요"

	stock_detail = pd.read_csv(pathlib.Path.cwd().joinpath("Data", "Stock", "stock_detail.csv"), encoding="cp949")
	stock_detail = stock_detail.loc[:, ["종목코드", "종목명", "시장구분", "업종코드", "업종명"]]
	stock_detail["종목코드"] = stock_detail["종목코드"].apply(lambda x: str(x).zfill(6))
	company = stock_detail[stock_detail['종목코드'] == stock_code]['종목명'].tolist()[0]

	# 뉴스 페이지의 끝을 찾기 위해 크롤링 수행
	page_max_url = f"https://finance.naver.com/item/news_news.naver?code={stock_code}"
	page_max_html = requests.get(page_max_url).text
	page_max_tree = HTMLParser(page_max_html)
	nodes = page_max_tree.css("body > div > table.Nnavi > tbody > tr > td.pgRR > a")

	try:
		PAGE_MAX = min(int(re.findall(r"(?<=page=)\d+", nodes[-1].attributes['href'])[0]), num_limit // 10)
	except IndexError:
		print("기사가 존재하지 않습니다.")
		return

	content_df = pd.DataFrame([], columns=['company', 'title', 'link', "writer", 'date', 'content'])

	cur = 0
	# 전체 페이지 순환
	for i in trange(1, PAGE_MAX + 1):
		if cur < num_limit:
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

				content.append(company)
				# 연관기사로 묶여있는 리스트가 비어있기 때문에, 아닌 것들만 기사 추출
				if len(tds) != 0:
					# title -> link -> writer -> date 순으로 추출
					for td in tds:
						content.append(td.text(strip=True))
						if len(td.css("a")) != 0:
							content.append(td.css("a")[0].attributes["href"])

					# 본문 추출을 위해 위에서 link를 이용해 html을 다시 요청
					content_html = requests.get(f"https://finance.naver.com/{content[2]}").text
					content_tree = HTMLParser(content_html)

					# 본문이 없는 기사(삭제된 기사)가 종종 존재 -> 예외처리
					try:
						res = content_tree.css(
							"body > div#wrap  > div#middle.new_totalinfo  > div.content_wrap  > div#content > div.section.inner_sub > table.view > tbody > tr > td > div")[
							0].html
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
					# full link로 수정
					content['link'] = content['link'].apply(lambda x: "https://finance.naver.com" + x)
					# 전체 기사에 추가
					content_df = pd.concat([content_df, content], ignore_index=True)
					if cur < num_limit:
						insert_into_DB(content)
						cur += 1
					else:
						print("Crawling Finished")
						break
					# Ban되는걸 조심하도록 delay 추가
					time.sleep(delay)
		else:
			break
	# content 기준으로 중복 제거
	content_df = content_df.drop_duplicates(['content'], keep='first').reset_index(drop=True)

	# csv로 저장
	content_df.to_csv(pathlib.Path.cwd().joinpath("Data", "News", f"{stock_code}.csv"), encoding="utf-8", index=False)

	return content_df

def insert_into_DB(content_df: pd.DataFrame):
	url = ""
	key = ""
	supabase = create_client(url, key)

	_, _ = supabase.table("news").insert(content_df.iloc[0, :].to_dict()).execute()