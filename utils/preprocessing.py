import re
import pandas as pd


def remove_text(texts):
	pattern = r"\*? ?재판매 및 DB 금지"

	preprocessed_text = ''

	text = re.sub(pattern, " ", texts).strip()
	if text:
		preprocessed_text = text

	return preprocessed_text


def remove_press(texts):
	"""
    언론 정보를 제거
    -> ~~ 기자 (연합뉴스)
    -> (서울=연합뉴스) ~~ 특파원
    """
	re_patterns = [
		r"\([^(]*?(뉴스|경제|일보|미디어|데일리|한겨례|타임즈|위키트리)\)",
		r"[가-힣]{0,4}\s+(기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장)\s?=?",  # 이름 + 기자
		# r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리)",  # (... 연합뉴스) ..
		r"(\/?뉴스|news|News)\d",  # (... 연합뉴스) ..
		r"[\(\[]\s+[\)\]]",  # (  )
		r"[\(\[]=\s+[\)\]]",  # (=  )
		r"[\)\]]\s+=[\)\]]",  # (  =)
	]

	preprocessed_text = ''
	for re_pattern in re_patterns:
		texts = re.sub(re_pattern, " ", str(texts))
	if texts:
		preprocessed_text = texts

	return preprocessed_text


def remove_photo_info(texts):
	## 수정 필요
	"""
    뉴스의 이미지 설명 대한 label 제거
    """
	preprocessed_text = []

	preprocessed_text = re.sub(r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ", " ", texts).strip()
	preprocessed_text = re.sub(r"\/?사진제공(=|\:)\w+", " ", preprocessed_text)
	preprocessed_text = re.sub(r"\/? ?(사진|그래픽) ?= ?\w+", " ", preprocessed_text)
	preprocessed_text = re.sub(r"\/\w+\s?제공", " ", preprocessed_text)
	# preprocessed_text = re.sub(r"\/사진제공=\S?+?\s+ | \/\s사진?=", " ", preprocessed_text)

	return preprocessed_text


def change_quotation(texts):
	pattern = r"\""
	replacement = "\'"

	processed_text = re.sub(pattern, replacement, texts)

	double_quotation = r"[\u201C\u201D\u2018\u2019]"

	processed_text = re.sub(double_quotation, replacement, processed_text)


	return processed_text


def remove_email(texts):
	"""
    이메일을 제거
    """
	preprocessed_text = ''

	text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", " ", texts).strip()
	if text:
		preprocessed_text = text

	return preprocessed_text


def remove_day(texts):
	"""
    날짜와 관련된 숫자 제거
    """
	pattern = r'\d{4}\.\d{1,2}\.\d{1,2}\.?'

	text = re.sub(pattern, " ", texts)
	return text


def remove_triangle(texts):
	pattern = r'▶\s?.+=?'

	text = re.sub(pattern, " ", texts)
	return text


def remove_parentheses(texts):
	"""
    괄호와 그 안에 있는 내용들을 제거
    """
	pattern = r'[\(\[][^(^[]*[\)\]]'

	processed_text = re.sub(pattern, ' ', texts)

	return processed_text


def remove_copyright(texts):
	"""
    뉴스 내 포함된 저작권 관련 텍스트를 제거합니다.
    """
	re_patterns = [
		r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
		r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
	]
	preprocessed_text = ''

	for re_pattern in re_patterns:
		text = re.sub(re_pattern, " ", texts)
	if text:
		preprocessed_text = text

	return preprocessed_text


def split_sentence(texts):
	sentence_list = texts.split(". ")

	return sentence_list


# 혹시 필요한 것들 모음
def remove_hashtag(texts):
	"""
    해쉬태그(#)를 제거합니다.
    """
	preprocessed_text = ''

	text = re.sub(r"#\S+", " ", texts).strip()
	if text:
		preprocessed_text = text

	return preprocessed_text


def remove_url(texts):
	"""
    URL을 제거합니다.
    주소: www.naver.com`` -> 주소:
    """
	preprocessed_text = ''

	text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", " ", texts).strip()
	text = re.sub(r"pic\.(\w+\.)+\S*", " ", text).strip()

	preprocessed_text = text

	return preprocessed_text


def remove_special_str(texts):
	preprocessed_text = ''

	pattern = r"[◆◇▲▼■●▶◀△▽→←⇒⇔➜➔❯❮]"
	preprocessed_text = re.sub(pattern, " ", texts)
	preprocessed_text = re.sub("\↑", "증가 ", texts)

	return preprocessed_text

def remove_space_dup(texts):
	preprocessed_text = re.sub(r"\s+", " ", texts)

	return preprocessed_text


def all_preprocessing(texts):
	texts = str(texts)
	texts = remove_text(texts)
	texts = remove_press(texts)
	texts = remove_photo_info(texts)
	texts = remove_email(texts)
	texts = remove_copyright(texts)
	texts = remove_day(texts)
	texts = remove_triangle(texts)
	texts = remove_parentheses(texts)
	texts = remove_special_str(texts)

	texts = change_quotation(texts)
	texts = remove_space_dup(texts)

	return texts


def preprocess_dataframe(df):
	df['content_corpus'] = df['content'].apply(all_preprocessing)

	return df


def preprocess_dataframe_to_sentence(df):
	df_sentence = pd.DataFrame(columns=['title', 'date', 'content_sentence'])

	for index, row in df.iterrows():
		title = row['title']
		date = row['date']
		content = row['content']

		content = split_sentence(content)

		for sentence in content:
			l = len(df_sentence)
			sentence = all_preprocessing(sentence)
			new_row = {'title': title, 'date': date, 'content_sentence': sentence}

			new_row = pd.DataFrame(new_row, index=[l])
			df_sentence = pd.concat([df_sentence, new_row], axis=0)

	return df_sentence


if __name__ == "__main__":
	# text = "‘스마트싱스’ 중심의 지속 가능한 일상 전시 에너지 줄이는 비스포크 가전 라인업 선봬[이데일리 김응열 기자] 삼성전자가 29일부터 내달 1일까지 광주광역시 김대중컨벤션센터에서 열리는 ‘2023 국제IoT가전로봇박람회’에 참가해 각종 에너지 절감 기술을 적용한 가전제품 라인업과 솔루션을 대거 소개한다.삼성전자 모델이 29일부터 내달 1일까지 광주광역시 김대중컨벤션센터에서 진행되는 ‘2023국제IoT가전로봇박람회’에 마련된 삼성전자 부스에서 ‘스마트싱스 에너지 세이빙’ 솔루션을 소개하고 있다. 접목 디지털 제어 기술 △스마트싱스 기반 에너지 관리 솔루션 ’스마트싱스 에너지‘의 ’AI 절약모드‘ 등으로 추가적인 에너지 절감이 가능하다. 가령 비스포크 무풍에어컨 갤러리의 에너지 특화 모델은 1등급 최저 기준보다도 냉방 효율이 10% 더 뛰어나다. AI 절약 모드 기능을 활용하면 전력 사용량을 최대 20% 추가 절약할 수 있다.삼성전자 모델이 29일부터 내달 1일까지 광주광역시 김대중컨벤션센터에서 진행되는 ‘2023국제IoT가전로봇박람회’에 마련된 삼성전자 부스에서 스마트싱스 기반의 ‘넷 제로 홈’을 소개하고 있다. (사진=삼성전자)삼성전자는 이번 전시에서 스마트싱스 기반의 ’넷 제로 홈(Net Zero Home)‘으로 에너지 리더십도 강조한다. 넷 제로 홈에서는 태양광 패널로 생산한 에너지를 활용할뿐 아니라 스마트싱스를 이용해 가전제품이나 집안 전체 에너지 사용량을 모니터링하고 줄일 수 있다. 삼성전자는 에너지 절약을 위해 한국전력공사, 서울특별시, 나주시와 협력하는 ’주민 DR(Demand Response)‘ 서비스 사업도 함께 소개했다. 전력거래소나 지방자치단체가 DR 발령 시 자동으로 연동된 삼성전자 제품을 AI 절약 모드로 전환하거나 전원을 끄는 등 전력량을 최소화한다. 이 기능은 에어컨, 냉장고, 세탁기·건조기, 식기세척기, TV 등 총 9종의 삼성전자 가전제품과 파트너사의 스마트 기기까지 지원한다. ’지속 가능한 일상(Everyday Sustainability)‘을 주제로 한 전시 공간에서는 파트너십을 바탕으로 탄생한 자원순환 솔루션을 소개한다. △세탁 과정에서 발생하는 미세 플라스틱 배출 저감을 위해 글로벌 아웃도어 브랜드 파타고니아(Patagonia)와 협업 개발한 ’미세 플라스틱 저감 필터‘ △문승지 디자이너와 업사이클링 패션 브랜드 플리츠마마가 협업해 버려진 페트병, 자투리 원단 등으로 만든 ’제로 에디션(Zero Edition)‘ 의자와 러그 등을 전시하다. 박찬우 삼성전자 생활가전사업부 부사장은 “삼성전자 비스포크 가전은 제품 고유의 기술은 물론 AI와 사물인터넷(IoT)를 접목해 일상을 더욱 풍요롭게 하고 에너지를 절감하는 솔루션을 제시해 왔다”며 “앞으로는 손쉽게 실천할 수 있는 지속 가능 솔루션을 다양하게 선보이며 소비자들이 가치 있는 일상을 경험할 수 있도록 만드는 데에 주력할 것”이라고 말했다.삼성전자 모델이 29일부터 내달 1일까지 광주광역시 김대중컨벤션센터에서 진행되는 ‘2023국제 IoT가전로봇박람회’에 마련된 삼성전자 부스의 ‘지속 가능한 일상’ 주제의 전시 공간에서 자원순환 솔루션을 소개하고 있다. (사진=삼성전자)"
	# text = "/사진제공=한화한화 건설부문은 "
	# text = "윤선희 기자 ="
	# text = "서울 서초구 삼성전자 서초사옥의 모습. 2023.4.7/뉴스1  News1 신웅수 기자 김정은 기자 = "
	text = " ▶공매도 대기자금, 전체 증시 83조·삼전 11조 돌파 ‘역대 최대’=1일 헤럴드경제가 금융투자협회 종합통계포털을 분석한 결과 지난달 26일 기준 삼성전자 한 종목에 대한 일간 대차거래 잔액 규모는 11조2507억원에 달했다. 이는 공매도가 부분 재개된 2021년 5월 이후는 물론, 관련 통계가 집계되기 시작한 2009년 이후 가장 큰 액수다.  공매도 부분 재개 후 7조원대 이하에 머물던 삼성전자에 대한 대차거래 잔액 규모는 지난해 11월 8조원대를 돌파해 9조원선을 터치했다. 올해 첫 거래일(1월 2일) 6조7630억원 규모로 시작한 대차"
	text = all_preprocessing(text)
	# text = remove_day(text)
	# text = remove_press(text)
	print(text)