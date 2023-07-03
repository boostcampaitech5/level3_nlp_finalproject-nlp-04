from typing import Union
import urllib.request

from selectolax.parser import HTMLParser, Node

from fastapi import FastAPI

def get_search_nodes(code: str, time_range: int) -> list[Node]:
    """주어진 code와 time_tange에 대해 NAVER 검색 결과에 대한 node를 반환받습니다. 

    Args:
        code (str): 종목 코드 
        time_range (int): 시간 범위(시 단위, 범위 해제 시 0으로 설정)

    Returns:
        list[Node]: 검색된 결과에 대한 node. 
    """
    dict_hr2code = {0: "0", 1: "7", 2: "8", 3: "9", 4: "10", 5: "11", 6: "12"}
    url = f"https://search.naver.com/search.naver?where=news&query={code}&sm=tab_opt&sort=1&photo=0&field=0&pd={dict_hr2code[time_range]}&ds=&de=&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Add%2Cp%3Aall&is_sug_officeid=0"
    
    with urllib.request.urlopen(url) as response:
        html = response.read()

    tree = HTMLParser(html)
    nodes = tree.css('a[class="info"]')

    return nodes

def get_news_body(url: str) -> dict:
    """주어진 뉴스 기사 URL에서 기사 제목과 본문을 가져옵니다. 

    Args:
        url (str): 뉴스 기사 URL

    Returns:
        dict: 뉴스 기사에 대한 제목과 본문
    """
    with urllib.request.urlopen(url) as response:
        html = response.read()

    tree = HTMLParser(html)

    title = tree.css_first('.media_end_head_headline').text()
    body = tree.css_first('.go_trans._article_content').text()

    return {"title": title, "body": body}

# for node in get_search_nodes("005930", 0):
#     attr = node.attributes
#     print(get_news_body(attr['href']))

app = FastAPI()

@app.get("/{stock_code}")
def get_news_docs(stock_code: str, hr: Union[int, None]=0) -> dict:
    """HTTP GET에서 stock_code과 hr을 이용하여 JSON 형태의 기사 제목과 본문을 반환한다. 

    Args:
        stock_code (str): 주가 종목 코드
        hr (Union[int, None], optional): 기사 검색 시간 범위(시 단위, 범위 해제 시 0으로 설정). 기본값 0. 

    Returns:
        dict: 뉴스 기사 제목과 본문
    """
    dict_news_docs = {}

    for idx, node in enumerate(get_search_nodes(stock_code, hr)):
        attr = node.attributes
        dict_news_docs[idx] = get_news_body(attr['href'])
        
    return dict_news_docs