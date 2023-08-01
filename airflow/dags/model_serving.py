import sys
import pendulum
import json
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
from supabase import Client, create_client
from collections import Counter, defaultdict


sys.path.append('/opt/ml/level3_nlp_finalproject-nlp-04')
from utils.secrets import Secrets
from utils.preprocessing import *

tz = pendulum.timezone("Asia/Seoul")
default_args = {
        'owner': 'yungi',
        'depends_on_past' : False,
        'start_date' : datetime(2023, 7, 30, 15), # UTC 시간 -> KST +9 시간 하면 7월 19일부터
        'retires' : 1,
        'retry_delay' : timedelta(minutes=5),
        }
url = Secrets.url
key = Secrets.key

supabase: Client = create_client(url, key)


def get_news(cur_time,content_length_thr):
	print("**** Current Time **** :",cur_time)
	cur_time = datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S")
	stock_name = supabase.table("ticker").select("name").eq("ticker", "005930").execute().data[0]["name"]
	res = supabase.table("news").select("*").eq("company", stock_name).filter("date", "gt",(cur_time - timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')
																		  ).filter("date", "lt",cur_time.strftime('%Y-%m-%d %H:%M:%S')).execute().data
	# 짧은 뉴스 제거
	for cont in res:
		if len(cont['content']) <= content_length_thr:
			res.remove(cont)
	return res

def get_sentimental(text):
	articles = eval(text)
	res_sentimental = requests.post("http://115.85.183.242:30007/classify_sentiment",
				  data=json.dumps({
						  "corpus_list" : [all_preprocessing(news['content']) for news in articles],
						  "company_list" : [news['company'] for news in articles],
				  })).content
	return json.loads(res_sentimental)

def get_keywords(text):
	articles = eval(text)
	keywords = json.loads(requests.post("http://118.67.133.198:30008/keywordExtraction/5",
										data=json.dumps({
											"data_input" : {
												"titles" : articles['title'] if type(articles) == dict else [all_preprocessing(news['title']) for news in articles ],
												"contents" : articles['content'] if type(articles) == dict else [all_preprocessing(news['content']) for news in articles],
												"dates" : articles['date'] if type(articles) == dict else [news['date'] for news in articles],
									  		},
											"stop_words" : ["삼성", "전자", "삼성전자"]}),
										params={"vectorizer_type" : "tfidf"}
										).content)
	return keywords

def get_summary(articles, **config):
	pp = requests.post(f"http://118.67.143.119:30007/api/v1/summerize/{config['model_name']}",
			  data=json.dumps({
				  "prompt": "아래의 뉴스를 사실에 입각하여 맥락의 내용을 요약해줘.",
				  "title" : articles['title'],
				  "contents" : all_preprocessing(articles['content']),
				  "num_split" : config["num_split"],
				  "options": {
					  "do_sample": True,
					  "early_stopping": "never",
					  "eos_token_id": 2,
					  "max_new_tokens": config["max_new_tokens"],
					  "no_repeat_ngram_size": config["no_repeat_ngram_size"],
					  "temperature" : config["temperature"],
					  "top_k": config["top_k"],
					  "top_p": config["top_p"],
				  }
			  }
			  )
			).content
	return json.loads(pp)['arr_summerized']

def filter_push_keyword_sentimental(news, keywords, sentimentals, cur_time, score_thr=0.2):
	pos_cnt = Counter()
	neg_cnt = Counter()
	pos_keyword_id = defaultdict(list)
	neg_keyword_id = defaultdict(list)

	news = eval(news)
	keywords = eval(keywords)
	sentimentals = eval(sentimentals)

	for i, sent in enumerate(sentimentals):
		if sent == "긍정":
			for keyword, score in keywords[0][i]:
				if score > score_thr:
					pos_cnt[keyword] += 1
					pos_keyword_id[keyword].append(news[i]['id'])
		elif sent == "부정":
			for keyword, score in keywords[0][i]:
				if score > score_thr:
					neg_cnt[keyword] += 1
					neg_keyword_id[keyword].append(news[i]['id'])

	for key in set(list(pos_cnt.keys()) + list(neg_cnt.keys())):
		pos_c = pos_cnt.get(key, 0)
		neg_c = neg_cnt.get(key, 0)
		pos_k = pos_keyword_id.get(key, [])
		neg_k = neg_keyword_id.get(key, [])
		supabase.table("keywords").insert({"keyword": key,
										   "count": pos_c + neg_c,
										   "create_time": datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S").strftime(
											   "%Y-%m-%d %H:%M:%S"),
										   "pos_cnt": pos_c,
										   "neg_cnt": neg_c,
										   "summary_id": {
											   "pos_news": pos_k,
											   "neg_news": neg_k
										   }
										   }).execute()
	return {"pos_cnt" : pos_cnt, "neg_cnt" : neg_cnt, "pos_keyword_id" : pos_keyword_id, "neg_keyword_id" : neg_keyword_id}

def push_summary(text):
	config = {
		"model_name" : "t5",
		"num_split" : 9999,
		"max_new_tokens" : 256,
		"no_repeat_ngram_size" : 8,
		"temperature" : 0.08,
		"top_k" : 50,
		"top_p" : 0.98,
	}
	articles = eval(text)
	for i, news in enumerate(articles):
		summary = get_summary(news, **config)[0]
		supabase.table("news_summary").insert({"origin_id": articles[i]['id'], "summarization": summary}).execute()

with DAG(
	dag_id="model_serving_v0.1.0",
	user_defined_macros={'local_dt' : lambda execution_date: execution_date.in_timezone(tz).strftime("%Y-%m-%d %H:%M:%S")}, # KST시간으로 Timezone 변경(+9시간) 매크로
	default_args=default_args,
	schedule_interval="0 16,22,4,10 * * *", # 크롤링 어느정도 끝나고 현재로 오면, 6시간마다 크롤링하도록(UTC: 15 = KST:다음날 0시)
	# schedule_interval="0 0 * * *", # 과거일자 크롤링할 땐 그냥 하루단위로 작업하도록
	catchup=True,
	tags = ['serving']
) as dag:
	execution_date = "{{ local_dt(execution_date) }}"  # 실제 실행 시간은 KST기준(+9시간)으로 수행하도록
	# execution_date = "{{ ds }}"

	get_news = PythonOperator(
		task_id="get_news",
		python_callable=get_news,
		op_kwargs={'cur_time': execution_date,
				   "content_length_thr" : 400}
	)
	get_keywords = PythonOperator(
		task_id="get_keywords",
		python_callable=get_keywords,
		op_args=["{{task_instance.xcom_pull(task_ids='get_news')}}"]
	)
	get_sentimental = PythonOperator(
		task_id="get_sentimental",
		python_callable=get_sentimental,
		op_args=["{{task_instance.xcom_pull(task_ids='get_news')}}"]
	)
	get_push_summary = PythonOperator(
		task_id="get_push_summary",
		python_callable=push_summary,
		op_args=["{{task_instance.xcom_pull(task_ids='get_news')}}"],
	)
	filter_push_keyword_sentimental = PythonOperator(
		task_id="filter_keyword_sentimental",
		python_callable=filter_push_keyword_sentimental,
		op_kwargs={"news": "{{task_instance.xcom_pull(task_ids='get_news')}}",
				   "keywords": "{{task_instance.xcom_pull(task_ids='get_keywords')}}",
				   "sentimentals": "{{task_instance.xcom_pull(task_ids='get_sentimental')}}",
				   "cur_time" : execution_date,
				   "score_thr" : 0.2}
	)
	get_news >> [get_keywords, get_sentimental, get_push_summary]  >> filter_push_keyword_sentimental