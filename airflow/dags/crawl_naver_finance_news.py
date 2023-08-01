import sys
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

sys.path.append('/opt/ml/level3_nlp_finalproject-nlp-04/utils')
from crawling_utils import crawl_finance_news

tz = pendulum.timezone("Asia/Seoul")
default_args = {
        'owner': 'yungi',
        'depends_on_past' : False,
        'start_date' : datetime(2023, 7, 18, 15), # UTC 시간 -> KST +9 시간 하면 7월 19일부터
        'retires' : 1,
        'retry_delay' : timedelta(minutes=5),
        }
def print_date(timestr):
	print("***** Execution Time *****:",timestr)

with DAG(
	dag_id="crawl_naver_finance_news_v0.2.1",
	user_defined_macros={'local_dt' : lambda execution_date: execution_date.in_timezone(tz).strftime("%Y-%m-%d")}, # KST시간으로 Timezone 변경(+9시간) 매크로
	default_args=default_args,
	schedule_interval="0 15,21,3,9 * * *", # 크롤링 어느정도 끝나고 현재로 오면, 6시간마다 크롤링하도록(UTC시간 기준시작, 15부터 해야지 KST=다음날 0시)
	# schedule_interval="0 0 * * *", # 과거일자 크롤링할 땐 그냥 하루단위로 작업하도록
	tags = ['crawl']
) as dag:
	execution_date = "{{ local_dt(execution_date) }}" # 실제 실행 시간은 KST기준(+9시간)으로 수행하도록
	# execution_date = "{{ ds }}"
	python_task_jinja = PythonOperator(
		task_id="crawl_naver_finance_news",
		python_callable=crawl_finance_news,
		op_kwargs={'cur_time' : execution_date}
	)
	print_date_jinja = PythonOperator(
		task_id="print_date",
		python_callable=print_date,
		op_args=[execution_date]
	)

	print_date_jinja >> python_task_jinja

