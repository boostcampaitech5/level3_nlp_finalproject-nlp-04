import sys

import pendulum

from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from supabase import create_client, Client

sys.path.append(str(Path.home().joinpath("level3_nlp_finalproject-nlp-04")))
from keyword_extractor.model import KeyBert
from utils.secrets import Secrets

app = FastAPI()
tz = pendulum.timezone("Asia/Seoul")

@app.get("/")
def hello_word():
	return {"hello": "world"}

@app.get("/get_db", description="Stock Code, Company Name 가져오기")
async def get_stock():
	url = Secrets.url
	key = Secrets.key

	supabase: Client = create_client(url, key)

	res = supabase.table("ticker").select("*").execute().data
	return res

@app.get("/get_db/{stock_code}", description="종목 코드를 이용해서, DB에서 데이터 가져오기")
async def get_db(stock_code: str,
				 start_date: Optional[str] = (datetime.now(tz=tz) - timedelta(days=7)).strftime('%Y%m%d%H%M%S'),
				 end_date: Optional[str] = datetime.now(tz=tz).strftime('%Y%m%d%H%M%S')):
	url = Secrets.url
	key = Secrets.key

	supabase: Client = create_client(url, key)


	stock_name = supabase.table("ticker").select("name").eq("ticker", stock_code).execute().data[0]["name"]
	res = supabase.table("news").select("*").eq("company", stock_name).filter("date", "gt",
																			  datetime.strptime(start_date, "%Y%m%d%H%M%S").strftime('%Y-%m-%d %H:%M:%S')
																			  ).filter("date", "lt",
																					   datetime.strptime(end_date, "%Y%m%d%H%M%S").strftime('%Y-%m-%d %H:%M:%S')).execute().data
	return res