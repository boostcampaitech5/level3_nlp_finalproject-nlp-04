from payload import Stock

import os

from fastapi import FastAPI
from fastapi import HTTPException

import pandas as pd
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

app = FastAPI()

@app.get("/")
def read_root():
    """API 상태 확인용 API. 

    Returns:
        dict: API의 status를 반환한다. 
    """

    return {"status": True}


@app.post("/stock/")
def read_item(stock: Stock):
    """주어진 옵션을 이용하여 DB에 주가 종가를 insert 하는 API. 

    Args:
        stock (Stock): 주가를 받아오기 위해 사용되는 옵션. 

    Raises:
        HTTPException: 주어진 'option'이 올바르지 않은 경우. 

    Returns:
        dict: 모두 완료된 경우 결과 반환. 
    """

    if stock.option == "now":
        df = pdr.get_data_yahoo(stock.ticker + stock.sufix, stock.date_start, stock.date_end)
    elif stock.option == "range":
        df = pdr.get_data_yahoo(stock.ticker + stock.sufix, stock.date_start, stock.date_end)
    else:
        raise HTTPException(status_code=404, detail="'option' argument invalid! Choose 'now' or 'range'. ")

    for idx, item in df.iterrows():
        data = {"ticker": stock.ticker, "price": int(item["Close"]), "date": str(idx)}

        data, count = supabase\
                    .table('price')\
                    .insert(data)\
                    .execute()

    return {"data": data, "count": count}
