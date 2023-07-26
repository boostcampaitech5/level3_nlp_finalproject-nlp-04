from datetime import datetime
from pydantic import BaseModel

class Stock(BaseModel):
    ticker: str = "005930"
    sufix: str = ".KS"
    date_start: datetime
    date_end: datetime
    frequency: str = "day"
    option: str = "now"
