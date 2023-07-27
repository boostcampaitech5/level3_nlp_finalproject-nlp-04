import sys

import pendulum

from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any, Tuple


from datetime import datetime
import pandas as pd


sys.path.append(str(Path.home().joinpath("level3_nlp_finalproject-nlp-04")))
from keyword_extractor.model import KeyBert, KeyExtract

app = FastAPI()
tz = pendulum.timezone("Asia/Seoul")

@app.get("/")
def hello_word():
	return {"hello": "world"}


class Item(BaseModel):
    titles: Union[str, List[str]]
    contents: Union[str, List[str]]

class Parameter(BaseModel):
    stop_words: Optional[Union[str, List[str]]] = None
    top_k: Optional[int] = 5
    diversity: Optional[float] = 0.7
    min_df: Optional[int] = 1
    candidate_frac: Optional[float] = 0.3
    vectorizer_type: Optional[str] = "tfidf"
    tag_type: Optional[str] = "okt"



def get_model(model_number):
    # model 반환 함수
    if model_number == "1":
        model = KeyBert("jhgan/ko-sroberta-multitask")
    elif model_number == "2":
        model = KeyBert("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    elif model_number == "3":
        model = KeyBert("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    elif model_number == "4":
        model = KeyExtract()
    return model


@app.post(
    "/keywordExtraction/{model_number}",
    name="keyword extraction",
    description="뉴스 기사에서 키워드를 추출하는 요약하는 API 입니다. `model_number`에는 `1`, `2`, `3`, `4` 중 하나를 선택하시면 됩니다.\
        기본 모델 목록 KeyBert 응용 (1:sroberta, 2:SBERT, 3:paraphrase-multilingual), 4: 자체 모델"
)
async def keywordExtraction(model_number: str, data_input: Item, parameter: Parameter = Depends()):
    if not model_number in ["1", "2","3", "4"]:
        raise HTTPException(status_code=404, detail="'model_number' argument invalid! Type model name. \n Model: 1, 2, 3, 4")

    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")

    model = get_model(model_number)
    
    result = model.extract_keywords(docs=data_input.contents,
                                    titles=data_input.titles,
                                    stop_words = parameter.stop_words if parameter.stop_words else [],
                                    top_k = parameter.top_k,
                                    diversity = parameter.diversity,
                                    candidate_frac = parameter.candidate_frac,
                                    vectorizer_type = parameter.vectorizer_type,
                                    tag_type = parameter.tag_type,
                                    )
    return result, parameter
