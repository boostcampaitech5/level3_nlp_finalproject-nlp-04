import sys

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any, Tuple

from datetime import datetime
import pandas as pd

sys.path.append(str(Path.home().joinpath("level3_nlp_finalproject-nlp-04")))
from keyword_extractor.model import KeyBert

app = FastAPI()

@app.get("/")
def hello_word():
	return {"hello": "world"}


class Item(BaseModel):
    titles: Union[str, List[str]] = []
    contents: Union[str, List[str]] = []
    dates: Union[str, List[str]] = []

class Parameter(BaseModel):
    stop_words: Optional[List[str]] = None
    top_k: Optional[int] = 5
    diversity: Optional[float] = 0.7
    min_df: Optional[int] = 1
    candidate_frac: Optional[float] = 0.3
    vectorizer_type: Optional[str] = "tfidf"
    tag_type: Optional[str] = "okt"


def df_transform(data_input: Item):
    # DataFrame으로 변환하는 함수
    df = pd.DataFrame({'title': data_input.titles,
                    	'content': data_input.contents,
                        'date': data_input.dates})
    return df

def process_input_data(data_input: Item):
    # 단일 문자열을 리스트로 변환하는 함수
    def convert_to_list(item):
        if isinstance(item, str):
            return [item]
        return item

    # data_input의 필드에 대해 타입 변환 수행
    data_input.titles = convert_to_list(data_input.titles)
    data_input.contents = convert_to_list(data_input.contents)
    data_input.dates = convert_to_list(data_input.dates)

    return data_input

def get_model(model_number):
    # model 반환 함수
    if model_number == "1":
        model = KeyBert("jhgan/ko-sroberta-multitask")
    elif model_number == "2":
        model = KeyBert("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    elif model_number == "3":
        model = KeyBert("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    return model


@app.post(
    "/keywordExtraction/{model_number}",
    name="keyword extraction",
    description="뉴스 기사에서 키워드를 추출하는 요약하는 API 입니다. `model_number`에는 `1`, `2`,`3` 중 하나를 선택하시면 됩니다.\
        기본 모델 목록 1: jhgan/ko-sroberta-multitask, 2:snunlp/KR-SBERT-V40K-klueNLI-augSTS, 3:sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
async def keywordExtraction(model_number: str, data_input: Item = Depends(process_input_data), parameter: Parameter = Depends()):
    if not model_number in ["1", "2","3"]:
        raise HTTPException(status_code=404, detail="'model_number' argument invalid! Type model name. \n Model: 1, 2, 3")

    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        df = df_transform(data_input)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    model = get_model(model_number)

    result = model.extract_keywords(df['content'].tolist(),
                                    stop_words = parameter.stop_words if parameter.stop_words else [],
                                    top_k = parameter.top_k,
                                    diversity = parameter.diversity,
                                    candidate_frac = parameter.candidate_frac,
                                    vectorizer_type = parameter.vectorizer_type,
                                    tag_type = parameter.tag_type,
                                    )
    return result, parameter