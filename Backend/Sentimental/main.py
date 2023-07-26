from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

MODEL_PATH = "/opt/ml/input/model-roberta_large-sota_trainer"
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

@app.get("/")
def hello_world():
    return {"hello": "world"}


def predict_sentiment(text):
    model.eval()
    with torch.no_grad() :
        temp = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            ##
            max_length=100,
            # stride=stride,
            # return_overflowing_tokens=True,
            return_offsets_mapping=False
            )
        

        predicted_label = model(input_ids=temp['input_ids'],
                                token_type_ids=temp['token_type_ids'])
    
    print(predicted_label)
    
    results = []
    results = torch.nn.Softmax(dim=-1)(predicted_label.logits)


    answer = []
    print(results)
    for result in results :
        if result[0]>=result[1] :
            answer.append("부정")
        
        else :
            answer.append("긍정")
    
    return answer
    
class FinanaceSentiment(BaseModel):
    corpus_list: list = []
    title: str = "title"
    company: str = "삼성전자"
    result: Optional[List]

@app.post("/classify_sentiment", description="문장의 감정을 분류합니다.")
async def classify_sentiment(finance: FinanaceSentiment):
    # 입력으로 받은 텍스트를 모델로 예측합니다.
    predictions = predict_sentiment(finance.corpus_list)
    
    # 결과를 반환합니다.
    result = {
        "title": finance.title,
        # "input_text": finance.corpus,
        "sentiment": predictions
    }
    
    return predictions