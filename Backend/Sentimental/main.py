from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

MODEL_PATH = "/opt/ml/outputs/klue/roberta-large_merged-4_100-26_100"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
special_tokens_dict = {'additional_special_tokens': ['[COMPANY]','[/COMPANY]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model.resize_token_embeddings(len(tokenizer))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def extract_sentences_token(input_dict, pad_token_id):
    '''
    512 토큰 초과라면 input_ids, token_type_ids, attention_maks를
    앞 128, 뒤 384으로 분리합니다.
    이하라면 뒤는 pad 토큰으로 채웁니다.

    사용 방법:
      train_encoding = tokenizer(text)
      train_encoding = extract_sentences_token(train_encoding, tokenizer.pad_token_id)
    '''
    new = {}
    batch_size = len(input_dict['input_ids'])
    new['input_ids'] = torch.ones(batch_size, 512, dtype=int)
    new['token_type_ids'] = torch.ones(batch_size, 512, dtype=int)
    new['attention_mask'] = torch.ones(batch_size, 512, dtype=int)
    # batch_size, 512
    for i in range(batch_size):
        a = input_dict['input_ids'][i]
        a = a[a != pad_token_id]
        length = len(a)
        if length > 512:
            left, right = 1, 3
            a = torch.cat((a[:128*left], a[-128*right:]), dim=0)
            new['input_ids'][i] = a
            new['token_type_ids'][i] = input_dict['token_type_ids'][i][:512]
            new['attention_mask'][i] = input_dict['attention_mask'][i][:512]
        else:
            new['input_ids'][i] = input_dict['input_ids'][i][:512]
            new['token_type_ids'][i] = input_dict['token_type_ids'][i][:512]
            new['attention_mask'][i] = input_dict['attention_mask'][i][:512]
    return new

def predict_sentiment(text: List[str]) -> List[str]:
    answer = []
    model.eval()

    loader = torch.utils.data.DataLoader(dataset=text, batch_size=32, shuffle=False)
    with torch.no_grad() :
        for batch_text in loader:
            temp = tokenizer(
                batch_text,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=3000, # 충분히 커서 모두 토큰화할 길이
                )
            temp = extract_sentences_token(temp, tokenizer.pad_token_id)
            if torch.cuda.is_available():
                temp = {key: value.to(device) for key, value in temp.items()}
            predicted_label = model(**temp)

            results = torch.nn.Softmax(dim=-1)(predicted_label.logits)
            for result in results :
                if result[0]>=result[1] :
                    answer.append("부정")
                else :
                    answer.append("긍정")
                    
    return answer
    
class FinanaceSentiment(BaseModel):
    corpus_list: list = []
    company_list: list = []
    

@app.post("/classify_sentiment", description="문장의 감정을 분류합니다.")
async def classify_sentiment(finance: FinanaceSentiment):
    # 입력으로 받은 텍스트를 모델로 예측합니다.
    input = []
    for corpus, company in zip(finance.corpus_list, finance.company_list) :
        input.append(f"이 기사는[COMPANY]{company}[/COMPANY]에 대한 기사야. {corpus}")
        
    predictions = predict_sentiment(input)
    
    # 결과를 반환합니다.
    result = {
        "sentiment": predictions
    }
    
    return predictions