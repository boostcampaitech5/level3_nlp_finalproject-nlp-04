import torch
import random
import numpy as np
import re
import json


def config_seed(SEED) :
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# 오류로 라벨링이 되어있지 않은 row를 제거
def remove_idx_row(data) : 
    patterns = [r'idx\s*:?\s*.+?', r'라벨링\s*:?\s*.+?']
    
    for pattern in patterns :
        mask = data['labels'].str.match(pattern)
        data = data.drop(data[mask].index)

    return data

# ['labels'] 전처리
def preprocessing_label(json_str) :
    json_str = re.sub(r"^.*### 출력\s?:?\s?\n?\s?", "", str(json_str))
    # json_str= json_str.replace("\"", "'")
    json_str = json_str.replace("'", "\\'") # python interpreter에서 한번, json에서 한번
    
    return json_str


# title과 content_corpus에서 원하는 문장 추출
# TODO: 가장 적잘한 값으로 수정
def extract_sentences(text):
    sentences = text.split('. ')
    if len(sentences) >= 5 :
        return '. '.join([sentences[0], sentences[1], sentences[-2], sentences[-1]])
    else :
        return '. '+text
    
# json 형태로 되어있지 않은 형태를 최대한 json 형태로 바꿔주고 label 추출 
def extract_label(json_str) :
    json_str = json_str.replace("'", "\"")
    try:
        json_data = json.loads(json_str)

    except json.JSONDecodeError as e:
        if json_str[-2:] == '.}' :
            json_str = json_str[:-2] + ".\"}"
        elif json_str[-1] == "\"" :
            json_str = json_str + "}"
        else:
            json_str += "\"}"
    
    # 마지막에도 되지 않으면 None으로 처리
    try:
        data_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None

    return data_dict["label"]

def gpt_preprocessing_labels(data) :
    data = remove_idx_row(data)
    data["labels"] = data["labels"].apply(preprocessing_label)
    data['labels'] = data["labels"].apply(extract_label)
    data['content_corpus_company'] = data.apply(lambda row: '이 기사는 [COMPANY]'+ str(row["company"]) +'[/COMPANY]에 대한 기사. [SEP]'+ extract_sentences(row['title']) + ' ' + extract_sentences(row['content_corpus']), axis=1) #TODO: 학습에 들어가는 부분이기 때문에 조정 필수
    data['labels'] = data['labels'].map({'부정':0, '긍정':1})
    data = data[["title", "date", "content_corpus", "labels", "content_corpus_company"]]
    data = data[data['labels'].notna()] # 최후에 처리 되지 않은 None row 제거
    
    return data