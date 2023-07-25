import torch
import pandas as pd
import sklearn
import random
import numpy as np
import wandb

from transformers import AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from transformers import DebertaV2ForSequenceClassification

from dataset.datasets import SentimentalDataset
from metrics.metrics import compute_metrics

from sklearn.datasets import load_iris # 샘플 데이터 로딩
from sklearn.model_selection import train_test_split

from utils.utils import config_seed

import json
import re
'''
- git clone https://github.com/ukairia777/finance_sentiment_corpus.git 먼저 실행
- readme.md 작성 예정
'''
def train() :
    # 설정
    SEED = 42
    config_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    # 모델
    MODEL_NAME = 'klue/roberta-large'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    special_tokens_dict = {'additional_special_tokens': ['[COMPANY]','[/COMPANY]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))


    # 데이터
    data = pd.read_csv("/opt/ml/finance_sentiment_corpus/merged/merged_all.csv")

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
    
    data = remove_idx_row(data)
    data["labels"] = data["labels"].apply(preprocessing_label)
    data['labels'] = data["labels"].apply(extract_label)
    data['content_corpus_company'] = data.apply(lambda row: '이 기사는 [COMPANY]'+ str(row["company"]) +'[/COMPANY]에 대한 기사. [SEP]'+ extract_sentences(row['title']) + ' ' + extract_sentences(row['content_corpus']), axis=1) #TODO: 학습에 들어가는 부분이기 때문에 조정 필수
    data['labels'] = data['labels'].map({'부정':0, '긍정':1})
    data = data[["title", "date", "content_corpus", "labels", "content_corpus_company"]]
    data = data[data['labels'].notna()] # 최후에 처리 되지 않은 None row 제거
    
    
    # "labels" 값을 추출하여 새로운 Series 생성
    dataset = train_test_split(data['content_corpus_company'], data['labels'],
                            test_size=0.2, shuffle=True, stratify=data['labels'], # label에 비율을 맞춰서 분리
                            random_state=SEED)


    sentence_train, sentence_val, label_train, label_val = dataset


    max_length=500
    stride=0
    
    # TODO 임의의 값으로 차후 수정
    train_encoding = tokenizer(sentence_train.tolist(), # pandas.Series -> list
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                ##
                                max_length=max_length,
                                stride=stride,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=False
                                )

    val_encoding = tokenizer(sentence_val.tolist(),
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            ##
                            max_length=max_length,
                            stride=stride,
                            return_overflowing_tokens=True,
                            return_offsets_mapping=False
                            )

    train_set = SentimentalDataset(train_encoding, label_train.reset_index(drop=True))
    val_set = SentimentalDataset(val_encoding, label_val.reset_index(drop=True))
    
    
    # 학습
    # run = wandb.init(project="final_sentimental", entity="nlp-10")
    # run.name = f"model: {MODEL_NAME} / batch_size: {per_device_train_batch_size} / lr: {learning_rate}"
    logging_steps = 200
    num_train_epochs = 3
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    learning_rate = 5e-6
    
    training_args = TrainingArguments(
        output_dir = './outputs',
        logging_steps = logging_steps,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        learning_rate = learning_rate,
        evaluation_strategy="epoch", 
        fp16=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics
    )

    print('---train start---')
    trainer.train()
    # wandb.finish()
    
    
def sweep_train(config=None) :
    ## wandb
    run = wandb.init(config=wandb.config)
    run.name = f"max_length: {wandb.config.max_length}"

    # 설정
    SEED = 42
    config_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    # 모델
    MODEL_NAME = 'klue/roberta-large'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    special_tokens_dict = {'additional_special_tokens': ['[COMPANY]','[/COMPANY]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))


    # 데이터
    data = pd.read_csv("/opt/ml/finance_sentiment_corpus/merged/merged_all.csv")

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
    
    data = remove_idx_row(data)
    data["labels"] = data["labels"].apply(preprocessing_label)
    data['labels'] = data["labels"].apply(extract_label)
    data['content_corpus_company'] = data.apply(lambda row: '이 기사는 [COMPANY]'+ str(row["company"]) +'[/COMPANY]에 대한 기사. [SEP]'+ extract_sentences(row['title']) + ' ' + extract_sentences(row['content_corpus']), axis=1) #TODO: 학습에 들어가는 부분이기 때문에 조정 필수
    data['labels'] = data['labels'].map({'부정':0, '긍정':1})
    data = data[["title", "date", "content_corpus", "labels", "content_corpus_company"]]
    data = data[data['labels'].notna()] # 최후에 처리 되지 않은 None row 제거
    
    
    # "labels" 값을 추출하여 새로운 Series 생성
    dataset = train_test_split(data['content_corpus_company'], data['labels'],
                            test_size=0.2, shuffle=True, stratify=data['labels'], # label에 비율을 맞춰서 분리
                            random_state=SEED)


    sentence_train, sentence_val, label_train, label_val = dataset


    max_length=wandb.config.max_length #원본:500
    stride=0
    
    # TODO 임의의 값으로 차후 수정
    train_encoding = tokenizer(sentence_train.tolist(), # pandas.Series -> list
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                ##
                                max_length=max_length,
                                stride=stride,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=False
                                )

    val_encoding = tokenizer(sentence_val.tolist(),
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            ##
                            max_length=max_length,
                            stride=stride,
                            return_overflowing_tokens=True,
                            return_offsets_mapping=False
                            )

    train_set = SentimentalDataset(train_encoding, label_train.reset_index(drop=True))
    val_set = SentimentalDataset(val_encoding, label_val.reset_index(drop=True))
    
    
    # 학습
    # run = wandb.init(project="final_sentimental", entity="nlp-10")
    # run.name = f"model: {MODEL_NAME} / batch_size: {per_device_train_batch_size} / lr: {learning_rate}"
    logging_steps = 200
    num_train_epochs = 3
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    learning_rate = 5e-6
    
    training_args = TrainingArguments(
        output_dir = './outputs',
        logging_steps = logging_steps,
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        learning_rate = learning_rate,
        evaluation_strategy="epoch", 
        fp16=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics
    )

    print('---train start---')
    trainer.train()
    # wandb.finish()

if __name__ == '__main__':
    do_train = False
    if do_train:
        train()
    else:
        wandb_config = {
            'method': 'grid',
            'parameters':
            {
                'max_length': {'values' : [500, 250]},
            }
        }
        sweep_id = wandb.sweep(sweep=wandb_config,
                            project='sweep_merged_all'
                            )
        wandb.agent(sweep_id, function=sweep_train)