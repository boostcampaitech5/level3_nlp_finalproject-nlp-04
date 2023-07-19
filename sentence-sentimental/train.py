import torch
import pandas as pd
import sklearn
import random
import numpy as np
import wandb

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import pipeline

from dataset.datasets import SentimentalDataset
from metrics.metrics import compute_metrics

from sklearn.datasets import load_iris # 샘플 데이터 로딩
from sklearn.model_selection import train_test_split

from utils.utils import config_seed
'''
- git clone https://github.com/ukairia777/finance_sentiment_corpus.git 먼저 실행
- readme.md 작성 예정
'''
def train() :
    ## 설정
    SEED = 42
    config_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    ## 모델
    MODEL_NAME = 'klue/roberta-large'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    
    ## 데이터
    data = pd.read_csv('/opt/ml/finance_sentiment_corpus/merged_samsung_filtered.csv')
    def extract_label(json_str):
        data_dict = eval(json_str)  # JSON 문자열을 파이썬 딕셔너리로 변환
        return data_dict["label"]

    # "label" 값을 추출하여 새로운 Series 생성

    temp = train_test_split(data['content_corpus'], data['labels'],
                            test_size=0.2, shuffle=True, stratify=data['labels'], # label에 비율을 맞춰서 분리
                            random_state=SEED)
    
    sentence_train, sentence_val, label_train, label_val = temp

    max_length=200
    stride=10
    ## TODO 임의의 값으로 차후 수정
    train_encoding = tokenizer(sentence_train.tolist(), ## pandas.Series -> list
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
    
    ## 학습
    training_args = TrainingArguments(
        output_dir = './outputs',
        logging_steps = 50,
        num_train_epochs = 2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate = 5e-6,
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
    
    
    
def sweep_train(config=None) :
    ## wandB
    run = wandb.init(config=config)
    config = wandb.config
    run.name = f"model: klue/roberta-large' / batch_size: {config.batch_size} / lr: {config.lr} / warmup: {config.warmup_step}"
    
    ## 설정
    SEED = 42
    config_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    ## 모델
    MODEL_NAME = 'klue/roberta-large'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    ## 데이터
    data = pd.read_csv('/opt/ml/finance_sentiment_corpus/merged_samsung_filtered.csv')
    def extract_label(json_str):
        data_dict = eval(json_str)  # JSON 문자열을 파이썬 딕셔너리로 변환
        return data_dict["label"]

    # "label" 값을 추출하여 새로운 Series 생성

    temp = train_test_split(data['content_corpus'], data['labels'],
                            test_size=0.2, shuffle=True, stratify=data['labels'], # label에 비율을 맞춰서 분리
                            random_state=SEED)
    
    sentence_train, sentence_val, label_train, label_val = temp

    max_length=200
    stride=10
    ## TODO 임의의 값으로 차후 수정
    train_encoding = tokenizer(sentence_train.tolist(), ## pandas.Series -> list
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
    
    ## 학습
    training_args = TrainingArguments(
        output_dir = './outputs',
        logging_steps = 50,
        num_train_epochs = 2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate = 5e-6,
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

if __name__ == '__main__':
    train()
