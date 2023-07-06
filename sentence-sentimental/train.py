import torch
import pandas as pd
import sklearn
import random
import numpy as np

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
    SEED = 42

    config_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # 모델과 tokenizer 설정
    MODEL_NAME = 'klue/roberta-base'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # data 받기 및 train/test 나누기
    data = pd.read_csv('/opt/ml/finance_sentiment_corpus/finance_data.csv')
    data['labels'] = data['labels'].map({'negative':0, 'neutral':1, 'positive':2})

    sentence_train, sentence_test, label_train, label_test = train_test_split(data['kor_sentence'], data['labels'],
                                                                                test_size=0.2, 
                                                                                shuffle=True, stratify=data['labels'], # label에 비율을 맞춰서 분리
                                                                                random_state=SEED)

    train_encoding = tokenizer(sentence_train.tolist(),
                                return_tensors='pt',
                                padding=True,
                                truncation=True
                                )

    test_encoding = tokenizer(sentence_test.tolist(),
                            return_tensors='pt',
                            padding=True,
                            truncation=True
                            )

    train_set = SentimentalDataset(train_encoding, label_train.reset_index(drop=True))
    test_set = SentimentalDataset(test_encoding, label_test.reset_index(drop=True))

    training_args = TrainingArguments(
        output_dir = './outputs',
        logging_steps = 50,
        num_train_epochs = 1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=train_set,
        compute_metrics=compute_metrics
    )

    print('---train start---')
    trainer.train()

    print('---evaulate start---')
    trainer.evaluate()

    print('---inference start---')

    ## 방법 1
    model = model.to('cpu') ## 주의
    my_text = '삼성전자, 올해부터 다운턴 끝나고 매출 상승 시작할 듯'
    classifier = pipeline("sentiment-analysis", model=model,
                        tokenizer=tokenizer)
    inference_output = classifier(my_text)
    print(inference_output)
    # [{'label': 'LABEL_2', 'score': 0.8627877831459045}]

    ## 방법 2
    my_text = '삼성전자, 올해부터 다운턴 끝나고 매출 상승 시작할 듯'
    model.eval()
    with torch.no_grad() : # 기울기 그래프가 안 생겨서 속도가 빨라진다.
        temp = tokenizer(
            my_text,
            return_tensors='pt',
            padding=True,
            truncation=True
            )
        predicted_label = model(**temp)
        print(torch.nn.Softmax(dim=-1)(predicted_label.logits))
        # tensor([[0.0108, 0.1264, 0.8628]])