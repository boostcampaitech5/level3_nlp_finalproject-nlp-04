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
    MODEL_NAME = 'klue/roberta-base'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    ## 데이터
    data = pd.read_csv('/opt/ml/finance_sentiment_corpus/finance_data.csv')
    data['labels'] = data['labels'].map({'negative':0, 'neutral':1, 'positive':2})

    temp = train_test_split(data['kor_sentence'], data['labels'],
                            test_size=0.2, shuffle=True, stratify=data['labels'], # label에 비율을 맞춰서 분리
                            random_state=SEED)
    sentence_train, sentence_val, label_train, label_val = temp

    max_length=30
    stride=0
    ## TODO 임의의 값으로 차후 수정
    train_encoding = tokenizer(sentence_train.tolist(),
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

    data2 = pd.read_csv('/opt/ml/finance_sentiment_corpus/native_korean_news_100.csv')
    val2_encoding = tokenizer(data2['kor_sentence'].tolist(),
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                ##
                                max_length=max_length,
                                stride=stride,
                                return_overflowing_tokens=True,
                                return_offsets_mapping=False
                                )
    val2_set = SentimentalDataset(val2_encoding, data2['labels'])

    
    ## 학습
    run = wandb.init(project="final_sentimental", entity="nlp-10")
    training_args = TrainingArguments(
        output_dir = './outputs',
        logging_steps = 50,
        num_train_epochs = 1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        fp16=True,
        report_to="wandb",
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


    ## 평가
    print('---val evaulate start---')
    trainer.evaluate(eval_dataset=val_set, metric_key_prefix='val1')
    
    print('---test evaluate start---')
    trainer.evaluate(eval_dataset=val2_set, metric_key_prefix='val2')

    # print('---inference start---') ## 추론 방법 1
    # model = model.to('cpu') ## 주의
    # my_text = '삼성전자, 올해부터 다운턴 끝나고 매출 상승 시작할 듯'
    # classifier = pipeline("sentiment-analysis", model=model,
    #                     tokenizer=tokenizer)
    # inference_output = classifier(my_text)
    # print(inference_output)
    # # [{'label': 'LABEL_2', 'score': 0.8627877831459045}]

    print('---inference start---') ## 추론 방법 2
    my_text = '삼성전자, 올해부터 다운턴 끝나고 매출 상승 시작할 듯'*10
    model = model.to('cpu')
    model.eval()
    with torch.no_grad() :
        temp = tokenizer(
            my_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            ##
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False
            )
        temp = {
            'input_ids':temp['input_ids'],
            'token_type_ids':temp['token_type_ids'],
            'attention_mask':temp['attention_mask'],
        }
        predicted_label = model(**temp)
        print(torch.nn.Softmax(dim=-1)(predicted_label.logits).mean(dim=0))
        # tensor([[0.0108, 0.1264, 0.8628]])
if __name__ == '__main__':
    train()
