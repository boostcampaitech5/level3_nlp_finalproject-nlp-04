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

from utils.utils import config_seed, extract_sentences_token, gpt_preprocessing_labels_token

'''
- git clone https://github.com/ukairia777/finance_sentiment_corpus.git 먼저 실행
- readme.md 작성 예정
'''

def sweep_train(config=None) :
    # wandb
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
    data_val = pd.read_csv("/opt/ml/finance_sentiment_corpus/26_company_half_rest_labeled.csv")
    
    data = gpt_preprocessing_labels_token(data) # gpt에서 출력한 오류들을 json 형식으로 맞춰주고 labels를 수정하는 것    
    data_val = gpt_preprocessing_labels_token(data_val)
    
    # "labels" 값을 추출하여 train/val dataset 나누기
    dataset = train_test_split(data['content_corpus_company'], data['label'],
                            test_size=0.2, shuffle=True, stratify=data['label'], # label에 비율을 맞춰서 분리
                            random_state=SEED)


    sentence_train, sentence_val, label_train, label_val = dataset

    sentence_train = data["content_corpus"]
    label_train = data["label"]
    sentence_val = data_val["content_corpus"]
    label_val = data_val["label"]

    max_length=wandb.config.max_length #원본:500
    stride=0
    
    # TODO 임의의 값으로 차후 수정
    train_encoding = tokenizer(sentence_train.tolist(), # pandas.Series -> list
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=max_length,
                                # stride=stride,
                                # return_overflowing_tokens=True,
                                # return_offsets_mapping=False
                                )

    val_encoding = tokenizer(sentence_val.tolist(),
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            # stride=stride,
                            # return_overflowing_tokens=True,
                            # return_offsets_mapping=False
                            )
    
    train_encoding = extract_sentences_token(train_encoding, tokenizer.pad_token_id)
    val_encoding = extract_sentences_token(val_encoding, tokenizer.pad_token_id) # 앞 128, 뒷 384 토큰으로 센텐스를 추출합니다.

    train_set = SentimentalDataset(train_encoding, label_train.reset_index(drop=True))
    val_set = SentimentalDataset(val_encoding, label_val.reset_index(drop=True))
    
    
    # 학습
    logging_steps = 200
    num_train_epochs = 3
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
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

    name = f'{MODEL_NAME}_merged-4_100-26_50' # 원하는 이름으로 바꾸기
    trainer.save_model(output_dir=f'./outputs/{name}')

if __name__ == '__main__':
    wandb_config = {
        'method': 'grid',
        'parameters':
        {
            'max_length': {'values' : [3000]}, # 충분히 커서 본문 전체를 토크나이징.
        }
    }
    sweep_id = wandb.sweep(sweep=wandb_config,
                        project='sweep_merged_all'
                        )
    wandb.agent(sweep_id, function=sweep_train)