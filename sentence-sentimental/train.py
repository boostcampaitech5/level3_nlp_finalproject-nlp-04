from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import pipeline
import torch
from torch.utils.data import Dataset
import pandas as pd
import sklearn
from datasets import load_metric
import wandb

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = load_metric('accuracy').compute(predictions=preds, references=labels)['accuracy']
    f1 = load_metric('f1').compute(predictions=preds, references=labels, average='micro')['f1']

    return {'accuracy':acc, 'f1':f1}

class SentimentalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()
        return data

    def __len__(self):
        return len(self.labels)



## git clone https://github.com/ukairia777/finance_sentiment_corpus.git
## 먼저 해야 함
## 나중에 readme.md 작성할 예정

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'klue/roberta-base'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

data = pd.read_csv(
    '/opt/ml/finance_sentiment_corpus/finance_data.csv'
    )
data['labels'] = data['labels'].map({'negative':0, 'neutral':1, 'positive':2})
train_encoding = tokenizer(
    data['kor_sentence'].tolist(),
    return_tensors='pt',
    padding=True,
    truncation=True
    )
train_set = SentimentalDataset(train_encoding, data['labels'])


run = wandb.init(project="final_sentimental", entity="nlp-10")
training_args = TrainingArguments(
    output_dir = './outputs',
    logging_steps = 50,
    num_train_epochs = 1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    report_to="wandb",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=train_set,
    compute_metrics=compute_metrics
)

##

print('train start')
trainer.train()

print('evaulate start')
trainer.evaluate()
run.finish()

print('inference start')
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
with torch.no_grad():
    temp = tokenizer(
        my_text,
        return_tensors='pt',
        padding=True,
        truncation=True
        )
    predicted_label = model(**temp)
    print(torch.nn.Softmax(dim=-1)(predicted_label.logits))
    # tensor([[0.0108, 0.1264, 0.8628]])
