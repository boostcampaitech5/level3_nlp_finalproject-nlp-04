import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.utils import extract_sentences_token


def inference() :
    MODEL_PATH = "" #TODO: write_your_model_path
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    arr = ["이 기사는[COMPANY] 예시 기업 1 [/COMPANY]에 대한 기사야. [SEP] news corpus 1",
        "이 기사는[COMPANY] 예시 기업 2 [/COMPANY]에 대한 기사야. [SEP] news corpus 2"]
    answer = []

    model.eval()
    with torch.no_grad() :
        loader = torch.utils.data.DataLoader(dataset=arr, batch_size=32, shuffle=False)
        for batch_text in loader:
            temp = tokenizer(
                batch_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=3000
                )
            temp = extract_sentences_token(temp, tokenizer.pad_token_id)
            if torch.cuda.is_available():
                temp = {key: value.to(device) for key, value in temp.items()}
            predicted_label = model(**temp)
            print(predicted_label)

            results = torch.nn.Softmax(dim=-1)(predicted_label.logits)    
            for result in results :
                if result[0]>=result[1] :
                    answer.append("부정")
                else :
                    answer.append("긍정")
    print(answer)
    
if __name__== "__main__" :
    inference()