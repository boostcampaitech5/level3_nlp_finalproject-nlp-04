import torch
from torch.utils.data import Dataset

class SentimentalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        try:
            self.labels = labels[self.encoding['overflow_to_sample_mapping'].tolist()].reset_index(drop=True)
           # return_overflowing_tokens true의 경우 여러 개로 나뉜 문장에 대해 1개의 라벨을 매핑해줍니다.
        except:
            self.labels = labels
        
    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()
        return data

    def __len__(self):
        return len(self.labels)