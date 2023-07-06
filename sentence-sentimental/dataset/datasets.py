import torch
from torch.utils.data import Dataset

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