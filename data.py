import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_legth, stride):
        self.input_ids = []
        self.target_ids = []
        tokenized_texts = [tokenizer.encode(text) for text in texts]
        for i in range(0, len(tokenized_texts)-max_legth, stride):
            input_ids = tokenized_texts[i:i+max_legth]
            target_ids = tokenized_texts[i+1:i+max_legth+1]
            self.input_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(texts, max_length, stride, batch_size, shuffle=True, drop_last=False, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(texts, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader
