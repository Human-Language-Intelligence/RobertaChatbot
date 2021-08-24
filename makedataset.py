import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class MakeDataset(Dataset):
    '''for preprocessing dataset'''
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        self.Q = [self.tokenizer(t, padding="max_length", truncation=True, max_length=40).input_ids for t in self.df['Q'].to_list()]
        self.A = [self.tokenizer(t, padding="max_length", truncation=True, max_length=40).input_ids for t in self.df['A'].to_list()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.Q[idx]
        answer = self.A[idx]
        return question, answer


if __name__ == "__main__":
  dataset = MakeDataset()
  print(dataset)