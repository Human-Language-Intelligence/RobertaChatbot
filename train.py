import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from makedataset import MakeDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from roberta2roberta import Roberta2Roberta

#update dataset
df = pd.read_csv("./data/train_data.csv")
df = df.astype({'Q': 'str','A': 'str'})

parser = argparse.ArgumentParser(description="Conversation chatbot training by roberta")

parser.add_argument("--batch_size", 
                    type=int,
                    default=32, 
                    help="batch size for training (default: 32)")
parser.add_argument("--n_epoch", 
                    type=int,
                    default=20, 
                    help="epoch for trainig (default: 20)")
parser.add_argument("--lr",
                    type=float, 
                    default=5e-5,
                    help="The initial learning rate")

args = parser.parse_args()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = args.batch_size
n_epoch = args.batch_size
lr = args.lr

train_dataset = MakeDataset(df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Roberta2Roberta()
model.to(device)

losses = []
epoch_ = []
model.train()
for epoch in range(n_epoch):
    count = 0
    epoch_loss = 0.0
    for questions, answers in tqdm(train_loader):
        optimizer.zero_grad()
        questions = torch.stack(questions)
        answers = torch.stack(answers)
        input_ids = questions.transpose(1,0)
        label = answers.transpose(1,0)
        input_ids = input_ids.to(device)
        label = label.to(device)
        loss = model(input_ids, label).loss
        loss.backward()
        #if you want to cliping gradient normalization, implement below code
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        epoch_loss += loss.item()
        count += 1
    epoch_loss = epoch_loss / count

    torch.save(model.state_dict(),"./model/model_ver2.pt")

    epoch_.append(epoch)
    losses.append(epoch_loss)
    
    print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, n_epoch, epoch_loss))
