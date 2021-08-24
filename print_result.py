import argparse
import logging
from numpy import result_type

from transformers import EncoderDecoderModel, BertTokenizer
import torch
import pandas as pd
import re
from roberta2roberta import Roberta2Roberta

parser = argparse.ArgumentParser(description="Consulation chatbot based on Roberta")

parser.add_argument("--chat", default=False, help='response generation on given user input')
parser.add_argument("--test", default=False, help="for test")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = parser.parse_args()
logging.info(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("klue/roberta-base")
model = Roberta2Roberta()
model.load_state_dict(torch.load("./model/roberta2roberta.pt", map_location=device))
model.to(device)

if __name__ == "__main__":
    if args.chat:
        while True:
            sentence = input("me>")
            if sentence == "q":
                break
            input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False), device=device).unsqueeze(0)
            generated = tokenizer.decode(model.generative(input_ids)[0])
            generated = generated[:generated.find("[SEP]")]
            output = re.sub('[^.-?-가-힣ㄱ-ㅎㅏ-ㅣ]',' ',generated)
            print("chatbot>")
            print(output.lstrip())

    if args.test:
        test = pd.read_csv("./data/test_data.csv")
        test_Q = test['Q'].to_list()
        result = []
        for sentence in test_Q:
            input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False), device=device).unsqueeze(0)
            generated = tokenizer.decode(model.generative(input_ids)[0])
            generated = generated[:generated.find("[SEP]")]
            output = re.sub('[^.-?-가-힣ㄱ-ㅎㅏ-ㅣ]',' ',generated)
            result.append(output)
        test['chatbot'] = result
        test.to_csv("./result/chat_result.csv")