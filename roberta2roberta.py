import torch.nn as nn
from transformers import EncoderDecoderModel
import torch

model = EncoderDecoderModel.from_encoder_decoder_pretrained("klue/roberta-base", "klue/roberta-base")

class Roberta2Roberta(nn.Module):
    def __init__(self):
        super(Roberta2Roberta, self).__init__()
        self.roberta = model

    def generative(self, 
        input_ids,
        do_sample=True, 
        max_length=50,
        top_p=0.95,
        top_k=80,
        temperature=0.6, 
        no_repeat_ngram_size=2,
        num_return_sequence=3,
        early_stopping = False
        ):
        return self.roberta.generate(
            input_ids,
            do_sample=do_sample,
            max_length=max_length,
            #num_beams=num_beams,
            temperature=temperature,
            top_k = top_k,
			top_p = top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            #num_return_sequence=num_return_sequence,
            early_stopping=early_stopping

        )
 
    def forward(self, input, labels):
        outputs = self.roberta(input_ids=input, decoder_input_ids=labels, labels=labels)
        return outputs
