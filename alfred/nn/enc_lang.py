import torch
from torch import nn
from transformers import T5Tokenizer, T5ForConditionalGeneration


class EncoderLang(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        '''
        transformer encoder for language inputs
        '''
        super(EncoderLang, self).__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device).float()
        
        for param in self.model.parameters():
            param.share_memory_()
            param.requires_grad_(False)
        
    def forward(self, x, vocab=None):
        with torch.no_grad():
            toks = self.tokenize(x, vocab).to(self.device)
            return self.embed(toks)
    
    def tokenize(self, tok, vocab=None):
        if vocab != None:
            return [vocab.word2index(tok)]
        else:
            return self.tokenizer(tok, return_tensors="pt").input_ids
        
    def detokenize(self, toks):
        return self.tokenizer.decode(toks)
        
    def embed(self, toks):
        with torch.no_grad():
            return self.model.encoder.embed_tokens(toks).to(self.device)