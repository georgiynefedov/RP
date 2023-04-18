import torch
from torch import nn

from transformers import T5Tokenizer, T5ForConditionalGeneration


class EncoderVL(nn.Module):
    def __init__(self, args):
        '''
        transformer encoder for language, frames and action inputs
        '''
        super(EncoderVL, self).__init__()

        self.device = args.device
        self.demb = args.demb
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base").to(self.device).float()
        setattr(self.model.config, 'max_length', 300)
        for param in self.model.parameters():
            param.share_memory_()

    def forward(self,
                input_emb,
                input_mask,
                gt_action,
                attn_masks=True):
        '''
        pass embedded inputs through embeddings and encode them using a transformer
        '''
        output = self.model(encoder_outputs = (input_emb, ), attention_mask=input_mask, labels=gt_action)
        return {'action': output.logits, 'loss': output.loss}
