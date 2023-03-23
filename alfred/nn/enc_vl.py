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
                emb_lang,
                emb_frames,
                emb_actions,
                lengths_lang,
                lengths_frames,
                lengths_action,
                length_frames_max,
                gt_actions,
                attn_masks=True):
        '''
        pass embedded inputs through embeddings and encode them using a transformer
        '''
        # padd lang and actions embeddings
        emb_lang = torch.cat((emb_lang, torch.zeros((emb_lang.shape[0], emb_lang.shape[1], self.demb - emb_lang.shape[2])).to(self.device)), dim=2)
        emb_actions = torch.cat((emb_actions, torch.zeros((emb_actions.shape[0], emb_actions.shape[1], self.demb - emb_actions.shape[2])).to(self.device)), dim=2)
        # emb_lang is processed on each GPU separately so they size can vary
        length_lang_max = lengths_lang.max().item()
        emb_lang = emb_lang[:, :length_lang_max]
        length_actions_max = lengths_action.max().item()
        emb_actions = emb_actions[:, :length_actions_max]
        # create a mask for padded elements
        length_mask_pad = length_frames_max + length_lang_max + length_actions_max
        mask_pad = torch.zeros((len(emb_lang), length_mask_pad), device=emb_lang.device).bool()
        for i, (len_l, len_f, len_a) in enumerate(
                zip(lengths_lang, lengths_frames, lengths_action)):
            # mask padded words
            mask_pad[i, len_l: length_lang_max] = True
            # mask padded frames
            mask_pad[i, length_lang_max + len_f:
                      length_lang_max + length_frames_max] = True
            # mask padded actions
            mask_pad[i, length_lang_max + length_frames_max + len_a:] = True

        # encode the inputs
        emb_all = torch.cat((emb_lang, emb_frames, emb_actions), dim=1)
        output = self.model(encoder_outputs = (emb_all, ), attention_mask=mask_pad, labels=gt_actions)
        return {'action': output.logits, 'loss': output.loss}
