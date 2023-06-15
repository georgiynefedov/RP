import torch
from torch import nn
from torch.nn import functional as F

from alfred.model import base
from alfred.nn.enc_lang import EncoderLang
from alfred.nn.enc_visual import Resnet50Bridge
from alfred.nn.enc_vl import EncoderVL
from alfred.utils import model_util
from alfred.utils import data_util


class Model(base.Model):
    def __init__(self, args, embs_ann, vocab_out, pad, seg):
        '''
        transformer agent
        '''
        super().__init__(args, embs_ann, vocab_out, pad, seg)
        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)
        # feature embeddings
        self.bridge = Resnet50Bridge(args.device)
        # language embeddings
        self.encoder_lang = EncoderLang(args.device)
        
        self.args = args
        # final touch
        self.reset()
        
    def forward(self, input, input_mask, gt_action):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        output = self.encoder_vl(input, input_mask, gt_action)
        return output

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = torch.zeros(1, 0, *self.visual_tensor_shape)
        self.action_traj = self.encoder_lang.tokenize('NoOp').to(self.args.device).squeeze(0)[:-1]

    def step(self, input_dict, vocab, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''
        device = self.args.device
        if prev_action is not None:
            prev_action_tensor = self.encoder_lang.tokenize(prev_action)[0].to(device)[:-1]
            self.action_traj = torch.cat(
                (self.action_traj.to(device), prev_action_tensor), dim=0)
        print("action traj", self.action_traj, self.encoder_lang.detokenize(self.action_traj))
        input_dict = input_dict.copy()
        if input_dict['frames'] is not None:
            frames = input_dict['frames']
            self.frames_traj = torch.cat((self.frames_traj.to(device), frames[None]), dim=1)
            
        input_dict['frames'] = self.frames_traj.squeeze(0).to(device)
        print("frames traj", input_dict['frames'].shape)
        input_dict['action'] = self.encoder_lang.embed(self.action_traj.to(device))
        _, full_input_dict, _ = data_util.tensorize_and_pad(zip([None], [input_dict]), device, self.pad, self.bridge, self.encoder_lang)
        
        step_out = self.encoder_vl.step(full_input_dict['input'], full_input_dict['input_mask'], action_embeds=input_dict['action'][None])
        return step_out

    def compute_batch_loss(self, model_out, gt_dict):
        '''
        loss function for Seq2Seq agent
        '''
        losses = {'action': model_out['loss']}
        return losses

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose=False):
        '''
        compute exact matching and f1 score for action predictions
        '''
        # import IPython; IPython.embed()
        
        
        action = torch.argmax(model_out['action'], axis=-1)
        gt_action = gt_dict['gt_action']
        GT = gt_action != -100
        T = (action == gt_action) & GT
        print(T.sum())
        print(GT.sum())
        metrics_dict['token_acc'].append(T.sum().item()/GT.sum().item())
        
        # preds = model_util.extract_action_preds(
        #     model_out, self.pad, self.vocab_out, lang_only=True)
        # stop_token = self.vocab_out.word2index('<<stop>>')
        # gt_actions = model_util.tokens_to_lang(
        #     gt_dict['action'], self.vocab_out, {self.pad, stop_token})
        # model_util.compute_f1_and_exact(
        #     metrics_dict, [p['action'] for p in preds], gt_actions, 'action')
