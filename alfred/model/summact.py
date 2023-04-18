import torch
from torch import nn
from torch.nn import functional as F

from alfred.model import base
from alfred.nn.enc_lang import EncoderLang
from alfred.nn.enc_visual import Resnet50Bridge
from alfred.nn.enc_vl import EncoderVL
from alfred.utils import model_util


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
        
        # final touch
        self.reset()
        
    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        output = self.encoder_vl(inputs['input'], inputs['input_mask'], inputs['gt_action'])
        return output

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = torch.zeros(1, 0, *self.visual_tensor_shape)
        self.action_traj = torch.zeros(1, 0).long()

    def step(self, input_dict, vocab, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''
        frames = input_dict['frames']
        device = frames.device
        if prev_action is not None:
            prev_action_int = vocab['action_low'].word2index(prev_action)
            prev_action_tensor = torch.tensor(prev_action_int)[None, None].to(device)
            self.action_traj = torch.cat(
                (self.action_traj.to(device), prev_action_tensor), dim=1)
        self.frames_traj = torch.cat(
            (self.frames_traj.to(device), frames[None]), dim=1)
        # at timestep t we have t-1 prev actions so we should pad them
        action_traj_pad = torch.cat(
            (self.action_traj.to(device),
             torch.zeros((1, 1)).to(device).long()), dim=1)
        model_out = self.forward(
            vocab=vocab['word'],
            lang=input_dict['lang'],
            lengths_lang=input_dict['lengths_lang'],
            length_lang_max=input_dict['length_lang_max'],
            frames=self.frames_traj.clone(),
            lengths_frames=torch.tensor([self.frames_traj.size(1)]),
            length_frames_max=self.frames_traj.size(1),
            action=action_traj_pad)
        step_out = {}
        for key, value in model_out.items():
            # return only the last actions, ignore the rest
            step_out[key] = value[:, -1:]
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
        preds = model_util.extract_action_preds(
            model_out, self.pad, self.vocab_out, lang_only=True)
        stop_token = self.vocab_out.word2index('<<stop>>')
        gt_actions = model_util.tokens_to_lang(
            gt_dict['action'], self.vocab_out, {self.pad, stop_token})
        model_util.compute_f1_and_exact(
            metrics_dict, [p['action'] for p in preds], gt_actions, 'action')
