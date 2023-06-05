import os
import torch
import pickle
import numpy as np
from io import BytesIO
import warnings

from alfred.gen import constants
from alfred.data.zoo.base import BaseDataset
from alfred.utils import data_util
from alfred.utils.helper_util import timeit
from alfred.nn.enc_lang import EncoderLang


class AlfredDataset(BaseDataset):
    def __init__(self, name, partition, args, ann_type):
        super().__init__(name, partition, args, ann_type)
        # preset values
        self.max_subgoals = constants.MAX_SUBGOALS
        self._load_features = True
        self._load_frames = True
        self.encoder_lang = EncoderLang()
        # load the vocabulary for object classes
        self.vocab_obj = torch.load(os.path.join(
            constants.ET_ROOT, constants.OBJ_CLS_VOCAB))

    def load_data(self, path):
        return super().load_data(
            path, feats=True, masks=True, jsons=True)

    def __getitem__(self, idx):
        # base_idx is the index of the sequence in the original dataset
        # offsets contains a map from new dataset indices to original indices
        sequence_idx = idx
        while sequence_idx not in self.offsets.keys():
            sequence_idx -= 1
        timestep = idx - sequence_idx
        sequence_idx = self.offsets[sequence_idx]
        annotator = 0
        while timestep >= len(self.jsons_and_keys[sequence_idx][annotator]['num']['low_to_high_idx']):
            timestep -= len(self.jsons_and_keys[sequence_idx][annotator]['num']['low_to_high_idx'])
            annotator += 1
            
        task_json = self.jsons_and_keys[sequence_idx][annotator]
        feat_dict = {}
        # print(f"\nGetting dataset item {idx}, which is frame {timestep} in sequence {sequence_idx} by annotator {annotator}")
        if self._load_features:
            feat_dict = self.load_features(task_json, timestep)
        if self._load_frames:
            feat_dict['frames'] = self.load_frames(sequence_idx, timestep, task_json)
            # print(f"Loading frames: {feat_dict['frames'].shape}")
        return task_json, feat_dict

    # @timeit
    def load_features(self, task_json, timestep):
        '''
        load features from task_json
        '''
        feat = dict()
        # language inputs
        
        feat['lang'] = self.load_instrs(task_json)

        # action outputs
        if not self.test_mode:
            # action
            feat['action'], feat['gt_action'] = self.load_actions(timestep, task_json, self.encoder_lang)
            # low-level valid interact
            feat['action_valid_interact'] = [
                a['valid_interact'] for a in sum(
                    task_json['num']['action_low'], [])]
        return feat
        
    # @timeit
    def load_instrs(self, task_json):
        '''
        load pre-processed instructions embeddings from the disk
        '''
        instrs = ' '.join([task_json['ann']['goal'], *task_json['ann']['instr']])
        # print(f"Encoding goal/instructions: {instrs}")
        instrs = self.encoder_lang.tokenize(instrs).to(self.args.device)[0][:-1]
        # print("Tokenized instructions", instrs, '\n')
        return self.encoder_lang.embed(instrs)
    
    # @timeit
    def load_actions(self, timestep, task_json, encoder_lang: EncoderLang):
        '''
        load and embed actions
        @timestep: the number of actions to load from the sequence
        '''
        def get_action(a):
            return a['action'] + ' ' + a['objectId'].lower() if a['objectId'] else a['action']
        actions = ['<<start>>'] + sum([[get_action(a) for a in al] for al in task_json['num']['action_low']], [])[:timestep + 1]
        actions_history, action_gt = data_util.translate_actions_to_natural_language(actions)
        # print(f"Loading Action History \n\t {actions_history} \nand GT Action \n\t {action_gt}")
        actions_history = encoder_lang.tokenize(actions_history)[0].to(self.args.device)
        # print("Tokenized Action History", actions_history, '\n')
        actions_history = encoder_lang.embed(actions_history)
        action_gt = encoder_lang.tokenize(action_gt)[0].to(self.args.device)
        # print("Tokenized GT action", action_gt, '\n')
        return actions_history, action_gt
    