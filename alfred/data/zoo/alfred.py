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
        base_idx = idx
        while not base_idx in self.offsets.keys():
            base_idx -= 1
        timestep = idx - base_idx
        base_idx = self.offsets[base_idx]
        task_json, key = self.jsons_and_keys[base_idx]
        feat_dict = {}
        if self._load_features:
            feat_dict = self.load_features(task_json, timestep, key)
        if self._load_frames:
            feat_dict['frames'] = self.load_frames(key, timestep, task_json)
        return task_json, feat_dict

    # @timeit
    def load_features(self, task_json, timestep, key):
        '''
        load features from task_json
        '''
        feat = dict()
        # language inputs
        
        feat['lang'] = self.load_instrs(key)

        # action outputs
        if not self.test_mode:
            # action
            feat['action'] = self.load_actions(timestep, task_json)
            feat['gt_action'] = self.load_gt_actions(task_json)
            # low-level valid interact
            feat['action_valid_interact'] = [
                a['valid_interact'] for a in sum(
                    task_json['num']['action_low'], [])]
        return feat

    # @timeit
    def load_gt_actions(self, task_json):
        '''
        load and embed actions
        '''
        gt_actions = sum([sum([[a['action']], a['action_high_args']], []) for a in task_json['num']['action_high']], [])
        gt_actions = data_util.translate_pddl_to_english(gt_actions)
        gt_actions = self.encoder_lang.tokenize(gt_actions)[0].to(self.args.device)
        return gt_actions
        
    # @timeit
    def load_instrs(self, key):
        '''
        load pre-processed instructions embeddings from the disk
        '''
        if not hasattr(self, 'instrs_lmdb'):
            self.instrs_lmdb, self.instrs = self.load_lmdb(self.instrs_lmdb_path)
        instrs_bytes = self.instrs.get(key)
        instrs_numpy = np.frombuffer(instrs_bytes, dtype=np.float32)
        instrs_numpy = instrs_numpy.reshape((1, -1, 768))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            instrs = torch.tensor(instrs_numpy).to(self.args.device)
        return instrs
    
    # @timeit
    def load_actions(self, timestep, task_json):
        '''
        load and embed actions
        @timestep: the number of actions to load from the sequence
        '''
        prev_actions = sum([sum([[a['action']], a['action_high_args']], []) for a in task_json['num']['action_high'][:timestep]], [])
        prev_actions = data_util.translate_pddl_to_english(prev_actions)
        prev_actions = self.encoder_lang.forward(prev_actions)[0].to(self.args.device)
        return prev_actions
    