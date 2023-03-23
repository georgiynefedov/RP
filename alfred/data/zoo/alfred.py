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


class AlfredDataset(BaseDataset):
    def __init__(self, name, partition, args, ann_type):
        super().__init__(name, partition, args, ann_type)
        # preset values
        self.max_subgoals = constants.MAX_SUBGOALS
        self._load_features = True
        self._load_frames = True
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
        offset = idx - base_idx
        base_idx = self.offsets[base_idx]
        task_json, key = self.jsons_and_keys[base_idx]
        feat_dict = {}
        if self._load_features:
            feat_dict = self.load_features(task_json, offset, key)
        if self._load_frames:
            feat_dict['frames'] = self.load_frames(key, offset)
        return task_json, feat_dict

    def load_masks(self, key):
        '''
        load interaction masks from the disk
        '''
        if not hasattr(self, 'masks_lmdb'):
            self.masks_lmdb, self.masks = self.load_lmdb(
                self.masks_lmdb_path)
        masks_bytes = self.masks.get(key)
        masks_list = pickle.load(BytesIO(masks_bytes))
        masks = [data_util.decompress_mask_bytes(m) for m in masks_list]
        return masks

    # @timeit
    def load_features(self, task_json, offset, key):
        '''
        load features from task_json
        '''
        feat = dict()
        # language inputs
        
        feat['lang'] = self.load_instrs(key)

        # action outputs
        if not self.test_mode:
            # action
            feat['action'] = self.load_actions(key, offset, task_json)
            feat['gt_action'] = self.load_gt_actions(task_json)
            # low-level valid interact
            feat['action_valid_interact'] = [
                a['valid_interact'] for a in sum(
                    task_json['num']['action_low'], [])]

        # auxillary outputs
        if not self.test_mode:
            # subgoal completion supervision
            if self.args.subgoal_aux_loss_wt > 0:
                feat['subgoals_completed'] = np.array(
                    task_json['num']['low_to_high_idx']) / self.max_subgoals
            # progress monitor supervision
            if self.args.progress_aux_loss_wt > 0:
                num_actions = len(task_json['num']['low_to_high_idx'])
                goal_progress = [(i + 1) / float(num_actions)
                                 for i in range(num_actions)]
                feat['goal_progress'] = goal_progress
        return feat

    # @timeit
    def load_gt_actions(self, task_json):
        '''
        load pre-processed action as a list of tokens from task_json
        '''
        lang_action = task_json['num']['action_low_tokenized'].to(self.args.device)
        return lang_action
    
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
    def load_actions(self, key, offset, task_json):
        '''
        load actions embeddings from the disk
        @offset: the number of actions to load from the sequence @key
        '''
        s = sum([[a['action'] for a in a_list] for a_list in task_json['num']['action_low']], [])
        s = s[:offset]
        # since actions may consist of several tokens, we need to sum the number of tokens
        actions_offset = sum(list(map(lambda x: len(x.split()), s)))
        if not hasattr(self, 'actions_lmdb'):
            self.actions_lmdb, self.actions = self.load_lmdb(self.actions_lmdb_path)
        actions_bytes = self.actions.get(key)
        actions_numpy = np.frombuffer(actions_bytes, dtype=np.float32)
        actions_numpy = actions_numpy.reshape((-1, 768))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            actions = torch.tensor(actions_numpy).to(self.args.device)
        return actions[:actions_offset]
    