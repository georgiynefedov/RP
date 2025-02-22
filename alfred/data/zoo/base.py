import os
import pickle
import lmdb
import torch
import warnings
import numpy as np

from torch.utils.data import Dataset as TorchDataset
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

from alfred.gen import constants
from alfred.utils import data_util


class BaseDataset(TorchDataset):
    def __init__(self, name, partition, args, ann_type):
        path = os.path.join(constants.ET_DATA, name)
        self.partition = partition
        self.name = name
        self.args = args
        self.sampler_weights = []
        self.counts = defaultdict(int)
        if ann_type not in ('lang', 'frames', 'lang_frames'):
            raise ValueError('Unknown annotation type: {}'.format(ann_type))
        self.ann_type = ann_type
        self.test_mode = False
        self.pad = 0

        # read information about the dataset
        self.dataset_info = data_util.read_dataset_info(name)
        if self.dataset_info['visual_checkpoint']:
            print('The dataset was recorded using model at {}'.format(
                self.dataset_info['visual_checkpoint']))

        # load data
        self._length = self.load_data(path)
        # if 'train' not in partition:
        #     self._length = 50000
        if self.args.fast_epoch:
            self._length = 16
            
        print('{} dataset size = {}'.format(partition, self._length))

        # load vocabularies for input language and output actions
        vocab = data_util.load_vocab(name, ann_type)
        self.vocab_in = vocab['word']
        out_type = 'action_low'
        self.vocab_out = vocab[out_type]

    def load_data(self, path, feats=True, masks=True, jsons=True):
        '''
        load data
        '''
        self.feats_lmdb_path = os.path.join(path, self.partition, 'feats')
        self.actions_lmdb_path = os.path.join(path, self.partition, 'actions')
        self.instrs_lmdb_path = os.path.join(path, self.partition, 'instrs')

        # load jsons with pickle and parse them
        d_length = 0 # d_length is the length of the dataset
        # offsets is a map from new dataset indices to original indices
        # e.g. {
            # 0: 0, 
            # 7: 1, 
            # 18: 2,
            #...
        # }
        self.offsets = {0: 0} 
        if jsons:
            with open(os.path.join(
                    path, self.partition, 'jsons.pkl'), 'rb') as jsons_file:
                jsons = pickle.load(jsons_file)
            self.jsons_and_keys = defaultdict(list)
            for idx in range(len(jsons)):
                key = '{:06}'.format(idx).encode('ascii')
                task_jsons = jsons[key]
                for json in task_jsons:
                    # compatibility with the evaluation
                    if 'task' in json and isinstance(json['task'], str):
                        pass
                    else:
                        json['task'] = '/'.join(json['root'].split('/')[-3:-1])
                    # add dataset idx and partition into the json
                    json['dataset_name'] = self.name
                    self.jsons_and_keys[idx].append(json)
                    d_length += sum([len(al) for al in json['num']['action_low']])
                    self.sampler_weights += list(map(lambda x: x['action'], sum(json['num']['action_low'], [])))
                    actions = list(map(lambda x: x['action'], sum(json['num']['action_low'], [])))
                    for a in actions:
                        self.counts[a] += 1
                    # if the dataset has script annotations, do not add identical data
                    if len(set([str(j['ann']['instr']) for j in task_jsons])) == 1:
                        break
                self.offsets[d_length] = idx + 1
        self.sampler_weights = list(map(lambda x: d_length / self.counts[x] if self.counts[x] > 0 else 1, self.sampler_weights))
        # return the true length of the loaded data
        return d_length if jsons else None     

    def load_frames(self, sequence_idx, timestep, task_json):
        '''
        load image features from the disk
        @offset: the number of frames to load from sequence @key
        '''
        key = '{:06}'.format(sequence_idx).encode('ascii')
        if not hasattr(self, 'feats_lmdb'):
            self.feats_lmdb, self.feats = self.load_lmdb(self.feats_lmdb_path)
        feats_bytes = self.feats.get(key)
        feats_numpy = np.frombuffer(feats_bytes, dtype=np.float32)
        feats_numpy = feats_numpy.reshape(self.dataset_info['feat_shape'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            frames = torch.tensor(feats_numpy)
        return frames[:timestep + 1].to(self.args.device)
    
    def load_lmdb(self, lmdb_path):
        '''
        load lmdb (should be executed in each worker on demand)
        '''
        database = lmdb.open(
            lmdb_path, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=252)
        cursor = database.begin(write=False)
        return database, cursor

    def __len__(self):
        '''
        return dataset length
        '''
        return self._length

    def __getitem__(self, idx):
        '''
        get item at index idx
        '''
        raise NotImplementedError

    @property
    def id(self):
        return self.partition + ':' + self.name + ';' + self.ann_type

    def __del__(self):
        '''
        close the dataset
        '''
        if hasattr(self, 'feats_lmdb'):
            self.feats_lmdb.close()
        if hasattr(self, 'masks_lmdb'):
            self.masks_lmdb.close()

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.id)
