import os
import re
import json
import torch
import lmdb
import shutil
import pickle
import warnings
import numpy as np

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from copy import deepcopy
from vocab import Vocab
from pathlib import Path

from alfred.gen import constants
from alfred.gen.utils import image_util
from alfred.utils import helper_util, model_util
import re

actions_to_nl = {
    '<<pad>>': 'pad',
    '<<seg>>': 'seg',
    '<<goal>>': 'goal',
    '<<mask>>': 'mask',
    'LookDown_15': 'down',
    'RotateLeft_90': 'left',
    'MoveAhead_25': 'ahead',
    'RotateRight_90': 'right',
    'PickupObject': 'pickup',
    'PutObject': 'put',
    'LookUp_15': 'up',
    '<<stop>>': 'stop',
    'SliceObject': 'slice',
    'OpenObject': 'open',
    'CloseObject': 'close',
    'ToggleObjectOn': 'on',
    'ToggleObjectOff': 'off'
}

nl_to_actions = {v: k for k, v in actions_to_nl.items()}

# from summact
# objects_to_nl = {
#     'diningtable': 'dining table', 'sinkbasin': 'sink basin', 'sidetable': 'side table', 'butterknife': 'butter knife',
#     'garbagecan': 'garbage can', 'tissuebox': 'tissue box', 'desklamp': 'desk lamp', 'winebottle': 'wine bottle', 
#     'coffeetable': 'coffee table', 'spraybottle': 'spray bottle', 'floorlamp': 'floor lamp', 'alarmclock': 'alarm clock',
#     'remotecontrol': 'remote control', 'coffeemachine': 'coffee machine', 'toiletpaper': 'toilet paper', 
#     'toiletpaperhanger': 'toilet paper hanger', 'creditcard': 'credit card', 'stoveburner': 'stove burner', 
#     'handtowelholder': 'hand towel holder', 'handtowel': 'hand towel', 'bathtubbasin': 'bathtub basin', 'soapbar':
#     'soap bar', 'tennisracket': 'tennis racket', 'soapbottle': 'soap bottle', 'glassbottle': 'glass bottle', 'dishsponge': 
#     'dish sponge', 'wateringcan': 'watering can', 'baseballbat': 'baseball bat', 'saltshaker': 'salt shaker',
#     'peppershaker': 'pepper shaker', 'stoveknob': 'stove knob', 'showercurtain': 'shower curtain', 'tomatosliced':
#     'sliced tomato', 'wateringcan': 'watering can', 'potatosliced': 'sliced potato', 'breadsliced': 'sliced bread',
#     'applesliced': 'sliced apple', 'lettucesliced': 'sliced lettuce', 'eggcracked': 'cracked egg', 'laundryhamper': 
#     'laundry hamper', 'laundryhamperlid': 'laundry hamper lid', 'tvstand': 'tv stand', 'footstool': 'foot stool',
#     'showerhead': 'shower head', 'showerdoor': 'shower door', 'showerglass': 'shower glass', 'scrubbrush': 'scrub brush',
#     'lightswitch': 'light switch', 'towlholder': 'towel holder'
# }

# nl_to_objects = {v: k for k, v in objects_to_nl.items()}

def read_images(image_path_list):
    images = []
    for image_path in image_path_list:
        image_orig = Image.open(image_path)
        images.append(image_orig.copy())
        image_orig.close()
    return images

# from summact
def translate_actions_to_natural_language(actions, task_json):
    result = ['start']
    for idx, action in enumerate(actions):
        if 'Object' in action:
            result.append(actions_to_nl[action] + ' ' + task_json['num']['action_high'][task_json['num']['low_to_high_idx'][idx]]['action_high_args'][0])
        elif action in actions_to_nl:
            result.append(actions_to_nl[action])
        else:
            result.append(action)
    return ' '.join(result)


def read_traj_images(json_path, image_folder):
    root_path = json_path.parents[0]
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
    image_names = [None] * len(json_dict['plan']['low_actions'])
    for im_idx, im_dict in enumerate(json_dict['images']):
        if image_names[im_dict['low_idx']] is None:
            image_names[im_dict['low_idx']] = im_dict['image_name']
    before_last_image = json_dict['images'][-1]['image_name']
    last_image = '{:09d}.png'.format(int(before_last_image.split('.')[0]) + 1)
    image_names.append(last_image)
    fimages = [root_path / image_folder / im for im in image_names]
    if not any([os.path.exists(path) for path in fimages]):
        # maybe images were compressed to .jpg instead of .png
        fimages = [Path(str(path).replace('.png', '.jpg')) for path in fimages]
    if not all([os.path.exists(path) for path in fimages]):
        return None
    assert len(fimages) > 0
    # this reads on images (works with render_trajs.py)
    # fimages = sorted(glob.glob(os.path.join(root_path, image_folder, '*.png')))
    try:
        images = read_images(fimages)
    except:
        return None
    return images


def extract_features(images, extractor):
    if images is None:
        return None
    feat = extractor.featurize(images, batch=8).type(torch.float32)
    return feat.cpu()


def decompress_mask_alfred(mask_compressed_alfred):
    '''
    decompress mask array from ALFRED compression (initially contained in jsons)
    '''
    mask = np.zeros((constants.DETECTION_SCREEN_WIDTH,
                     constants.DETECTION_SCREEN_HEIGHT))
    for start_idx, run_len in mask_compressed_alfred:
        for idx in range(start_idx, start_idx + run_len):
            mask[idx // constants.DETECTION_SCREEN_WIDTH,
                 idx % constants.DETECTION_SCREEN_HEIGHT] = 1
    return mask


def decompress_mask_bytes(mask_bytes):
    '''
    decompress mask given as a binary string and cast them to tensors (for optimization)
    '''
    mask_pil = image_util.decompress_image(mask_bytes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = transforms.ToTensor()(mask_pil)
    return mask


def process_masks(traj):
    masks = []
    for action_low in traj['plan']['low_actions']:
        if 'mask' in action_low['discrete_action']['args']:
            mask = decompress_mask_alfred(
                action_low['discrete_action']['args'].pop('mask'))
            masks.append(mask)
        else:
            masks.append(None)

    masks_compressed = []
    for mask in masks:
        if mask is not None:
            mask = image_util.compress_image(mask.astype('int32'))
        masks_compressed.append(mask)
    return masks_compressed


def process_traj(traj_orig, traj_path, r_idx, preprocessor):
    # copy trajectory
    traj = traj_orig.copy()
    # root & split
    traj['root'] = str(traj_path)
    partition = traj_path.parents[2 if 'tests_' not in str(traj_path) else 1].name
    traj['split'] = partition
    traj['repeat_idx'] = r_idx
    # numericalize actions for train/valid splits
    if 'test' not in partition: # expert actions are not available for the test set
        preprocessor.process_actions(traj_orig, traj)
    # numericalize language
    instr = preprocessor.process_language(traj_orig, traj, r_idx)
    return traj, instr


def gather_feats(files, output_path):
    print('Writing features to LMDB')
    if output_path.is_dir():
        shutil.rmtree(output_path)
    lmdb_feats = lmdb.open(str(output_path), 700*1024**3, writemap=True)
    os.makedirs(output_path / 'pickle', exist_ok=True) # for debugging
    with lmdb_feats.begin(write=True) as txn_feats:
        for idx, path in tqdm(enumerate(files)):
            traj_feats = torch.load(path).numpy()
            txn_feats.put('{:06}'.format(idx).encode('ascii'), traj_feats.tobytes())
            torch.save(torch.load(path), output_path / 'pickle' / '{:06}.pkl'.format(idx))
    lmdb_feats.close()


def gather_masks(files, output_path):
    print('Writing masks to LMDB')
    if output_path.is_dir():
        shutil.rmtree(output_path)
    lmdb_masks = lmdb.open(str(output_path), 50*1024**3, writemap=True)
    with lmdb_masks.begin(write=True) as txn_masks:
        for idx, path in tqdm(enumerate(files)):
            with open(path, 'rb') as f:
                masks_list = pickle.load(f)
                masks_list = [el for el in masks_list if el is not None]
                masks_buffer = BytesIO()
                pickle.dump(masks_list, masks_buffer)
                txn_masks.put('{:06}'.format(idx).encode('ascii'), masks_buffer.getvalue())
    lmdb_masks.close()


def gather_jsons(files, output_path):
    print('Writing JSONs to PKL')
    if output_path.exists():
        os.remove(output_path)
    jsons = {}
    for idx, path in tqdm(enumerate(files)):
        with open(path, 'rb') as f:
            jsons_idx = pickle.load(f)
            jsons['{:06}'.format(idx).encode('ascii')] = jsons_idx
    with output_path.open('wb') as f:
        pickle.dump(jsons, f)


def get_preprocessor(PreprocessorClass, subgoal_ann, lock, vocab_path=None):
    if vocab_path is None:
        init_words = ['<<pad>>', '<<seg>>', '<<goal>>', '<<mask>>']
    else:
        init_words = []
    vocabs_with_lock = {
        'word': helper_util.VocabWithLock(deepcopy(init_words), lock),
        'action_low': helper_util.VocabWithLock(deepcopy(init_words), lock),
        'action_high': helper_util.VocabWithLock(deepcopy(init_words), lock)}
    if vocab_path is not None:
        vocabs_loaded = torch.load(vocab_path)
        for vocab_name, vocab in vocabs_with_lock.items():
            loaded_dict = vocabs_loaded[vocab_name].to_dict()
            for i, w in enumerate(loaded_dict['index2word']):
                vocab.word2index(w, train=True)
                vocab.counts[w] = loaded_dict['counts'][w]

    preprocessor = PreprocessorClass(vocabs_with_lock, subgoal_ann)
    return preprocessor

def get_input_embedding(feat_list, pad, bridge, device, encoder_lang):
    '''
    Concatenate and pad input features in the following format:
    <instruction embedding> images: <image embeddings> actions: <action history embedding> <eos>
    '''
    def embed_input(x):
        '''
        embed a single input
        '''
        # print(x.keys())
        x['lang'] = torch.squeeze(x['lang'])[:-1] # squeeze and remove last token <eos>
        x['frames'] = bridge(x['frames'].to(device)).to(device)
        images_prefix_embeds = torch.squeeze(encoder_lang.forward('images:'))[:-1] # remove last token <eos>
        actions_prefix_embeds = torch.squeeze(encoder_lang.forward('actions:'))[:-1] # remove last token <eos>
        x = torch.cat([x['lang'], images_prefix_embeds, x['frames'], actions_prefix_embeds, x['action']], dim=0).to(device)
        return x
    
    embeds = [embed_input(x) for x in feat_list]
    input = pad_sequence(embeds, padding_value=pad, batch_first=True)
    input_mask = torch.ones(input.shape[:-1], dtype=torch.bool, device=device)
    for i in range(len(embeds)):
        input_mask[i, len(embeds[i]):] = False
    
    return {
        'input': input,
        'input_mask': input_mask
    }

def tensorize_and_pad(batch, device, pad, bridge, encoder_lang):
    '''
    cast values to torch tensors, put them to the correct device and pad sequences
    '''
    device = torch.device(device)
    traj_data, feat_list = list(zip(*batch))
    input_dict, gt_dict, feat_dict = get_input_embedding(feat_list, pad, bridge, device, encoder_lang), dict(), dict()
    for key in feat_list[0].keys():
        feat_dict[key] = [el[key] for el in feat_list]
    # check that all samples come from the same dataset
    if traj_data[0]:
        assert len(set([t['dataset_name'] for t in traj_data])) == 1
    
    k = 'gt_action'
    if k in feat_dict:
        v = feat_dict['gt_action']
        seqs = v if isinstance(v[0], torch.Tensor) else [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
        gt_dict[k] = pad_sequence(seqs, batch_first=True, padding_value=pad)
    
    return traj_data, input_dict, gt_dict


def sample_batches(iterators, device, pad, args, bridge, encoder_lang):
    '''
    sample a batch from each iterator, return Nones if the iterator is empty
    '''
    batches_dict = {}
    for dataset_id, iterator in iterators.items():
        try:
            batches = next(iterator)
        except StopIteration as e:
            return None
        dataset_name = dataset_id.split(':')[1]
        traj_data, input_dict, gt_dict = tensorize_and_pad(batches, device, pad, bridge, encoder_lang)
        batches_dict[dataset_name] = (traj_data, input_dict, gt_dict)
    return batches_dict


def load_vocab(name, ann_type='lang'):
    '''
    load a vocabulary from the dataset
    '''
    path = os.path.join(constants.ET_DATA, name, constants.VOCAB_FILENAME)
    vocab_dict = torch.load(path)
    # set name and annotation types
    for vocab in vocab_dict.values():
        vocab.name = name
        vocab.ann_type = ann_type
    return vocab_dict


def get_feat_shape(visual_archi, compress_type=None):
    '''
    Get feat shape depending on the training archi and compress type
    '''
    if visual_archi == 'fasterrcnn':
        # the RCNN model should be trained with min_size=224
        feat_shape = (-1, 2048, 7, 7)
    elif visual_archi == 'maskrcnn':
        # the RCNN model should be trained with min_size=800
        feat_shape = (-1, 2048, 10, 10)
    elif visual_archi == 'resnet50':
        feat_shape = (-1, 2048, 7, 7)
    else:
        raise NotImplementedError('Unknown archi {}'.format(visual_archi))

    if compress_type is not None:
        if not re.match(r'\d+x', compress_type):
            raise NotImplementedError('Unknown compress type {}'.format(compress_type))
        compress_times = int(compress_type[:-1])
        feat_shape = (
            feat_shape[0], feat_shape[1] // compress_times,
            feat_shape[2], feat_shape[3])
    return feat_shape


def feat_compress(feat, compress_type):
    '''
    Compress features by channel average pooling
    '''
    assert re.match(r'\d+x', compress_type) and len(feat.shape) == 4
    times = int(compress_type[:-1])
    assert feat.shape[1] % times == 0
    feat = feat.reshape((
        feat.shape[0], times,
        feat.shape[1] // times,
        feat.shape[2],
        feat.shape[3]))
    feat = feat.mean(dim=1)
    return feat


def read_dataset_info(data_name):
    '''
    Read dataset a feature shape and a feature extractor checkpoint path
    '''
    path = os.path.join(constants.ET_DATA, data_name, 'params.json')
    with open(path, 'r') as f_params:
        params = json.load(f_params)
    return params
