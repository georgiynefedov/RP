import copy

from vocab import Vocab

from alfred.utils import model_util
from alfred.nn.enc_lang import EncoderLang
import torch


class Preprocessor(object):
    def __init__(self, vocab, subgoal_ann=False, is_test_split=False, frame_size=300, device=torch.device('cuda')):
        self.subgoal_ann = subgoal_ann
        self.is_test_split = is_test_split
        self.frame_size = frame_size
        self.device = device

        if vocab is None:
            self.vocab = {
                'word': Vocab(['<<pad>>', '<<seg>>', '<<goal>>', '<<mask>>']),
                'action_low': Vocab(['<<pad>>', '<<seg>>', '<<stop>>', '<<mask>>']),
                'action_high': Vocab(['<<pad>>', '<<seg>>', '<<stop>>', '<<mask>>']),
            }
        else:
            self.vocab = vocab
        self.encoder_lang = EncoderLang()
        self.word_seg = self.vocab['word'].word2index('<<seg>>', train=False)

    def process_language(self, ex, traj, r_idx):
        # tokenize language
        if not self.subgoal_ann:
            goal_ann = ex['turk_annotations']['anns'][r_idx]['task_desc']
            instr_anns = ex['turk_annotations']['anns'][r_idx]['high_descs']
        else:
            goal_ann = ['<<seg>>']
            instr_anns = [[a['action']] + a['action_high_args']
                          for a in traj['num']['action_high']]
            instr_anns = [
                [self.vocab['action_high'].index2word(w) for w in instr_ann]
                for instr_ann in instr_anns]


        traj['ann'] = {
            'goal': '<<goal>> ' + goal_ann,
            'instr': ['<<instr>>'] + instr_anns,
            'repeat_idx': r_idx
        }
        if not self.subgoal_ann:
            traj['ann']['instr'] += ['<<stop>>']

        # convert words to tokens
        if 'num' not in traj:
            traj['num'] = {}
        # pre-process instructions
        instrs = ' '.join([traj['ann']['goal'], *traj['ann']['instr']])
        # print(f"Encoding goal/instructions: {instrs}")
        return self.encoder_lang.forward(instrs).cpu()


    def process_actions(self, ex, traj):
        # deal with missing end high-level action
        self.fix_missing_high_pddl_end_action(ex)

        # end action for low_actions
        end_action = {
            'api_action': {'action': 'NoOp'},
            'discrete_action': {'action': '<<stop>>', 'args': {}},
            'high_idx': ex['plan']['high_pddl'][-1]['high_idx']
        }

        # init action_low and action_high
        num_hl_actions = len(ex['plan']['high_pddl'])
        # temporally aligned with HL actions
        if 'num' not in traj:
            traj['num'] = {}
        traj['num']['action_low'] = [list() for _ in range(num_hl_actions)]
        traj['num']['action_high'] = []
        low_to_high_idx = []

        for a in (ex['plan']['low_actions'] + [end_action]):
            # high-level action index (subgoals)
            high_idx = a['high_idx']
            low_to_high_idx.append(high_idx)

            # low-level action (API commands)
            traj['num']['action_low'][high_idx].append({
                'high_idx': a['high_idx'],
                'action': a['discrete_action']['action'],
                'action_high_args': a['discrete_action']['args'],
                'objectId': a['api_action']['objectId'].split('|')[0] if 'objectId' in a['api_action'] else None,
            })

            # low-level bounding box (not used in the model)
            if 'bbox' in a['discrete_action']['args']:
                xmin, ymin, xmax, ymax = [
                    float(x) if x != 'NULL' else -1
                    for x in a['discrete_action']['args']['bbox']]
                traj['num']['action_low'][high_idx][-1]['centroid'] = [
                    (xmin + (xmax - xmin) / 2) / self.frame_size,
                    (ymin + (ymax - ymin) / 2) / self.frame_size,
                ]
            else:
                traj['num']['action_low'][high_idx][-1]['centroid'] = [-1, -1]

            # low-level interaction mask (Note: this mask needs to be decompressed)
            if 'mask' in a['discrete_action']['args']:
                mask = a['discrete_action']['args']['mask']
            else:
                mask = None
            traj['num']['action_low'][high_idx][-1]['mask'] = mask

            # interaction validity
            valid_interact = 1 if model_util.has_interaction(
                a['discrete_action']['action']) else 0
            traj['num']['action_low'][high_idx][-1]['valid_interact'] = valid_interact

        # low to high idx
        traj['num']['low_to_high_idx'] = low_to_high_idx
        
        # high-level actions
        for a in ex['plan']['high_pddl']:
            a['discrete_action']['args'] = [
                w.strip().lower() for w in a['discrete_action']['args']]
            traj['num']['action_high'].append({
                'high_idx': a['high_idx'],
                'action': a['discrete_action']['action'],
                'action_high_args': a['discrete_action']['args'],
            })

        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = len(traj['num']['action_low'])
        if self.subgoal_ann:
            lang_instr_seg_len = len(traj['num']['action_high'])
        else:
            lang_instr_seg_len = len(ex['turk_annotations']['anns'][
                0]['high_descs']) + 1

        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 0:
            assert (seg_len_diff == 1) # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            self.merge_last_two_low_actions(traj)

    def fix_missing_high_pddl_end_action(self, ex):
        '''
        appends a terminal action to a sequence of high-level actions
        '''
        if ex['plan']['high_pddl'][-1]['planner_action']['action'] != 'End':
            ex['plan']['high_pddl'].append({
                'discrete_action': {'action': 'NoOp', 'args': []},
                'planner_action': {'value': 1, 'action': 'End'},
                'high_idx': len(ex['plan']['high_pddl'])
            })

    def merge_last_two_low_actions(self, conv):
        '''
        combines the last two action sequences into one sequence
        '''
        extra_seg = copy.deepcopy(conv['num']['action_low'][-2])
        for sub in extra_seg:
            sub['high_idx'] = conv['num']['action_low'][-3][0]['high_idx']
            conv['num']['action_low'][-3].append(sub)
        del conv['num']['action_low'][-2]
        conv['num']['action_low'][-1][0]['high_idx'] = len(conv['plan']['high_pddl']) - 1
