from torch import nn

from alfred.utils import data_util


class Model(nn.Module):
    def __init__(self, args, embs_ann, vocab_out, pad, seg, encoder_lang):
        '''
        Abstract model
        '''
        nn.Module.__init__(self)
        self.args = args
        self.vocab_out = vocab_out
        self.pad, self.seg = pad, seg
        self.visual_tensor_shape = data_util.read_dataset_info(
            args.data['train'][0])['feat_shape'][1:]

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose):
        '''
        compute model-specific metrics and put it to metrics dict
        '''
        raise NotImplementedError

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        raise NotImplementedError()

    def compute_batch_loss(self, model_out, gt_dict):
        '''
        compute the loss function for a single batch
        '''
        raise NotImplementedError()

    def compute_loss(self, model_outs, gt_dicts):
        '''
        compute the loss function for several batches
        '''
        # compute losses for each batch
        losses = {}
        for dataset_key in model_outs.keys():
            losses[dataset_key] = self.compute_batch_loss(
                model_outs[dataset_key], gt_dicts[dataset_key])
        return losses
