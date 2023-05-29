import os
import types
import torch
import contextlib
import numpy as np
import torch.nn as nn

from torchvision import models
import torchvision.transforms.functional as F

from alfred.gen import constants
from alfred.nn.transforms import Transforms
from alfred.utils import data_util
import clip


class CLIPResnet50(nn.Module):
    '''
    pretrained Resnet18 from torchvision
    '''
    def __init__(self,
                 device,
                 checkpoint_path=None,
                 share_memory=False):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load("RN50", device=device)
        self.model = nn.Sequential(*list(self.model.visual.children())[:-1])
        self.model = self.model.to(torch.device(self.device)).float()
        # freeze CLIP
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()

    def extract(self, images):
        images_normalized = torch.stack(
                [self.preprocess(img) for img in images]).to(torch.device(self.device))
        features = self.model(images_normalized)
        return features
    
class Resnet50Bridge(nn.Module):
    """
    Bridge network to flatten ResNet50 feats
    """
    def __init__(self,
                 device,
                 checkpoint_path=None,
                 share_memory=False):
        super().__init__()
        # CLIP Conv to T5 FC bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 128, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(1568, 768)
        )
        if checkpoint_path is not None:
            print('Loading CLIP Conv to T5 bridge checkpoint from {}'.format(checkpoint_path))
            bridge_model_state_dict = torch.load(checkpoint_path, map_location=device)
            self.bridge.load_state_dict(bridge_model_state_dict)
        self.bridge = self.bridge.to(torch.device(device)).float()
        self.bridge = self.bridge.train()
        if share_memory:
            self.bridge.share_memory()
            
    def forward(self, x):
        return self.bridge(x)        
    
class RCNN(nn.Module):
    '''
    pretrained FasterRCNN or MaskRCNN from torchvision
    '''
    def __init__(self,
                 archi,
                 device='cuda',
                 checkpoint_path=None,
                 share_memory=False,
                 load_heads=False):
        super().__init__()
        self.device = device
        self.feat_layer = '3'
        if archi == 'maskrcnn':
            self.model = models.detection.maskrcnn_resnet50_fpn(
                weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                weights_backbone=models.ResNet50_Weights.DEFAULT,
                min_size=800)
        elif archi == 'fasterrcnn':
            self.model = models.detection.fasterrcnn_resnet50_fpn(
                weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                weights_backbone=models.ResNet50_Weights.DEFAULT,
                min_size=224)
        else:
            raise ValueError('Unknown model type = {}'.format(archi))

        if archi == 'maskrcnn':
            self._transform = self.model.transform
        else:
            self._transform = Transforms.get_transform('default')
        if not load_heads:
            for attr in ('backbone', 'body'):
                self.model = getattr(self.model, attr)

        if checkpoint_path is not None:
            self.load_from_checkpoint(
                checkpoint_path, load_heads, device, archi, 'backbone.body')
        self.model = self.model.to(torch.device(device)).float()
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()
        if load_heads:
            # if the model is used for predictions, prepare a vocabulary
            self.vocab_pred = {
                i: class_name for i, class_name in
                enumerate(constants.OBJECTS_ACTIONS)}

    def extract(self, images):
        if isinstance(
                self._transform, models.detection.transform.GeneralizedRCNNTransform):
            images_normalized = self._transform(
                torch.stack([F.to_tensor(img) for img in images]))[0].tensors
        else:
            images_normalized = torch.stack(
                [self._transform(img) for img in images])
        images_normalized = images_normalized.to(torch.device(self.device))
        model_body = self.model
        if hasattr(self.model, 'backbone'):
            model_body = self.model.backbone.body
        features = model_body(images_normalized)
        return features[self.feat_layer]

    def load_from_checkpoint(self, checkpoint_path, load_heads, device, archi, prefix):
        print('Loading RCNN checkpoint from {}'.format(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=device)
        if not load_heads:
            # load only the backbone
            state_dict = {
                k.replace(prefix + '.', ''): v
                for k, v in state_dict.items() if prefix + '.' in k}
        else:
            # load a full model, replace pre-trained head(s) with (a) new one(s)
            num_classes, in_features = state_dict[
                'roi_heads.box_predictor.cls_score.weight'].shape
            box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes)
            self.model.roi_heads.box_predictor = box_predictor
            if archi == 'maskrcnn':
                # and replace the mask predictor with a new one
                in_features_mask = \
                    self.model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
                    in_features_mask, hidden_layer, num_classes)
                self.model.roi_heads.mask_predictor = mask_predictor
        self.model.load_state_dict(state_dict)

    def predict_objects(self, image, confidence_threshold=0.0, verbose=False):
        image = F.to_tensor(image).to(torch.device(self.device))
        output = self.model(image[None])[0]
        preds = []
        for pred_idx in range(len(output['scores'])):
            score = output['scores'][pred_idx].cpu().item()
            if score < confidence_threshold:
                continue
            box = output['boxes'][pred_idx].cpu().numpy()
            label = self.vocab_pred[output['labels'][pred_idx].cpu().item()]
            if verbose:
                print('{} at {}'.format(label, box))
            pred = types.SimpleNamespace(
                label=label, box=box, score=score)
            if 'masks' in output:
                pred.mask = output['masks'][pred_idx].cpu().numpy()
            preds.append(pred)
        return preds


class FeatureExtractor(nn.Module):
    def __init__(self,
                 archi,
                 device='cuda',
                 checkpoint=None,
                 share_memory=False,
                 compress_type=None,
                 load_heads=False):
        super().__init__()
        self.feat_shape = data_util.get_feat_shape(archi, compress_type)
        self.eval_mode = True
        # changed 
        if archi == 'resnet50':
            assert not load_heads
            self.model = CLIPResnet50(device, checkpoint, share_memory)
        else:
            self.model = RCNN(
                archi, device, checkpoint, share_memory, load_heads=load_heads)
        self.compress_type = compress_type
        # load object class vocabulary
        vocab_obj_path = os.path.join(
            constants.ET_ROOT, constants.OBJ_CLS_VOCAB)
        self.vocab_obj = torch.load(vocab_obj_path)

    def featurize(self, images, batch=32):
        feats = []
        with (torch.set_grad_enabled(False) if not self.model.model.training
              else contextlib.nullcontext()):
            for i in range(0, len(images), batch):
                images_batch = images[i:i+batch]
                feats.append(self.model.extract(images_batch))
        feat = torch.cat(feats, dim=0)
        if self.compress_type is not None:
            feat = data_util.feat_compress(feat, self.compress_type)
        assert self.feat_shape[1:] == feat.shape[1:]
        return feat

    def predict_objects(self, image, verbose=False):
        with torch.set_grad_enabled(False):
            pred = self.model.predict_objects(image, verbose=verbose)
        return pred

    def train(self, mode):
        if self.eval_mode:
            return
        for module in self.children():
            module.train(mode)
