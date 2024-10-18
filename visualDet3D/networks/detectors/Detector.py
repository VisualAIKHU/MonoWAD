import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from visualDet3D.networks.utils.registry import DETECTOR_DICT
from visualDet3D.networks.detectors.MonoWAD import MonoWAD
from visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead
from visualDet3D.networks.heads.depth_losses import bin_depths, DepthFocalLoss


@DETECTOR_DICT.register_module
class MonoWAD_3D(nn.Module):
    def __init__(self, network_cfg):
        super(MonoWAD_3D, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.mono_core = MonoWAD(network_cfg.mono_backbone)

    def build_head(self, network_cfg):
        self.bbox_head = AnchorBasedDetection3DHead(
            **(network_cfg.head)
        )
        self.depth_loss = DepthFocalLoss(96)

    def train_forward(self, left_images, annotations, P2, depth_gt=None, foggy_images=None):
        
        features, depth, diff_loss = self.mono_core(dict(image=left_images, P2=P2, foggy=foggy_images, training=True))
        
        depth_output = depth

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=features,
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)
        
        depth_gt = bin_depths(depth_gt, mode = "LID", depth_min=1, depth_max=80, num_bins=96, target=True)

        if reg_loss.mean() > 0 and not depth_gt is None and not depth_output is None:
            
            depth_gt = depth_gt.unsqueeze(1)
            depth_loss = 1.0 * self.depth_loss(depth_output, depth_gt)
            loss_dict['depth_loss'] = depth_loss
            reg_loss += depth_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['depth_loss'] = torch.zeros_like(reg_loss)
            
        loss_dict['diff_loss'] = diff_loss
        return cls_loss, reg_loss, diff_loss, loss_dict

    def test_forward(self, left_images, P2, foggy_images=None, eval_weather_type:str="clear"):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing
        if eval_weather_type == "clear":
            inputs = left_images
        else:
            inputs = foggy_images
        features, _ = self.mono_core(dict(image=inputs, P2=P2, foggy=foggy_images, training=False))
        
        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=features,
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        
        return scores, bboxes, cls_indexes

    def forward(self, inputs, eval_weather_type:str="clear"):

        if isinstance(inputs, list) and len(inputs) >= 4:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs, eval_weather_type=eval_weather_type)
