"""
    This script contains function snippets for different training settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from visualDet3D.utils.utils import LossLogger, compound_annotation
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from typing import Tuple, List

@PIPELINE_DICT.register_module
@torch.no_grad()
def test_mono_detection(data, module:nn.Module,
                     writer:SummaryWriter, 
                     loss_logger:LossLogger=None, 
                     global_step:int=None, 
                     cfg:EasyDict=None,
                     eval_weather_type:str="clear") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    images, P2, foggy_images = data[0], data[1], data[5]

    scores, bbox, obj_index = module([images.cuda().float().contiguous(), torch.tensor(P2).cuda().float(), foggy_images.cuda().float().contiguous()], eval_weather_type=eval_weather_type)
    obj_types = [cfg.obj_types[i.item()] for i in obj_index]

    return scores, bbox, obj_types
