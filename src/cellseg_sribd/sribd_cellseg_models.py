
import os
join = os.path.join
import argparse
import numpy as np
import torch
import torch.nn as nn
from napari_plugin_engine import napari_hook_implementation
from collections import OrderedDict
from torchvision import datasets, models, transforms
from classifiers import resnet10, resnet18

from utils_modify import sliding_window_inference,sliding_window_inference_large,__proc_np_hv
from PIL import Image
import torch.nn.functional as F
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif
from models.flexible_unet_convnext import FlexibleUNet_star,FlexibleUNet_hv 
from transformers import PretrainedConfig
from typing import List
from transformers import PreTrainedModel
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
@napari_hook_implementation
class ModelConfig(PretrainedConfig):
    model_type = "cell_sribd"
    def __init__(
        self,
        version = 1,
        input_channels: int = 3,
        roi_size: int = 512,
        overlap: float = 0.5,
        device: str = 'cpu',
        **kwargs,
    ):
        
        self.device = device
        self.roi_size = (roi_size, roi_size)
        self.input_channels = input_channels
        self.overlap = overlap
        self.np_thres, self.ksize, self.overall_thres, self.obj_size_thres = 0.6, 15, 0.4, 100
        self.n_rays = 32
        self.sw_batch_size = 4
        self.num_classes= 4
        self.block_size = 2048
        self.min_overlap = 128
        self.context = 128
        super().__init__(**kwargs)
        
@napari_hook_implementation        
class MultiStreamCellSegModel(PreTrainedModel):
    config_class = ModelConfig
    #print(config.input_channels)
    def __init__(self, config):
        super().__init__(config)
        #print(config.input_channels)
        self.config = config
        self.cls_model = resnet18()
        self.model0 = FlexibleUNet_star(in_channels=config.input_channels,out_channels=config.n_rays+1,backbone='convnext_small',pretrained=False,n_rays=config.n_rays,prob_out_channels=1,)
        self.model1 = FlexibleUNet_star(in_channels=config.input_channels,out_channels=config.n_rays+1,backbone='convnext_small',pretrained=False,n_rays=config.n_rays,prob_out_channels=1,)
        self.model2 = FlexibleUNet_star(in_channels=config.input_channels,out_channels=config.n_rays+1,backbone='convnext_small',pretrained=False,n_rays=config.n_rays,prob_out_channels=1,)
        self.model3 = FlexibleUNet_hv(in_channels=config.input_channels,out_channels=2+2,backbone='convnext_small',pretrained=False,n_rays=2,prob_out_channels=2,)
        self.preprocess=transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    def load_checkpoints(self,checkpoints):
        self.cls_model.load_state_dict(checkpoints['cls_model'])
        self.model0.load_state_dict(checkpoints['class1_model']['model_state_dict'])
        self.model1.load_state_dict(checkpoints['class2_model']['model_state_dict'])
        self.model2.load_state_dict(checkpoints['class3_model']['model_state_dict'])
        self.model3.load_state_dict(checkpoints['class4_model'])
        
    def forward(self, pre_img_data):
        inputs=self.preprocess(Image.fromarray(pre_img_data)).unsqueeze(0)
        outputs = self.cls_model(inputs)
        _, preds = torch.max(outputs, 1)    
        label=preds[0].cpu().numpy()
        test_npy01 = pre_img_data
        if label in [0,1,2]:
            if label == 0:
                output_label = sliding_window_inference_large(test_npy01,self.config.block_size,self.config.min_overlap,self.config.context, self.config.roi_size,self.config.sw_batch_size,predictor=self.model0,device=self.config.device)
            elif label == 1:
                output_label = sliding_window_inference_large(test_npy01,self.config.block_size,self.config.min_overlap,self.config.context, self.config.roi_size,self.config.sw_batch_size,predictor=self.model1,device=self.config.device)
            elif label == 2:
                output_label = sliding_window_inference_large(test_npy01,self.config.block_size,self.config.min_overlap,self.config.context, self.config.roi_size,self.config.sw_batch_size,predictor=self.model2,device=self.config.device)
        else:
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0, 3, 1, 2).type(torch.FloatTensor)

            output_hv, output_np = sliding_window_inference(test_tensor, self.config.roi, self.config.sw_batch_size, self.model3, overlap=self.config.overlap,device=self.config.device)
            pred_dict = {'np': output_np, 'hv': output_hv}
            pred_dict = OrderedDict(
                    [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]  # NHWC
                )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
            pred_output = torch.cat(list(pred_dict.values()), -1).cpu().numpy() # NHW3
            pred_map = np.squeeze(pred_output) # HW3
            pred_inst = __proc_np_hv(pred_map, self.config.np_thres, self.config.ksize, self.config.overall_thres, self.config.obj_size_thres)
            raw_pred_shape = pred_inst.shape[:2]
            output_label = pred_inst
        return output_label       
