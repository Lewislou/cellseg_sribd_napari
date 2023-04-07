"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
######
import sys
sys.path.append('.')
import os
join = os.path.join
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import datasets, models, transforms
from . import resnet10, resnet18
from napari_plugin_engine import napari_hook_implementation
from . import sliding_window_inference,sliding_window_inference_large,__proc_np_hv
import PIL
import torch.nn.functional as F
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif
from . import FlexibleUNet_star,FlexibleUNet_hv 
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
        inputs=self.preprocess(PIL.Image.fromarray(pre_img_data)).unsqueeze(0)
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
        
        
        
import tqdm
from typing import TYPE_CHECKING
from typing import Any
import logging
import os, warnings, time, tempfile, datetime, pathlib, shutil, subprocess
from urllib.request import urlopen
from urllib.parse import urlparse
import pathlib
from pathlib import Path
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
import sys
models_logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    import napari
# initialize logger
# use -v or --verbose when starting napari to increase verbosity
logger = logging.getLogger(__name__)
if '--verbose' in sys.argv or '-v' in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
#@thread_worker
def read_logging(log_file, logwindow):
    with open(log_file, 'r') as thefile:
        #thefile.seek(0,2) # Go to the end of the file
        while True:
            line = thefile.readline()
            if not line:
                time.sleep(0.01) # Sleep briefly
                continue
            else:
                logwindow.cursor.movePosition(logwindow.cursor.End)
                logwindow.cursor.insertText(line)
                yield line
                
                
main_channel_choices = [('average all channels', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
                        ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)]
optional_nuclear_channel_choices = [('none', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
                                    ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)]
cp_strings = ['_cp_masks_', '_cp_outlines_', '_cp_flows_', '_cp_cellprob_']

_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath('.cellseg_sribd', 'models')
MODEL_DIR = _MODEL_DIR_DEFAULT
def cache_model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/Lewislou/cellseg_sribd/resolve/main/model.pt'
    cached_file = os.fspath(MODEL_DIR.joinpath('model.pt')) 
    #print(cached_file)
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        torch.hub.download_url_to_file(url, cached_file, hash_prefix=None,progress=True)
        #download_url_to_file(url, cached_file, progress=True)
    return cached_file
    
def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)
def widget_wrapper():
    from napari.qt.threading import thread_worker
    try:
        from torch import no_grad
    except ImportError:
        def no_grad():
            def _deco(func):
                return func
            return _deco

    #@thread_worker
    #@no_grad()
    #def run_cellpose(image, model_type, custom_model, channels, channel_axis, diameter,
                    #net_avg, resample, cellprob_threshold, 
                    #model_match_threshold, do_3D, stitch_threshold):
    @thread_worker
    @no_grad()
    def run_cellpose(image):
        #from sribd_cellseg_models import MultiStreamCellSegModel,ModelConfig
        print(os.getcwd())
        #flow_threshold = (31.0 - model_match_threshold) / 10.
        #if model_match_threshold==0.0:
            #flow_threshold = 0.0
            #logger.debug('flow_threshold=0 => no masks thrown out due to model mismatch')
        #logger.debug(f'computing masks with cellprob_threshold={cellprob_threshold}, flow_threshold={flow_threshold}')
        #if model_type=='custom':
            #CP = models.CellposeModel(pretrained_model=custom_model, gpu=True)
        #else:
            #CP = models.CellposeModel(model_type=model_type, gpu=True)
        #my_model = MultiStreamCellSegModel.from_pretrained("Lewislou/cellseg_sribd")
        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] > 3:
            image = image[:,:, :3]

        pre_img_data = np.zeros(image.shape, dtype=np.uint8)
        for i in range(3):
            img_channel_i = image[:,:,i]
            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
        my_model = MultiStreamCellSegModel(ModelConfig())
        file_path = cache_model_path()
        print(file_path)
        checkpoints = torch.load(file_path)
        #my_model.__init__(ModelConfig())
        my_model.load_checkpoints(checkpoints)
        with torch.no_grad():
            masks = my_model(pre_img_data)
        #masks, flows_orig, _ = CP.eval(image, 
                                    #channels=channels, 
                                    #channel_axis=channel_axis,
                                    #diameter=diameter,
                                    #net_avg=net_avg,
                                    #resample=resample,
                                    #cellprob_threshold=cellprob_threshold,
                                    #flow_threshold=flow_threshold,
                                    #do_3D=do_3D,
                                    #stitch_threshold=stitch_threshold)
        #del CP
        #if not do_3D and stitch_threshold==0 and masks.ndim > 2:
           # flows = [[flows_orig[0][i], 
                      #flows_orig[1][:,i],
                      #flows_orig[2][i],
                      #flows_orig[3][:,i]] for i in range(masks.shape[0])]
           # masks = list(masks)
            #flows_orig = flows
        return masks


    @magicgui(
        call_button='run segmentation',  
        layout='vertical',
        #model_type = dict(widget_type='ComboBox', label='model type', choices=['cyto'], value='cyto', tooltip='there is a <em>cyto</em> model, a new <em>cyto2</em> model from user submissions, and a <em>nuclei</em> model'),
        #custom_model = dict(widget_type='FileEdit', label='custom model path: ', tooltip='if model type is custom, specify file path to it here'),
        #main_channel = dict(widget_type='ComboBox', label='channel to segment', choices=main_channel_choices, value=0, tooltip='choose channel with cells'),
        #optional_nuclear_channel = dict(widget_type='ComboBox', label='optional nuclear channel', choices=optional_nuclear_channel_choices, value=0, tooltip='optional, if available, choose channel with nuclei of cells'),
        #diameter = dict(widget_type='LineEdit', label='diameter', value=30, tooltip='approximate diameter of cells to be segmented'),
        #compute_diameter_shape  = dict(widget_type='PushButton', text='compute diameter from shape layer', tooltip='create shape layer with circles and/or squares, select above, and diameter will be estimated from it'),
        #compute_diameter_button  = dict(widget_type='PushButton', text='compute diameter from image', tooltip='cellpose model will estimate diameter from image using specified channels'),
        #cellprob_threshold = dict(widget_type='FloatSlider', name='cellprob_threshold', value=0.0, min=-8.0, max=8.0, step=0.2, tooltip='cell probability threshold (set lower to get more cells and larger cells)'),
        #model_match_threshold = dict(widget_type='FloatSlider', name='model_match_threshold', value=27.0, min=0.0, max=30.0, step=0.2, tooltip='threshold on gradient match to accept a mask (set lower to get more cells)'),
        #compute_masks_button  = dict(widget_type='PushButton', text='recompute last masks with new cellprob + model match', enabled=False),
        #net_average = dict(widget_type='CheckBox', text='average 4 nets', value=True, tooltip='average 4 different fit networks (default) or if not checked run only 1 network (fast)'),
        #resample_dynamics = dict(widget_type='CheckBox', text='resample dynamics', value=False, tooltip='if False, mask estimation with dynamics run on resized image with diameter=30; if True, flows are resized to original image size before dynamics and mask estimation (turn on for more smooth masks)'),
        #process_3D = dict(widget_type='CheckBox', text='process stack as 3D', value=False, tooltip='use default 3D processing where flows in X, Y, and Z are computed and dynamics run in 3D to create masks'),
        #stitch_threshold_3D = dict(widget_type='LineEdit', label='stitch threshold slices', value=0, tooltip='across time or Z, stitch together masks with IoU threshold of "stitch threshold" to create 3D segmentation'),
        #clear_previous_segmentations = dict(widget_type='CheckBox', text='clear previous results', value=True),
        #output_flows = dict(widget_type='CheckBox', text='output flows and cellprob', value=True),
        #output_outlines = dict(widget_type='CheckBox', text='output outlines', value=True),
    )
    def widget(#label_logo, 
        viewer: Viewer,
        image_layer: Image,
        #model_type,
        #custom_model,
        #main_channel,
        #optional_nuclear_channel,
        #diameter,
        #shape_layer: Shapes,
        #compute_diameter_shape,
        #compute_diameter_button,
        #cellprob_threshold,
        #model_match_threshold,
        #compute_masks_button,
        #net_average,
        #resample_dynamics,
        #process_3D,
        #stitch_threshold_3D,
        #clear_previous_segmentations,
        #output_flows,
        #output_outlines
    ) -> None:
        # Import when users activate plugin

        if not hasattr(widget, 'cellpose_layers'):
            widget.cellpose_layers = []
        
        #if clear_previous_segmentations:
            #layer_names = [layer.name for layer in viewer.layers]
            #for layer_name in layer_names:
                #if any([cp_string in layer_name for cp_string in cp_strings]):
                    #viewer.layers.remove(viewer.layers[layer_name])
            #widget.cellpose_layers = []

        def _new_layers(masks):
            from cellpose.utils import masks_to_outlines
            from cellpose.transforms import resize_image
            import cv2

            #flows = resize_image(flows_orig[0], masks.shape[-2], masks.shape[-1],
                                        #interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            #cellprob = resize_image(flows_orig[2], masks.shape[-2], masks.shape[-1],
                                    #no_channels=True)
            #cellprob = cellprob.squeeze()
            #outlines = masks_to_outlines(masks) * masks  
            if masks.ndim==3 and widget.n_channels > 0:
                masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
                #outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis), 
                                    #widget.n_channels, axis=widget.channel_axis)
               # flows = np.repeat(np.expand_dims(flows, axis=widget.channel_axis), 
                                #widget.n_channels, axis=widget.channel_axis)
                #cellprob = np.repeat(np.expand_dims(cellprob, axis=widget.channel_axis), 
                                    #widget.n_channels, axis=widget.channel_axis)
                
            #widget.flows_orig = flows_orig
            #widget.masks_orig = masks
            widget.iseg = '_' + '%03d'%len(widget.cellpose_layers)
            layers = []
            #if widget.output_flows.value:
                #layers.append(viewer.add_image(flows, name=image_layer.name + '_cp_flows' + widget.iseg, visible=False, rgb=True))
                #layers.append(viewer.add_image(cellprob, name=image_layer.name + '_cp_cellprob' + widget.iseg, visible=False))
            #if widget.output_outlines.value:
                #layers.append(viewer.add_labels(outlines, name=image_layer.name + '_cp_outlines' + widget.iseg, visible=False))
            layers.append(viewer.add_labels(masks, name=image_layer.name + '_cp_masks' + widget.iseg, visible=False))
            widget.cellpose_layers.append(layers)

        def _new_segmentation(segmentation):
            masks = segmentation

            _new_layers(masks)
            for layer in viewer.layers:
                layer.visible = False
            viewer.layers[-1].visible = True
            image_layer.visible = True
            widget.call_button.enabled = True
            
        image = image_layer.data 
        # put channels last
        widget.n_channels = 0
        widget.channel_axis = None
        if image_layer.ndim == 4 and not image_layer.rgb:
            chan = np.nonzero([a=='c' for a in viewer.dims.axis_labels])[0]
            if len(chan) > 0:
                chan = chan[0]
                widget.channel_axis = chan
                widget.n_channels = image.shape[chan]
        elif image_layer.ndim==3 and not image_layer.rgb:
            image = image[:,:,:,np.newaxis]
        elif image_layer.rgb:
            widget.channel_axis = -1

        cp_worker = run_cellpose(image=image)
        cp_worker.returned.connect(_new_segmentation)
        cp_worker.start()


    def update_masks(masks):     
        from cellpose.utils import masks_to_outlines

        outlines = masks_to_outlines(masks) * masks
        if masks.ndim==3 and widget.n_channels > 0:
            masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis), 
                            widget.n_channels, axis=widget.channel_axis)
            outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
        
        widget.viewer.value.layers[widget.image_layer.value.name + '_cp_masks' + widget.iseg].data = masks
        outline_str = widget.image_layer.value.name + '_cp_outlines' + widget.iseg
        if outline_str in widget.viewer.value.layers:
            widget.viewer.value.layers[outline_str].data = outlines
        widget.masks_orig = masks
        logger.debug('masks updated')

    return widget            
@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'cellseg'}


