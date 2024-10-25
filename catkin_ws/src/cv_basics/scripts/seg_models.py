# Import libraries
import numpy as np
import cv2
import torch
from PIL import Image as PILImage

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from utils.defaults import DefaultPredictor
from utils.visualizer import Visualizer, ColorMode
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

import detectron2
from detectron2.utils.logger import setup_logger
from optimum.onnxruntime import ORTModelForSemanticSegmentation

import os
import subprocess

setup_logger()
setup_logger(name="oneformer")

# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, SegformerForSemanticSegmentation
from PIL import Image

import torch
import torch.quantization
from transformers import AutoImageProcessor
import numpy as np

class model_inference:

    def __init__(self):
        use_swin = True
        self.TASK_INFER = {"semantic": self.semantic_run}
        self.cpu_device = torch.device("cuda")
        self.SWIN_CFG_DICT = {"ade20k": "/home/developer/Desktop/project/catkin_ws/src/cv_basics/scripts/configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml"}
        self.predictor, self.metadata = self.setup_modules("ade20k", "/home/developer/Desktop/project/catkin_ws/src/cv_basics/scripts/250_16_swin_l_oneformer_ade20k_160k.pth", use_swin)


    def setup_cfg(self, dataset, model_path, use_swin):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)

        if use_swin:
            cfg_path = self.SWIN_CFG_DICT[dataset]
        else:
            cfg_path = DINAT_CFG_DICT[dataset]
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.DEVICE = 'cuda'
        cfg.MODEL.WEIGHTS = model_path
        cfg.freeze()
        return cfg

    def setup_modules(self, dataset, model_path, use_swin):
        cfg = self.setup_cfg(dataset, model_path, use_swin)
       
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
            from cityscapesscripts.helpers.labels import labels
            stuff_colors = [k.color for k in labels if k.trainId != 255]
            metadata = metadata.set(stuff_colors=stuff_colors)
        
        return predictor, metadata

    def semantic_run(self, img, predictor, metadata):
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
        predictions = predictor(img, "semantic")
        out = visualizer.draw_sem_seg(
            predictions["sem_seg"].argmax(dim=0).to(self.cpu_device), alpha=0.5
        )
        return out



    def predict_segmentaion(self, img):
        task = "semantic"
        out = self.TASK_INFER[task](img, self.predictor, self.metadata).get_image()
        return out


class model_inference_hg:

    def __init__(self, model_name):
        self.task = {"oneformer": self.oneformer, "segformer": self.segformer, "mask2former": self.mask2former}
        self.image_processor, self.model = self.task[model_name]()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        if self.device.type == 'cuda':
            self.model.half()  # Convert model to FP16

    def oneformer(self):
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        #model = ORTModelForSemanticSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", export=True, provider='CUDAExecutionProvider')
        return processor, model

    def mask2former(self):
        image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        return image_processor, model

    def segformer(self):
        return

    def preprocess(self, image):
        
        pixel_values = inputs['pixel_values']
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        return pixel_values

    def predict_segmentaion(self, image_tensor):
        inputs = self.image_processor(images=image_tensor,task_inputs=["semantic"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                prediction = self.model(**inputs)
        return prediction

    def visualization(self, seg_img, image):
        seg_image = self.image_processor.post_process_semantic_segmentation(seg_img, target_sizes=[image.size[::-1]])[0]
        seg_image = seg_image.cpu().numpy()
        color_segmentation_map = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)  # height, width, 3

        palette = ade_palette()
        for label, color in enumerate(palette):
            color_segmentation_map[seg_image == label, :] = color

        ground_truth_color_seg = color_segmentation_map[..., ::-1]

        img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
        img = img.astype(np.uint8)
        return color_segmentation_map
        #return img

def ade_palette():
    # Define the ADE20K palette
    return [
            (250,250,250), (250,250,250),(250,250,250),(0,250,0),(250,250,250),
            (250,250,250), (250,250,250),(250,250,250),(250,250,250),(250,250,250),
            (250,250,250),(250,250,250),(0,0,250)
                   # Add more colors as needed
    ]