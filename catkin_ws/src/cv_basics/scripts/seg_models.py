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
from transformers.models.oneformer.modeling_oneformer import OneFormerForUniversalSegmentationOutput

import detectron2
from detectron2.utils.logger import setup_logger
from optimum.onnxruntime import ORTModelForSemanticSegmentation

import os
import subprocess
import time

import onnxruntime as ort

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
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput
from PIL import Image

import torch
import torch.quantization
from transformers import AutoImageProcessor
import numpy as np

from predict_utils import post_process_semantic_segmentation1
import cupy as cp

def ade_palette():
    # Define the ADE20K palette
    return [
            (250,250,250), (250,250,250),(250,250,250),(0,250,0),(250,250,250),
            (250,250,250), (250,250,250),(250,250,250),(250,250,250),(250,250,250),
            (250,250,250),(250,250,250),(0,0,250)
                   # Add more colors as needed
    ]


class model_inference_hg:

    def __init__(self, model_name, flag="onnx", warmup_itr=10):
        self.flag = flag
        self.model_name = model_name

        self.task = {"oneformer": self.oneformer, "mask2former": self.mask2former}
        self.image_processor, self.model = self.task[model_name]()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.flag != "onnx":
            self.model.to(self.device).eval()
            if self.device.type == 'cuda':
                self.model.half()  
        else:
            warmup_inputs = self.onnx_preprocessing(Image.open("output.jpg"))
            for i in range(warmup_itr):
                self.onnx_infernence(warmup_inputs)
            print("warmup done")
       

    def oneformer(self):
        processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        if self.flag == "onnx":
            model = ort.InferenceSession("oneformer.onnx",providers=["CUDAExecutionProvider"])
        else:
            model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        return processor, model

    def mask2former(self):
        image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")

        if self.flag == "onnx":
            model = ort.InferenceSession("mask2former.onnx", providers=["CUDAExecutionProvider"])
        else:
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        return image_processor, model


    def onnx_preprocessing(self, image):
        if self.model_name == "oneformer":
            inputs = self.image_processor(images=image,task_inputs=["semantic"], return_tensors="pt")
            ort_inputs = {key: value.cpu().detach().numpy() for key, value in inputs.items()}
            ort_inputs.pop("pixel_mask")

        else:
            inputs = self.image_processor(image, return_tensors="pt")
            ort_inputs = {"pixel_values": inputs["pixel_values"].cpu().detach().numpy()}
          
        
        return ort_inputs
    
    def onnx_infernence(self, inputs):
        start = time.time()
        output = self.model.run(None, inputs)
        end = time.time()
        print(end - start)
        return output

    def onnx_post_processing(self, output):
        class_queries_logits, masks_queries_logits, *_ = output
        class_queries_logits = torch.tensor(class_queries_logits)
        masks_queries_logits = torch.tensor(masks_queries_logits)
        if self.model_name == "oneformer":
            result = OneFormerForUniversalSegmentationOutput(class_queries_logits=class_queries_logits,masks_queries_logits= masks_queries_logits)
        else:
            result = Mask2FormerForUniversalSegmentationOutput(class_queries_logits=class_queries_logits, masks_queries_logits=masks_queries_logits)
        return result

    def preprocess(self, image):
        
        pixel_values = inputs['pixel_values']
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(1)
        return pixel_values

    def predict_segmentaion(self, image_tensor):
        inputs = self.image_processor(images=image_tensor,task_inputs=["semantic"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                start = time.time()
                prediction = self.model(**inputs)
                end = time.time()
                print(end - start)
        return prediction

    def visualization(self, seg_img, image):
        seg_image = post_process_semantic_segmentation1(seg_img, target_sizes=[image.size[::-1]])[0]
        color_segmentation_map = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = ade_palette()
        for label, color in enumerate(palette):
            color_segmentation_map[seg_image == label, :] = color

        ground_truth_color_seg = color_segmentation_map[..., ::-1]

        img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
        img = img.astype(np.uint8)
        #return color_segmentation_map
        return img


class onnx_infernce:

    def __init__(self):
        self.model, self.processor = self.load_()
        

    def onnx_preprocessing(self, image):
        inputs = self.processor(image, return_tensors="pt")
        ort_inputs = {"pixel_values": inputs["pixel_values"].cpu().detach().numpy()}
        return ort_inputs

    def load_(self):
        image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        model = ort.InferenceSession("mask2former.onnx", providers=["CUDAExecutionProvider"])
        return model, image_processor
    
    def predict(self, inputs):
        return self.model.run(None, inputs) 
    
    def warmup(self, inputs, warmup_itr):
        for i in range(warmup_itr):
            self.predict(inputs)
        print("warmup done")
    
    def run(self, inputs):
        inputs = self.onnx_preprocessing(inputs)
        start = time.time()
        output = self.predict(inputs)
        end = time.time()
        print(f"==============prediction time {end - start}=========")
        return output


class Visualizer:

    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        self.palette = ade_palette()
    def post_processing(self, output):
        class_queries_logits, masks_queries_logits, *_ = output
        return Mask2FormerForUniversalSegmentationOutput(class_queries_logits=torch.tensor(class_queries_logits), masks_queries_logits=torch.tensor(masks_queries_logits))
    
    def convert_segmenation_poly(self, labels, seg_img, image):
        poly_images = []
        epsilon_const = {3: 0.01, 12: 0.007}

        for label in labels:
            binary_mask = np.where(seg_img == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            epsilon = epsilon_const[label] * cv2.arcLength(contours[0], True)
            polygons = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
            poly_images.append(cv2.fillPoly(np.zeros_like(image), polygons, (255, 255, 255)))

        if len(poly_images) == 2:
            return cv2.addWeighted(poly_images[0],1,poly_images[1],1,0)
        
        return np.array(poly_images[0], dtype=np.uint8)

    def visual(self, seg_img, image):
        default_color = (0, 0,0)
        sel_lables = [3,12]


        seg_img = self.post_processing(seg_img)
        seg_image = post_process_semantic_segmentation1(seg_img, target_sizes=[image.size[::-1]])[0]
        # color_segmentation_map = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)

        seg_image = seg_image.cpu()
        seg_image = cp.asarray(seg_image)
        color_segmentation_map = cp.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=cp.uint8)

        poly_image = seg_image
        
        palette_len = len(self.palette)

        within_palette_labels = [label for label in sel_lables if label < palette_len]
        outside_palette_labels = [label for label in sel_lables if label >= palette_len]

        for label in within_palette_labels:
            cp.copyto(color_segmentation_map, cp.asarray(self.palette[label]), where=(seg_image == label)[..., None])

        for label in outside_palette_labels:
            cp.copyto(color_segmentation_map, cp.asarray(default_color), where=(seg_image == label)[..., None])

        color_segmentation_map = cp.asnumpy(color_segmentation_map)

        # for label in sel_lables:
        #     if label < palette_len:
        #         color_segmentation_map[seg_image == label, :] = self.palette[label]
        #     else:
        #         color_segmentation_map[seg_image == label, :] = default_color
        
        # ground_truth_color_seg = color_segmentation_map[..., ::-1]
        # overlay_image = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
        # overlay_image = overlay_image.astype(np.uint8)
        return color_segmentation_map[..., ::-1], poly_image

  
        
