# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
# import torch
# from PIL import Image

# image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic",torchscript=True)

# model = model.eval()
# model.cuda()

# image = Image.open("/home/developer/Desktop/project/models/data/kitchen/frames/kitchen/image_0.png")

# inputs = image_processor(images=image, return_tensors="pt").to("cuda")



# torch.onnx.export(
#     model,
#     dict(inputs),
#     "mask2former.onnx",
#     export_params=True,
#     opset_version=16,
#     input_names = ['pixel_values','pixel_mask'],
#     output_names = ['cls_logits','mask_logits'],
#     dynamic_axes={'input' : {0 : 'batch_size'},
#                   'output' : {0 : 'batch_size'}
#                  }
# )


from onnxruntime import InferenceSession
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
ort_session = InferenceSession("mask2former.onnx", providers=["CUDAExecutionProvider"])

input_names = [input.name for input in ort_session.get_inputs()]


image_path = "/home/developer/Desktop/project/models/data/kitchen/frames/kitchen/image_16769.png"
image = Image.open(image_path)  

inputs = processor(Image.open(image_path), return_tensors="pt")

ort_inputs = {"pixel_values": inputs["pixel_values"].cpu().detach().numpy()}

# input_names[1]: inputs["pixel_mask"].cpu().detach().numpy()

inputs = {k: v.cuda() for k, v in inputs.items()} 

# ort_inputs = {key: value.cpu().detach().numpy() for key, value in inputs.items()}
print(ort_inputs.keys())

import time
import numpy as np
import cv2
for i in range(2):
    start = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    end = time.time()
    print(end-start)
    class_queries_logits, masks_queries_logits, *_ = ort_outputs
    
 
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput
import torch
from typing import List, Optional, Tuple

def post_process_semantic_segmentation1(outputs, target_sizes: Optional[List[Tuple[int, int]]] = None) -> "torch.Tensor":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits.cuda()  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits.cuda()  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        # masks_classes = masks_classes.cuda()
        # masks_probs = masks_probs.cuda()
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

def ade_palette():
    # Define the ADE20K palette
    return [
            (250,250,250), (250,250,250),(250,250,250),(0,250,0),(250,250,250),
            (250,250,250), (250,250,250),(250,250,250),(250,250,250),(250,250,250),
            (250,250,250),(250,250,250),(0,0,250)
                   # Add more colors as needed
    ]

def convert_segmenation_poly(seg_img, image):
    _, binary = cv2.threshold(seg_img, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cv2.approxPolyDP(contour, 0, True) for contour in contours]
    poly_image = cv2.fillPoly(np.zeros_like(image), polygons, (255, 255, 255))
    return poly_image


class_queries_logits = torch.from_numpy(class_queries_logits)
masks_queries_logits = torch.from_numpy(masks_queries_logits)

start = time.time()
seg_image = Mask2FormerForUniversalSegmentationOutput(class_queries_logits=class_queries_logits,masks_queries_logits= masks_queries_logits)
end = time.time()



processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
start = time.time()
seg_image = post_process_semantic_segmentation1(seg_image, target_sizes=[image.size[::-1]])[0]
end = time.time()
print(end-start)
start = time.time()

end = time.time()
color_segmentation_map = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)  # height, width, 3
end = time.time()

palette = ade_palette()
start = time.time()


seg_image = seg_image.cpu().numpy().astype(np.uint8)
binary_mask = np.where(seg_image == 3, 255, 0).astype(np.uint8)
binary_mask_0 = np.where(seg_image == 12, 255, 0).astype(np.uint8)

contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
epsilon = 0.007 * cv2.arcLength(contours[0], True)
polygons = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
poly_image = cv2.fillPoly(np.zeros_like(image), polygons, (0, 255, 0))

contours0, _ = cv2.findContours(binary_mask_0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
epsilon0 = 0.5 * cv2.arcLength(contours0[0], True)
polygons0 = [cv2.approxPolyDP(contour, epsilon0, True) for contour in contours0]
poly_image0 = cv2.fillPoly(np.zeros_like(image), polygons0, (0, 0, 255))

final_image = cv2.addWeighted(poly_image, 1, poly_image0, 1, 0)

cv2.imwrite("poly.png", final_image)

for label, color in enumerate(palette):
    color_segmentation_map[seg_image == label, :] = color

end = time.time()

ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)
cv2.imwrite("output.png", img)   