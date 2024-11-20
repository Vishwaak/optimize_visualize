import numpy as np
import pycuda.driver as cuda 
import tensorrt as trt
import time
import torch


from multiprocessing import Queue


from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput
from transformers import AutoImageProcessor
from predict_utils import post_process_semantic_segmentation1, ade_palette

image_queue = Queue()   

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
       

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
       
       

        # Allocate buffers
        

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings
            bindings.append(int(device_mem))

            # Append to the appropiate input/output list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # Transfer input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        cuda.memcpy_dtoh_async(self.outputs[1].host, self.outputs[1].device, self.stream)
        
        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs[0].host, self.outputs[1].host



class trt_infernce:

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.preprocess, self.trt_infernce = self.load(engine_path)

        pass
    
    def preprocess_image(self, image):
        inputs = self.preprocess(image, return_tensors="pt")
        return  np.array(inputs["pixel_values"])

    @staticmethod
    def load(engine_path):
        inference = TensorRTInference(engine_path)
        image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")

        return image_processor, inference
    
    def predict(self, image):
        class_queries_logits, masks_queries_logits = self.trt_infernce.infer(self.preprocess_image(image))
        print("done prediction")
        return class_queries_logits, masks_queries_logits


def visualize_image(class_queries_logits, masks_queries_logits, image):
    palette = ade_palette()
    class_queries_logits = class_queries_logits.reshape(1, 100, 151)
    masks_queries_logits = masks_queries_logits.reshape(1, 100, 96, 96)
    class_queries_logits = torch.from_numpy(class_queries_logits)
    masks_queries_logits = torch.from_numpy(masks_queries_logits)
    seg_image = Mask2FormerForUniversalSegmentationOutput(class_queries_logits=class_queries_logits,masks_queries_logits= masks_queries_logits)
    seg_image = post_process_semantic_segmentation1(seg_image, target_sizes=[image.size[::-1]])[0]
    color_segmentation_map = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)  # height, width, 3
    seg_image = seg_image.cpu().numpy().astype(np.uint8)
    for label, color in enumerate(palette):
        color_segmentation_map[seg_image == label, :] = color

    end = time.time()

    ground_truth_color_seg = color_segmentation_map[..., ::-1]

    img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
    img = img.astype(np.uint8)
    return img

