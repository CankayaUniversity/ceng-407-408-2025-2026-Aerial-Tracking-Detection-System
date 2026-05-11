import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

class TRTYOLO:
    def __init__(self, engine_path, conf_thresh=0.30, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def predict(self, frame):
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocessing (YOLOv8 style: resize, BGR->RGB, HWC->CHW, /255.0)
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        
        # Copy to host memory
        np.copyto(self.inputs[0]['host'], img.ravel())
        
        # Transfer to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        self.stream.synchronize()
        
        # Postprocessing
        out = self.outputs[0]['host'].reshape(self.outputs[0]['shape']) # [1, 5, 8400] for 1 class
        out = out[0].T # [8400, 5] (cx, cy, w, h, conf)
        
        boxes = []
        scores = []
        
        x_factor = orig_w / 640.0
        y_factor = orig_h / 640.0
        
        for row in out:
            conf = row[4]
            if conf > self.conf_thresh:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = int((cx - w/2) * x_factor)
                y1 = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([x1, y1, width, height])
                scores.append(float(conf))
                
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.iou_thresh)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(boxes[i])
                
        return detections
