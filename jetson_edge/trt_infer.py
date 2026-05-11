import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

def nms_numpy(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    scores_np = np.array(scores)
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores_np.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
        
    return keep

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

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
        
        # Preprocessing with letterbox to preserve aspect ratio
        img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640))
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
        
        r = ratio[0]
        
        for row in out:
            conf = row[4]
            if conf > self.conf_thresh:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Undo letterbox padding and scaling
                cx = (cx - dw) / r
                cy = (cy - dh) / r
                w = w / r
                h = h / r
                
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                width = int(w)
                height = int(h)
                
                boxes.append([x1, y1, width, height])
                scores.append(float(conf))
                
        # NMS
        indices = nms_numpy(boxes, scores, self.iou_thresh)
        
        detections = []
        for i in indices:
            detections.append(boxes[i])
                
        return detections
