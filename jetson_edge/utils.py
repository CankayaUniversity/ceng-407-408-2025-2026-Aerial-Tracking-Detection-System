import cv2
import numpy as np

def apply_highlight_test(frame, pred_bboxes, intensity=0.5, expand_ratio=3.0):
    input_frame = frame.copy()
    if not pred_bboxes:
        return input_frame
        
    if len(pred_bboxes) > 0 and not isinstance(pred_bboxes[0], (list, tuple, np.ndarray)):
        pred_bboxes = [pred_bboxes]
        
    h, w, _ = input_frame.shape

    for pred_bbox in pred_bboxes:
        px, py, pw, ph = pred_bbox
        cx, cy = px + (pw / 2), py + (ph / 2)

        new_w, new_h = pw * expand_ratio, ph * expand_ratio

        nx1 = int(max(0, cx - (new_w / 2)))
        ny1 = int(max(0, cy - (new_h / 2)))
        nx2 = int(min(w, cx + (new_w / 2)))
        ny2 = int(min(h, cy + (new_h / 2)))

        blue_ch = input_frame[ny1:ny2, nx1:nx2, 0].astype(np.float32)
        blue_ch += (255 * intensity)
        input_frame[ny1:ny2, nx1:nx2, 0] = np.clip(blue_ch, 0, 255).astype(np.uint8)

    return input_frame

def apply_highlight_ir(frame, pred_bboxes, intensity=0.5, expand_ratio=3.0):
    if len(frame.shape) == 2:
        input_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        input_frame = frame.copy()
    if not pred_bboxes:
        return input_frame
        
    if len(pred_bboxes) > 0 and not isinstance(pred_bboxes[0], (list, tuple, np.ndarray)):
        pred_bboxes = [pred_bboxes]
        
    h, w, _ = input_frame.shape
    for pred_bbox in pred_bboxes:
        px, py, pw, ph = pred_bbox
        cx, cy = px + (pw / 2), py + (ph / 2)
        new_w, new_h = pw * expand_ratio, ph * expand_ratio
        nx1 = int(max(0, cx - (new_w / 2)))
        ny1 = int(max(0, cy - (new_h / 2)))
        nx2 = int(min(w, cx + (new_w / 2)))
        ny2 = int(min(h, cy + (new_h / 2)))
        blue_ch = input_frame[ny1:ny2, nx1:nx2, 0].astype(np.float32)
        blue_ch += (255 * intensity)
        input_frame[ny1:ny2, nx1:nx2, 0] = np.clip(blue_ch, 0, 255).astype(np.uint8)
    return input_frame
