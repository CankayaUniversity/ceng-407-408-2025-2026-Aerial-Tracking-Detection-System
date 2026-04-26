import sys
import os
import cv2
import time
import torch
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QGridLayout, QSizePolicy, QComboBox, QSlider,
    QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont
# Import YOLO and tracker functions
from ultralytics import YOLO

# RGB Tracker and Highlight
from core.tracker_rgb import Tracker as RGBTracker, prediction_function as rgb_prediction_function
from core.utils import apply_highlight_test as apply_highlight_rgb

# IR Tracker and Highlight
from core.tracker_ir import Tracker as IRTracker, prediction_function as ir_prediction_function

from core.inference import (
    detection_and_featuremap, 
    get_roi, 
    roi_align_no_embedding, 
    load_resnet18_embedder, 
    resnet_embedding
)

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


def draw_hud_bbox(frame, x, y, w, h, track_id, sim=0.0, corner_len_ratio=0.25, color=(0, 255, 150), thickness=2):
    """Draw military-style corner bracket HUD bounding box with info tag."""
    x2, y2 = x + w, y + h
    cl = max(int(min(w, h) * corner_len_ratio), 8)
    # Top-left
    cv2.line(frame, (x, y), (x + cl, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + cl), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y), (x2 - cl, y), color, thickness)
    cv2.line(frame, (x2, y), (x2, y + cl), color, thickness)
    # Bottom-left
    cv2.line(frame, (x, y2), (x + cl, y2), color, thickness)
    cv2.line(frame, (x, y2), (x, y2 - cl), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, thickness)
    # Info tag
    label = f"ID:{track_id} S:{sim:.2f} {w}x{h}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    tag_y = y - 8
    if tag_y - th - 4 < 0:
        tag_y = y2 + th + 12
    cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + 4), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + 4, tag_y), font, scale, color, thick, cv2.LINE_AA)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    telemetry_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str, str)  # (severity, message)
    finished_signal = pyqtSignal()
    def __init__(self, video_path, mode="RGB"):
        super().__init__()
        self._run_flag = True
        self.video_path = video_path
        self.mode = mode
        self.is_paused = False
        
        # Dynamic params
        self.iou_thresh = 0.10
        self.sim_thresh = 0.50
        self.conf_thresh = 0.30
        self.target_fps = 0
        
    def set_params(self, iou, sim, conf, fps):
        self.iou_thresh = iou
        self.sim_thresh = sim
        self.conf_thresh = conf
        self.target_fps = fps

    def run(self):
        # Load models
        try:
            if self.mode == "RGB":
                without_model_path = os.path.join("models", "rgb_normal.pt")
                motion_model_path = os.path.join("models", "rgb_highlight.pt")
            else:
                without_model_path = os.path.join("models", "ir_normal.pt")
                motion_model_path = os.path.join("models", "ir_highlight.pt")
            
            without_model = YOLO(without_model_path)
            motion_model = YOLO(motion_model_path)
        except Exception as e:
            print(f"Error loading models: {e}")
            self.finished_signal.emit()
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_resnet = False
        if use_resnet and self.mode == "RGB":
            resnet_model, resnet_preprocess = load_resnet18_embedder(device)
            
        if self.mode == "RGB":
            tracker = RGBTracker(similarity_threshold=0.5, max_missing=10)
            self.log_signal.emit("INFO", "RGB Tracker initialized")
        else:
            tracker = IRTracker(iou_threshold=0.1, max_missing=10)
            self.log_signal.emit("INFO", "IR Tracker initialized")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Video açılamadı!")
            self.finished_signal.emit()
            return
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        prev_time = time.time()
        
        # initialize optical flow state
        prev_gray = None
        prev_pts = None

        while self._run_flag and cap.isOpened():
            loop_start = time.time()
            if self.is_paused:
                time.sleep(0.05)
                continue
            ret, frame = cap.read()
            if not ret:
                break
                
            # Optical Flow Background vector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_dx, bg_dy = 0.0, 0.0
            
            if prev_gray is not None:
                if prev_pts is None or len(prev_pts) < 10:
                    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                
                if prev_pts is not None and len(prev_pts) > 0:
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                    if curr_pts is not None and status is not None:
                        status = status.flatten()
                        good_new = curr_pts[status == 1]
                        good_old = prev_pts[status == 1]
                        
                        if len(good_new) > 0:
                            good_new_flat = good_new.reshape(-1, 2)
                            good_old_flat = good_old.reshape(-1, 2)
                            diffs = good_new_flat - good_old_flat
                            bg_dx = np.median(diffs[:, 0])
                            bg_dy = np.median(diffs[:, 1])
                            prev_pts = good_new.reshape(-1, 1, 2)
                        else:
                            prev_pts = None
            prev_gray = gray
            
            if self.mode == "RGB":
                tracker.similarity_threshold = self.sim_thresh
                scored_tracks = []
                for t in tracker.tracks:
                    history_score = min(len(t.history), 50) / 50.0
                    total_score = (history_score * 0.5) + (t.sim * 0.5)
                    scored_tracks.append((total_score, t))
                scored_tracks.sort(key=lambda x: x[0], reverse=True)
                best_track = tracker.get_best_track() 
                
                use_motion = best_track is not None and best_track.missing_frames == 0
                current_model_name = "Motion Model" if use_motion else "Base Model"
                if use_motion:
                    pred_bboxes = rgb_prediction_function(best_track)
                    input_frame = apply_highlight_rgb(frame, pred_bboxes)
                    model = motion_model
                else:
                    input_frame = frame
                    model = without_model
                # Run detection
                results, feat_map = detection_and_featuremap(model, input_frame, conf=self.conf_thresh)
                detections, embeddings = [], []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append(bbox)
                        if use_resnet:
                            emb = resnet_embedding(
                                resnet_model,
                                resnet_preprocess,
                                frame,
                                bbox,
                                device
                            )
                        else:
                            roi = get_roi(feat_map, bbox, orig_w, orig_h)
                            emb = roi_align_no_embedding(roi)
                        embeddings.append(emb)
                tracker.update(detections, embeddings, frame_idx, orig_w, orig_h, bg_dx, bg_dy)
            else:
                tracker.iou_threshold = self.iou_thresh
                scored_tracks = sorted(tracker.tracks, key=lambda t: t.missing_frames)
                best_track = tracker.get_best_track()
                
                if best_track is not None and best_track.is_stationary():
                    self.log_signal.emit("KILL", f"Track #{best_track.track_id} killed: stationary. Tracker reset.")
                    tracker.reset()
                    best_track = None
                # ====================================================

                # Eğer hala geçerli bir track varsa Highlight'a geç
                use_motion = best_track is not None and best_track.missing_frames == 0
                current_model_name = "Motion Model" if use_motion else "Base Model"
                
                if use_motion:
                    pred_bboxes = ir_prediction_function(best_track)
                    input_frame = apply_highlight_ir(frame, pred_bboxes)
                    model = motion_model
                else:
                    input_frame = frame
                    model = without_model

                results = model.predict(input_frame, imgsz=640, verbose=False, conf=self.conf_thresh)
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append(bbox)
                        
                tracker.update(detections, frame_idx, frame_width=orig_w, frame_height=orig_h, bg_dx=bg_dx, bg_dy=bg_dy)
            # --- Track rendering & target info ---
            active_tracks_count = 0
            best_active = None
            best_active_score = -1.0
            prev_track_ids = set(getattr(self, '_prev_track_ids', set()))
            curr_track_ids = set()
            
            for t in tracker.tracks:
                if t.missing_frames > 0:
                    continue
                active_tracks_count += 1
                curr_track_ids.add(t.track_id)
                
                x, y, w, h = map(int, t.bbox)
                sim_val = getattr(t, 'sim', 0.0)
                draw_hud_bbox(frame, x, y, w, h, t.track_id, sim=sim_val)
                
                score = min(len(t.history), 50) / 50.0 * 0.5 + sim_val * 0.5
                if score > best_active_score:
                    best_active_score = score
                    best_active = t
            
            # Log new tracks
            new_ids = curr_track_ids - prev_track_ids
            for nid in new_ids:
                self.log_signal.emit("INFO", f"Track #{nid} created")
            self._prev_track_ids = curr_track_ids
            
            # Log model switches
            if not hasattr(self, '_prev_model'):
                self._prev_model = ""
            if current_model_name != self._prev_model:
                self.log_signal.emit("INFO", f"Model switched to {current_model_name}")
                self._prev_model = current_model_name
            
            loop_end = time.time()
            elapsed = loop_end - loop_start
            if self.target_fps > 0:
                sleep_time = (1.0 / self.target_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            curr_time = time.time()
            actual_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Build target info
            if best_active is not None:
                bx, by, bw, bh = best_active.bbox
                tcx = int(bx + bw / 2)
                tcy = int(by + bh / 2)
                vel = 0.0
                if len(best_active.history) >= 2:
                    prev_b = best_active.history[-2]
                    vel = np.hypot((bx + bw/2) - (prev_b[0] + prev_b[2]/2),
                                   (by + bh/2) - (prev_b[1] + prev_b[3]/2))
                target_info = {
                    "target_id": str(best_active.track_id),
                    "target_pos": f"({tcx}, {tcy})",
                    "target_size": f"{int(bw)}x{int(bh)}",
                    "target_sim": f"{getattr(best_active, 'sim', 0.0):.2f}",
                    "target_vel": f"{vel:.1f} px/f",
                    "track_age": str(len(best_active.history)),
                }
            else:
                target_info = {
                    "target_id": "—", "target_pos": "—", "target_size": "—",
                    "target_sim": "—", "target_vel": "—", "track_age": "—",
                }
            
            self.change_pixmap_signal.emit(frame)
            telemetry_data = {
                "fps": f"{actual_fps:.1f}",
                "active_tracks": str(active_tracks_count),
                "model": current_model_name,
                "frame": str(frame_idx),
                "device": device.upper(),
            }
            telemetry_data.update(target_info)
            self.telemetry_signal.emit(telemetry_data)
            frame_idx += 1
        cap.release()
        self.finished_signal.emit()
    def stop(self):
        self._run_flag = False
        self.wait()
    def toggle_pause(self):
        self.is_paused = not self.is_paused

class TelemetryPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 12px;
                border: 1px solid #333333;
                padding: 10px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
                border: none;
                background: transparent;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(6)
        self.setLayout(layout)
        
        # --- System Status Section ---
        sec1_title = QLabel("▸ SYSTEM STATUS")
        sec1_title.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        layout.addWidget(sec1_title)
        
        grid1 = QGridLayout()
        grid1.setSpacing(4)
        
        grid1.addWidget(QLabel("FPS:"), 0, 0)
        self.lbl_fps = QLabel("0.0")
        self.lbl_fps.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 15px;")
        grid1.addWidget(self.lbl_fps, 0, 1)
        
        grid1.addWidget(QLabel("Frame:"), 1, 0)
        self.lbl_frame = QLabel("0")
        self.lbl_frame.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 15px;")
        grid1.addWidget(self.lbl_frame, 1, 1)
        
        grid1.addWidget(QLabel("Model:"), 2, 0)
        self.lbl_model = QLabel("—")
        self.lbl_model.setStyleSheet("color: #b3b3b3; font-weight: bold; font-size: 15px;")
        grid1.addWidget(self.lbl_model, 2, 1)
        
        grid1.addWidget(QLabel("Device:"), 3, 0)
        self.lbl_device = QLabel("—")
        self.lbl_device.setStyleSheet("color: #999999; font-weight: bold; font-size: 15px;")
        grid1.addWidget(self.lbl_device, 3, 1)
        
        grid1.addWidget(QLabel("Tracks:"), 4, 0)
        self.lbl_tracks = QLabel("0")
        self.lbl_tracks.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 15px;")
        grid1.addWidget(self.lbl_tracks, 4, 1)
        
        layout.addLayout(grid1)
        
        # --- Divider ---
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #333333; max-height: 1px; border: none;")
        layout.addWidget(divider)
        
        # --- Target Info Section ---
        sec2_title = QLabel("▸ TARGET INFO")
        sec2_title.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        layout.addWidget(sec2_title)
        
        grid2 = QGridLayout()
        grid2.setSpacing(4)
        
        grid2.addWidget(QLabel("Track ID:"), 0, 0)
        self.lbl_target_id = QLabel("—")
        self.lbl_target_id.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 15px;")
        grid2.addWidget(self.lbl_target_id, 0, 1)
        
        grid2.addWidget(QLabel("Position:"), 1, 0)
        self.lbl_target_pos = QLabel("—")
        self.lbl_target_pos.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 15px;")
        grid2.addWidget(self.lbl_target_pos, 1, 1)
        
        grid2.addWidget(QLabel("Size:"), 2, 0)
        self.lbl_target_size = QLabel("—")
        self.lbl_target_size.setStyleSheet("color: #b3b3b3; font-weight: bold; font-size: 15px;")
        grid2.addWidget(self.lbl_target_size, 2, 1)
        
        grid2.addWidget(QLabel("Velocity:"), 3, 0)
        self.lbl_target_vel = QLabel("—")
        self.lbl_target_vel.setStyleSheet("color: #999999; font-weight: bold; font-size: 15px;")
        grid2.addWidget(self.lbl_target_vel, 3, 1)
        
        grid2.addWidget(QLabel("Similarity:"), 4, 0)
        self.lbl_target_sim = QLabel("—")
        self.lbl_target_sim.setStyleSheet("color: #808080; font-weight: bold; font-size: 15px;")
        grid2.addWidget(self.lbl_target_sim, 4, 1)
        
        grid2.addWidget(QLabel("Track Age:"), 5, 0)
        self.lbl_track_age = QLabel("—")
        self.lbl_track_age.setStyleSheet("color: #707070; font-weight: bold; font-size: 15px;")
        grid2.addWidget(self.lbl_track_age, 5, 1)
        
        layout.addLayout(grid2)

    def update_data(self, data):
        self.lbl_fps.setText(data.get("fps", "0"))
        self.lbl_tracks.setText(data.get("active_tracks", "0"))
        self.lbl_model.setText(data.get("model", "—"))
        self.lbl_frame.setText(data.get("frame", "0"))
        self.lbl_device.setText(data.get("device", "—"))
        self.lbl_target_id.setText(data.get("target_id", "—"))
        self.lbl_target_pos.setText(data.get("target_pos", "—"))
        self.lbl_target_size.setText(data.get("target_size", "—"))
        self.lbl_target_vel.setText(data.get("target_vel", "—"))
        self.lbl_target_sim.setText(data.get("target_sim", "—"))
        self.lbl_track_age.setText(data.get("track_age", "—"))


class LogPanel(QFrame):
    MAX_LINES = 200
    COLORS = {
        "INFO": "#ffffff",
        "WARN": "#cccccc",
        "KILL": "#999999",
    }
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 12px;
                border: 1px solid #333333;
                padding: 8px;
            }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        self.setLayout(layout)
        
        title = QLabel("▸ EVENT LOG")
        title.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold; border: none; background: transparent;")
        layout.addWidget(title)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #111111;
                color: #e0e0e0;
                border: 1px solid #333333;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.text_edit)
        self._line_count = 0

    def append_log(self, severity, message):
        color = self.COLORS.get(severity, "#e0e0e0")
        ts = datetime.now().strftime("%H:%M:%S")
        html = f'<span style="color:#808080;">[{ts}]</span> <span style="color:{color}; font-weight:bold;">[{severity}]</span> <span style="color:{color};">{message}</span>'
        self.text_edit.append(html)
        self._line_count += 1
        if self._line_count > self.MAX_LINES:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, 1)
            cursor.removeSelectedText()
            cursor.deleteChar()
            self._line_count -= 1
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

class VideoLabel(QLabel):
    def sizeHint(self):
        return QSize(480, 360)

class VideoChannel(QFrame):
    closed_signal = pyqtSignal(object)
    
    def __init__(self, channel_id):
        super().__init__()
        self.channel_id = channel_id
        self.thread = None
        self.video_path = None
        
        self.setStyleSheet("""
            QFrame {
                background-color: #181818;
                border-radius: 12px;
                border: 1px solid #333333;
            }
        """)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Left side: Video & Controls
        left_layout = QVBoxLayout()
        
        # Header to show channel info
        header_layout = QHBoxLayout()
        header_lbl = QLabel(f"CHANNEL {self.channel_id}")
        header_lbl.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 14px; border: none; background: transparent;")
        header_layout.addWidget(header_lbl)
        header_layout.addStretch()
        
        self.btn_close = QPushButton("X")
        self.btn_close.setFixedSize(24, 24)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border-radius: 12px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        self.btn_close.clicked.connect(self.request_close)
        header_layout.addWidget(self.btn_close)
        
        left_layout.addLayout(header_layout)
        
        # Video Display
        self.video_label = VideoLabel("No Video Loaded")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(1, 1)
        self.video_label.setStyleSheet("""
            background-color: #000000;
            border-radius: 10px;
            color: #808080;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Segoe UI', Arial, sans-serif;
            border: 2px solid #333333;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        btn_style = """
            QPushButton {
                background-color: #2e4a36;
                color: #ffffff;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3c6648;
            }
            QPushButton:disabled {
                background-color: #222222;
                color: #555555;
            }
        """
        
        self.btn_load = QPushButton("Load Video")
        self.btn_load.setStyleSheet(btn_style)
        self.btn_load.clicked.connect(self.load_video)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["RGB", "IR"])
        self.combo_mode.setStyleSheet("""
            QComboBox {
                background-color: #2e4a36;
                color: #ffffff;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
        """)

        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.toggle_video)
        self.btn_start.setEnabled(False)
        
        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(self.combo_mode)
        controls_layout.addWidget(self.btn_start)
        left_layout.addLayout(controls_layout)
        
        # Parameters
        params_layout = QGridLayout()
        self.lbl_iou = QLabel("IoU Thresh: 0.10")
        self.slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.slider_iou.setRange(1, 100)
        self.slider_iou.setValue(10)
        self.slider_iou.valueChanged.connect(self.on_params_changed)
        params_layout.addWidget(self.lbl_iou, 0, 0)
        params_layout.addWidget(self.slider_iou, 0, 1)

        self.lbl_sim = QLabel("Sim Thresh: 0.50")
        self.slider_sim = QSlider(Qt.Orientation.Horizontal)
        self.slider_sim.setRange(1, 100)
        self.slider_sim.setValue(50)
        self.slider_sim.valueChanged.connect(self.on_params_changed)
        params_layout.addWidget(self.lbl_sim, 1, 0)
        params_layout.addWidget(self.slider_sim, 1, 1)

        self.lbl_conf = QLabel("Conf Thresh: 0.30")
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(1, 100)
        self.slider_conf.setValue(30)
        self.slider_conf.valueChanged.connect(self.on_params_changed)
        params_layout.addWidget(self.lbl_conf, 2, 0)
        params_layout.addWidget(self.slider_conf, 2, 1)

        self.lbl_fps = QLabel("Target FPS:")
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["Uncapped", "30 FPS", "60 FPS"])
        self.combo_fps.currentTextChanged.connect(self.on_params_changed)
        params_layout.addWidget(self.lbl_fps, 3, 0)
        params_layout.addWidget(self.combo_fps, 3, 1)
        
        left_layout.addLayout(params_layout)
        main_layout.addLayout(left_layout, stretch=3)
        
        # Right side: Telemetry + Log
        right_layout = QVBoxLayout()
        self.telemetry_panel = TelemetryPanel()
        right_layout.addWidget(self.telemetry_panel)
        
        self.log_panel = LogPanel()
        right_layout.addWidget(self.log_panel, stretch=1)
        
        main_layout.addLayout(right_layout, stretch=1)

    def request_close(self):
        self.close_channel()
        self.closed_signal.emit(self)

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if filename:
            if self.thread is not None:
                self.thread.stop()
            self.video_path = filename
            self.btn_start.setEnabled(True)
            self.video_label.setText("Video Loaded. Ready to start.")
            
    def on_params_changed(self):
        iou = self.slider_iou.value() / 100.0
        sim = self.slider_sim.value() / 100.0
        conf = self.slider_conf.value() / 100.0
        fps_text = self.combo_fps.currentText()
        if fps_text == "30 FPS":
            fps = 30
        elif fps_text == "60 FPS":
            fps = 60
        else:
            fps = 0
            
        self.lbl_iou.setText(f"IoU Thresh: {iou:.2f}")
        self.lbl_sim.setText(f"Sim Thresh: {sim:.2f}")
        self.lbl_conf.setText(f"Conf Thresh: {conf:.2f}")
        
        if self.thread is not None:
            self.thread.set_params(iou, sim, conf, fps)

    def toggle_video(self):
        if self.thread is None or not self.thread.isRunning():
            self.thread = VideoThread(self.video_path, self.combo_mode.currentText())
            self.on_params_changed()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.telemetry_signal.connect(self.telemetry_panel.update_data)
            self.thread.log_signal.connect(self.log_panel.append_log)
            self.thread.finished_signal.connect(self.video_finished)
            self.thread.start()
            self.btn_start.setText("Pause")
        else:
            self.thread.toggle_pause()
            if self.thread.is_paused:
                self.btn_start.setText("Resume")
            else:
                self.btn_start.setText("Pause")
                
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
        
    def video_finished(self):
        self.video_label.setText("Playback Finished.")
        self.btn_start.setText("Start")
        
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)
        
    def close_channel(self):
        if self.thread is not None:
            self.thread.stop()

from PyQt6.QtWidgets import QScrollArea

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aerial Target Detection & Tracking System")
        self.resize(1600, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #111111;
            }
            QWidget {
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Toolbar / Header
        header_layout = QHBoxLayout()
        title_lbl = QLabel("Aerial Target Detection & Tracking System")
        title_lbl.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff; border: none;")
        
        self.btn_add_channel = QPushButton("+ Add Channel")
        self.btn_add_channel.setStyleSheet("""
            QPushButton {
                background-color: #2e4a36;
                color: #ffffff;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #3c6648;
            }
        """)
        self.btn_add_channel.clicked.connect(self.add_channel)
        
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_add_channel)
        
        main_layout.addLayout(header_layout)
        
        # Channels Grid & Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #333333;
                min-height: 20px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #555555;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 14px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #333333;
                min-width: 20px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #555555;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("background-color: transparent;")
        
        self.channels_layout = QGridLayout(self.scroll_widget)
        self.channels_layout.setSpacing(15)
        self.channels_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(self.scroll_area, stretch=1)
        
        self.channels = []
        self.channel_counter = 0
        
        # Add a default channel at startup
        self.add_channel()
        
    def add_channel(self):
        if len(self.channels) >= 4:
            # We limit to 4 channels for layout and performance reasons
            self.btn_add_channel.setText("Max Channels Reached")
            self.btn_add_channel.setEnabled(False)
            self.btn_add_channel.setStyleSheet("""
                QPushButton {
                    background-color: #222222;
                    color: #555555;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: bold;
                    font-size: 15px;
                }
            """)
            return
            
        self.channel_counter += 1
        new_channel = VideoChannel(self.channel_counter)
        new_channel.closed_signal.connect(self.remove_channel)
        
        # Calculate grid position: (0,0), (0,1), (1,0), (1,1)
        idx = len(self.channels)
        row = idx // 2
        col = idx % 2
        
        self.channels_layout.addWidget(new_channel, row, col)
        self.channels.append(new_channel)
        
        if len(self.channels) >= 4:
            self.btn_add_channel.setText("Max Channels Reached")
            self.btn_add_channel.setEnabled(False)
            self.btn_add_channel.setStyleSheet("""
                QPushButton {
                    background-color: #222222;
                    color: #555555;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: bold;
                    font-size: 15px;
                }
            """)

    def remove_channel(self, channel):
        if channel in self.channels:
            self.channels.remove(channel)
            self.channels_layout.removeWidget(channel)
            channel.deleteLater()
            self.rearrange_channels()
            
            if len(self.channels) < 4:
                self.btn_add_channel.setText("+ Add Channel")
                self.btn_add_channel.setEnabled(True)
                self.btn_add_channel.setStyleSheet("""
                    QPushButton {
                        background-color: #2e4a36;
                        color: #ffffff;
                        border-radius: 8px;
                        padding: 10px 20px;
                        font-weight: bold;
                        font-size: 15px;
                    }
                    QPushButton:hover {
                        background-color: #3c6648;
                    }
                """)
                
    def rearrange_channels(self):
        for idx, ch in enumerate(self.channels):
            self.channels_layout.removeWidget(ch)
            row = idx // 2
            col = idx % 2
            self.channels_layout.addWidget(ch, row, col)
        
    def closeEvent(self, event):
        for ch in self.channels:
            ch.close_channel()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())