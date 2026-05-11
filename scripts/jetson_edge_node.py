import sys
import os
import cv2
import time
import socket
import struct
import json
import argparse
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path so we can import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from core.tracker_rgb import Tracker as RGBTracker, prediction_function as rgb_prediction_function
from core.tracker_ir import Tracker as IRTracker, prediction_function as ir_prediction_function
from core.utils import apply_highlight_test as apply_highlight_rgb
from core.inference import detection_and_featuremap

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

def discover_server(udp_port=50050, timeout=30):
    print(f"[*] Listening for Main Hub broadcast on UDP port {udp_port}...")
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    udp_socket.bind(("", udp_port))
    udp_socket.settimeout(timeout)
    
    try:
        while True:
            data, addr = udp_socket.recvfrom(1024)
            msg = data.decode('utf-8')
            if msg.startswith("AERIAL_TRACKER_MAIN_HUB:"):
                _, port_str = msg.split(":")
                server_ip = addr[0]
                server_port = int(port_str)
                print(f"[+] Found Main Hub at {server_ip}:{server_port}")
                return server_ip, server_port
    except socket.timeout:
        print("[-] Discovery timed out.")
        return None, None
    finally:
        udp_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Jetson Edge Node for Aerial Tracking")
    parser.add_argument("--video", type=str, default="0", help="Path to video file or camera index (default: 0)")
    parser.add_argument("--no-video", action="store_true", help="Send only telemetry, do not send video frames to save bandwidth/CPU")
    parser.add_argument("--mode", type=str, default="RGB", choices=["RGB", "IR"], help="Processing mode: RGB or IR")
    parser.add_argument("--base-model", type=str, default=None, help="Path to base YOLO model/engine (overrides default)")
    parser.add_argument("--motion-model", type=str, default=None, help="Path to motion YOLO model/engine (overrides default)")
    args = parser.parse_args()

    # Auto-discover main server
    server_ip, server_port = discover_server()
    if not server_ip:
        print("Could not find Main Hub. Exiting.")
        sys.exit(1)

    # Connect TCP
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[*] Connecting to {server_ip}:{server_port} via TCP...")
    try:
        client_socket.connect((server_ip, server_port))
        print("[+] Connected to Main Hub.")
    except Exception as e:
        print(f"[-] Connection failed: {e}")
        sys.exit(1)

    # Load YOLO models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.mode.upper() == "RGB":
        base_model_path = args.base_model or os.path.join(base_dir, "models", "rgb_normal.pt")
        motion_model_path = args.motion_model or os.path.join(base_dir, "models", "rgb_highlight.pt")
    else:
        base_model_path = args.base_model or os.path.join(base_dir, "models", "ir_normal.pt")
        motion_model_path = args.motion_model or os.path.join(base_dir, "models", "ir_highlight.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading models on {device}...")
    without_model = YOLO(base_model_path, task='detect')
    motion_model = YOLO(motion_model_path, task='detect')

    # Video Setup
    is_live = args.video.isdigit()
    video_source = int(args.video) if is_live else args.video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("[-] Failed to open video source.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.mode.upper() == "RGB":
        tracker = RGBTracker(similarity_threshold=0.5, max_missing=10, fps=fps)
    else:
        tracker = IRTracker(iou_threshold=0.1, max_missing=10, fps=fps)
    
    start_real_time = time.time()
    frame_idx = 0
    prev_gray = None
    prev_pts = None

    print("[*] Starting detection loop...")
    
    try:
        while cap.isOpened():
            loop_start = time.time()
            
            # Real-time synchronization for video files
            if not is_live:
                elapsed_real_time = time.time() - start_real_time
                target_frame = int(elapsed_real_time * fps)
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                if current_frame < target_frame:
                    # Skip to the target frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = cap.read()
            if not ret:
                print("[*] End of video stream.")
                break

            # Process Frame
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
                            diffs = good_new - good_old
                            bg_dx = np.median(diffs[:, 0])
                            bg_dy = np.median(diffs[:, 1])
                            prev_pts = good_new.reshape(-1, 1, 2)
                        else:
                            prev_pts = None
            prev_gray = gray
            
            if args.mode.upper() == "RGB":
                best_track = tracker.get_best_track() 
                if best_track is not None and best_track.is_stationary(frame_window=int(2 * fps)):
                    tracker.reset()
                    best_track = None
                
                use_motion = best_track is not None and best_track.missing_frames == 0
                current_model_name = "Motion Model" if use_motion else "Base Model"
                if use_motion:
                    pred_bboxes = rgb_prediction_function(best_track)
                    input_frame = apply_highlight_rgb(frame, pred_bboxes)
                    model = motion_model
                else:
                    input_frame = frame
                    model = without_model
                
                results, feat_map = detection_and_featuremap(model, input_frame, conf=0.30)
                detections, embeddings = [], []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append(bbox)
                        # We skip ROI embedder for edge speed unless needed
                        embeddings.append(None)
                tracker.update(detections, embeddings, frame_idx, orig_w, orig_h, bg_dx, bg_dy)
            else:
                tracker.iou_threshold = 0.1
                best_track = tracker.get_best_track()
                if best_track is not None and best_track.is_stationary(frame_window=int(2 * fps)):
                    tracker.reset()
                    best_track = None
                
                use_motion = best_track is not None and best_track.missing_frames == 0
                current_model_name = "Motion Model" if use_motion else "Base Model"
                
                if use_motion:
                    pred_bboxes = ir_prediction_function(best_track)
                    input_frame = apply_highlight_ir(frame, pred_bboxes)
                    model = motion_model
                else:
                    input_frame = frame
                    model = without_model

                results = model.predict(input_frame, imgsz=640, verbose=False, conf=0.30)
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append(bbox)
                        
                tracker.update(detections, frame_idx, frame_width=orig_w, frame_height=orig_h, bg_dx=bg_dx, bg_dy=bg_dy)

            active_tracks_count = 0
            best_active = None
            best_active_score = -1.0
            
            all_tracks_info = []
            
            for t in tracker.tracks:
                if t.missing_frames > 0: continue
                active_tracks_count += 1
                x, y, w, h = map(int, t.bbox)
                sim_val = getattr(t, 'sim', 0.0)
                all_tracks_info.append({
                    "id": t.track_id, 
                    "bbox": [x, y, w, h], 
                    "sim": float(sim_val)
                })
                
                score = min(len(t.history), int(2 * fps)) / (2 * fps) * 0.5 + sim_val * 0.5
                if score > best_active_score:
                    best_active_score = score
                    best_active = t

            # Telemetry dict
            loop_end = time.time()
            elapsed = loop_end - loop_start
            actual_fps = 1.0 / elapsed if elapsed > 0 else 0
            
            telemetry = {
                "fps": f"{actual_fps:.1f}",
                "active_tracks": str(active_tracks_count),
                "model": current_model_name,
                "frame": str(frame_idx),
                "device": f"EDGE ({device.upper()})",
                "tracks": all_tracks_info
            }

            if best_active is not None:
                bx, by, bw, bh = best_active.bbox
                tcx = int(bx + bw / 2)
                tcy = int(by + bh / 2)
                vel = 0.0
                if len(best_active.history) >= 2:
                    prev_b = best_active.history[-2]
                    vel = np.hypot((bx + bw/2) - (prev_b[0] + prev_b[2]/2),
                                   (by + bh/2) - (prev_b[1] + prev_b[3]/2))
                telemetry.update({
                    "target_id": str(best_active.track_id),
                    "target_pos": f"({tcx}, {tcy})",
                    "target_size": f"{int(bw)}x{int(bh)}",
                    "target_sim": f"{getattr(best_active, 'sim', 0.0):.2f}",
                    "target_vel": f"{vel:.1f} px/f",
                    "track_age": str(len(best_active.history)),
                })

            # Send over network
            # Protocol: [Payload Size 8 bytes] [JSON Size 4 bytes] [JSON Bytes] [JPEG Bytes]
            json_bytes = json.dumps(telemetry).encode('utf-8')
            json_size = len(json_bytes)
            
            jpeg_bytes = b""
            if not args.no_video:
                _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_bytes = encoded.tobytes()
                
            payload_size = 4 + json_size + len(jpeg_bytes)
            header = struct.pack("Q", payload_size)
            json_header = struct.pack("I", json_size)
            
            try:
                client_socket.sendall(header)
                client_socket.sendall(json_header)
                client_socket.sendall(json_bytes)
                if len(jpeg_bytes) > 0:
                    client_socket.sendall(jpeg_bytes)
            except Exception as e:
                print(f"[-] Connection lost: {e}")
                break
                
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("[*] Stopped by user.")
    finally:
        cap.release()
        client_socket.close()

if __name__ == "__main__":
    main()
