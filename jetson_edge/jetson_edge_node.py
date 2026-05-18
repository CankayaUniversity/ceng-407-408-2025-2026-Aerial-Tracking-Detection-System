import sys
import os
import cv2
import time
import socket
import struct
import json
import argparse
import numpy as np

# We ensure we ONLY import from the current directory, NOT from `core` which might crash
from tracker_rgb import Tracker as RGBTracker, prediction_function as rgb_prediction_function
from tracker_ir import Tracker as IRTracker, prediction_function as ir_prediction_function
from utils import apply_highlight_test as apply_highlight_rgb, apply_highlight_ir
from trt_infer import TRTYOLO

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

import threading

current_config = None
config_event = threading.Event()

def config_listener(sock):
    global current_config
    payload_size_struct = struct.calcsize("Q")
    data_buf = b""
    while True:
        try:
            while len(data_buf) < payload_size_struct:
                packet = sock.recv(4096)
                if not packet: return
                data_buf += packet
            packed_size = data_buf[:payload_size_struct]
            data_buf = data_buf[payload_size_struct:]
            payload_size = struct.unpack("Q", packed_size)[0]
            
            while len(data_buf) < payload_size:
                packet = sock.recv(4096)
                if not packet: return
                data_buf += packet
                
            payload = data_buf[:payload_size]
            data_buf = data_buf[payload_size:]
            
            json_size = struct.unpack("I", payload[:4])[0]
            json_bytes = payload[4:4+json_size]
            config_dict = json.loads(json_bytes.decode('utf-8'))
            
            if config_dict.get("type") == "config":
                current_config = config_dict
                config_event.set()
                print("[*] Received new remote config!")
        except Exception as e:
            print(f"[-] Config listener error: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Jetson Edge Node (Pure TensorRT) for Aerial Tracking")
    parser.add_argument("--video", type=str, default="0", help="Path to video file or camera index (default: 0)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between detections to save computation (default: 0)")
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

    # Start Config Listener
    listener_thread = threading.Thread(target=config_listener, args=(client_socket,), daemon=True)
    listener_thread.start()
    
    print("[*] Waiting for remote configuration from Main Hub...")
    config_event.wait()
    config_event.clear()

    # Video Setup
    is_live = args.video.isdigit()
    video_source = int(args.video) if is_live else args.video
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("[-] Failed to open video source.")
        sys.exit(1)

    try:
        while True:
            # Check if socket is still alive by checking if listener thread is alive
            if not listener_thread.is_alive():
                print("[-] Main Hub disconnected.")
                break

            global current_config
            mode = current_config.get("mode", "RGB")
            target_fps = float(current_config.get("fps", 30.0))
            conf_thresh = float(current_config.get("conf_thresh", 0.3))
            iou_thresh = float(current_config.get("iou_thresh", 0.45))
            no_video = current_config.get("no_video", False)
            skip_frames = int(current_config.get("skip_frames", args.skip_frames))
            use_dual_model = current_config.get("use_dual_model", True)

            # Engine Paths
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if mode.upper() == "RGB":
                base_model_path = os.path.join(base_dir, "models", "rgb_normal.engine")
                motion_model_path = os.path.join(base_dir, "models", "rgb_highlight.engine")
            else:
                base_model_path = os.path.join(base_dir, "models", "ir_normal.engine")
                motion_model_path = os.path.join(base_dir, "models", "ir_highlight.engine")

            print(f"[*] Loading TensorRT Engines for Mode: {mode}")
            without_model = TRTYOLO(base_model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            motion_model = TRTYOLO(motion_model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh)

            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if mode.upper() == "RGB":
                tracker = RGBTracker(similarity_threshold=0.5, max_missing=10, fps=target_fps)
            else:
                tracker = IRTracker(iou_threshold=0.1, max_missing=10, fps=target_fps)
            
            start_real_time = time.time()
            frame_idx = 0
            prev_gray = None
            prev_pts = None

            print(f"[*] Starting detection loop ({target_fps} FPS)...")
            
            while cap.isOpened():
                if config_event.is_set():
                    print("[*] Config changed, reloading...")
                    config_event.clear()
                    break # Break inner loop to restart with new config

                loop_start = time.time()
                
                if not is_live:
                    elapsed_real_time = time.time() - start_real_time
                    target_frame = int(elapsed_real_time * target_fps)
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    if current_frame < target_frame:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                
                t0 = time.time()
                ret, frame = cap.read()
                t_read = time.time() - t0
                if not ret:
                    if not is_live: # End of video
                        print("[*] End of video stream. Looping video...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        start_real_time = time.time()
                        continue
                    else:
                        break

                # Process Frame
                t0 = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downscale for optical flow to reduce CPU cost
                gray_small = cv2.resize(gray, (orig_w // 2, orig_h // 2))
                bg_dx, bg_dy = 0.0, 0.0
                
                if prev_gray is not None:
                    if prev_pts is None or len(prev_pts) < 10:
                        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                    if prev_pts is not None and len(prev_pts) > 0:
                        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_small, prev_pts, None)
                        if curr_pts is not None and status is not None:
                            status = status.flatten()
                            good_new = curr_pts[status == 1]
                            good_old = prev_pts[status == 1]
                            if len(good_new) > 0:
                                good_new_flat = good_new.reshape(-1, 2)
                                good_old_flat = good_old.reshape(-1, 2)
                                diffs = good_new_flat - good_old_flat
                                # Scale back to full resolution
                                bg_dx = np.median(diffs[:, 0]) * 2.0
                                bg_dy = np.median(diffs[:, 1]) * 2.0
                                prev_pts = good_new.reshape(-1, 1, 2)
                            else:
                                prev_pts = None
                prev_gray = gray_small
                t_flow = time.time() - t0
                
                t0 = time.time()
                run_detect = (skip_frames == 0) or (frame_idx % (skip_frames + 1) == 0)
                if mode.upper() == "RGB":
                    best_track = tracker.get_best_track() 
                    if best_track is not None and best_track.is_stationary(frame_window=int(2 * target_fps)):
                        tracker.reset()
                        best_track = None
                    
                    use_motion = use_dual_model and best_track is not None and best_track.missing_frames == 0
                    current_model_name = "Motion Model" if use_motion else "Base Model"
                    
                    if run_detect:
                        if use_motion:
                            pred_bboxes = rgb_prediction_function(best_track)
                            input_frame = apply_highlight_rgb(frame, pred_bboxes)
                            model = motion_model
                        else:
                            input_frame = frame
                            model = without_model
                        detections = model.predict(input_frame)
                    else:
                        detections = []  # tracker predicts on its own
                        
                    t_detect = time.time() - t0
                    t0 = time.time()
                    embeddings = [None] * len(detections)
                    tracker.update(detections, embeddings, frame_idx, orig_w, orig_h, bg_dx, bg_dy)
                else:
                    tracker.iou_threshold = 0.1
                    best_track = tracker.get_best_track()
                    if best_track is not None and best_track.is_stationary(frame_window=int(2 * target_fps)):
                        tracker.reset()
                        best_track = None
                    
                    use_motion = use_dual_model and best_track is not None and best_track.missing_frames == 0
                    current_model_name = "Motion Model" if use_motion else "Base Model"
                    
                    if run_detect:
                        if use_motion:
                            pred_bboxes = ir_prediction_function(best_track)
                            input_frame = apply_highlight_ir(frame, pred_bboxes)
                            model = motion_model
                        else:
                            input_frame = frame
                            model = without_model

                        detections = model.predict(input_frame)
                    else:
                        detections = []
                        
                    t_detect = time.time() - t0
                    t0 = time.time()
                    tracker.update(detections, frame_idx, frame_width=orig_w, frame_height=orig_h, bg_dx=bg_dx, bg_dy=bg_dy)
                t_track = time.time() - t0

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
                    
                    score = min(len(t.history), int(2 * target_fps)) / (2 * target_fps) * 0.5 + sim_val * 0.5
                    if score > best_active_score:
                        best_active_score = score
                        best_active = t

                loop_end = time.time()
                elapsed = loop_end - loop_start
                actual_fps = 1.0 / elapsed if elapsed > 0 else 0
                
                if frame_idx % 30 == 0:
                    total_ms = elapsed * 1000
                    print(
                        f"[PROFILE] Frame {frame_idx:4d} | "
                        f"Total: {total_ms:5.1f}ms ({actual_fps:.1f} FPS) | "
                        f"Read: {t_read*1000:4.1f}ms | "
                        f"Flow: {t_flow*1000:4.1f}ms | "
                        f"Detect: {t_detect*1000:5.1f}ms | "
                        f"Track: {t_track*1000:4.1f}ms | "
                        f"Tracks: {active_tracks_count}"
                    )
                
                telemetry = {
                    "fps": f"{actual_fps:.1f}",
                    "active_tracks": str(active_tracks_count),
                    "model": current_model_name,
                    "frame": str(frame_idx),
                    "device": f"EDGE (TRT)",
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
                json_bytes = json.dumps(telemetry).encode('utf-8')
                json_size = len(json_bytes)
                
                jpeg_bytes = b""
                if not no_video:
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
                    # Force exit to reconnect in outer bash loop if needed
                    sys.exit(1)
                    
                frame_idx += 1
                
    except KeyboardInterrupt:
        print("[*] Stopped by user.")
    finally:
        cap.release()
        client_socket.close()

if __name__ == "__main__":
    main()
