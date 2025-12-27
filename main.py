import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.ops import roi_align
from track import Tracker, prediction_function
from ultralytics import YOLO
from utils import *


# ======================================================
# MAIN FRAME-BASED TEST
# ======================================================
def test_metrics(
        image_root="dataset_rgb/images/test",
        label_root="dataset_rgb/labels/test"
):
    # Model init (once)
    without_model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\without3\weights\best.pt'
    motion_model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\3_channel14\weights\best.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    without_model = YOLO(without_model_path)
    motion_model = YOLO(motion_model_path)

    prev_video = None
    prev_match = {}
    prev_frame_id = None
    frame_idx = 0
    # For global average
    global_results = []

    # Video-based statistics
    stats = {"TP": 0, "FP": 0, "FN": 0, "IDSW": 0, "GT": 0}
    state_history = []
    images = sorted(os.listdir(image_root))

    try:
        for img_name in images:

            video_key = "_".join(img_name.split("_")[:4])
            frame_id = int(img_name.split("_")[-1].replace(".jpg", ""))

            if prev_frame_id is not None:
                gap = frame_id - prev_frame_id - 1
                if gap > 0:
                    for _ in range(gap):
                        tracker.update([], [], frame_idx, 1, 1)  # empty frame
                        frame_idx += 1
            else:
                gap = 0

            # ---------------- VIDEO CHANGED ----------------
            if prev_video is not None and video_key != prev_video:
                res = finalize_metrics(stats, state_history)
                print(f"\n📹 Video: {prev_video}")
                for k, v in res.items():
                    print(f"{k}: {v:.4f}")

                global_results.append(res)

                # reset
                tracker.reset()
                prev_match.clear()
                state_history = []
                prev_frame_id = None
                frame_idx = 0
                stats = {"TP": 0, "FP": 0, "FN": 0, "IDSW": 0, "GT": 0}

            prev_video = video_key

            # ---------------- FRAME LOAD ----------------
            img_path = os.path.join(image_root, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            label_path = os.path.join(label_root, img_name.replace(".jpg", ".txt"))
            gt_exist = os.path.exists(label_path)

            gt_bbox = None
            if gt_exist:
                with open(label_path) as f:
                    _, cx, cy, w, h = map(float, f.readline().split())
                H, W = frame.shape[:2]
                gt_bbox = [
                    (cx - w / 2) * W,
                    (cy - h / 2) * H,
                    w * W,
                    h * H
                ]
                stats["GT"] += 1

            # ---------------- MODEL SELECTION ----------------
            use_motion = len(tracker.tracks) > 0
            model = motion_model if use_motion else without_model
            input_frame = frame

            results, feat_map = detection_and_featuremap(model, input_frame)

            detections, embeddings = [], []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append(bbox)

                    roi = get_roi(feat_map, bbox, W, H)
                    embeddings.append(roi_align_no_embedding(roi))

            tracks = tracker.update(detections, embeddings, frame_idx, W, H)

            # ---------------- METRIC ----------------
            best_iou, best_tr = 0, None
            if gt_bbox:
                for tr in tracks:
                    i = iou(tr.bbox, gt_bbox)
                    if i > best_iou:
                        best_iou, best_tr = i, tr

            if gt_exist:
                if best_iou >= 0.5:
                    stats["TP"] += 1
                    if video_key in prev_match and prev_match[video_key] != best_tr.track_id:
                        stats["IDSW"] += 1
                    prev_match[video_key] = best_tr.track_id
                else:
                    stats["FN"] += 1
            else:
                stats["FP"] += len(detections)

            if best_tr is None:
                state_history.append({
                    "exist": 1 if gt_exist else 0,
                    "gt_bbox": gt_bbox,
                    "pred": None,
                    "p": 1.0
                })
            else:
                state_history.append({
                    "exist": 1 if gt_exist else 0,
                    "gt_bbox": gt_bbox,
                    "pred": {"bbox": best_tr.bbox},
                    "p": 0.0
                })

            prev_frame_id = frame_id
            frame_idx += 1

        # ---------------- LAST VIDEO ----------------
        if prev_video is not None:
            res = finalize_metrics(stats, state_history)
            print(f"\n📹 Video: {prev_video}")
            for k, v in res.items():
                print(f"{k}: {v:.4f}")
            global_results.append(res)

    except KeyboardInterrupt:
        print("\n User stopped")

    # ---------------- AVERAGE ----------------
    print("\n===== GENERAL AVERAGE =====")
    for k in global_results[0]:
        avg = sum(r[k] for r in global_results) / len(global_results)
        print(f"{k}: {avg:.4f}")


def detection_and_featuremap(yolo_model, frame, layer_idx=8):
    features = {}

    def hook_fn(module, inp, out):
        features["map"] = out.detach()  # Detach to prevent memory leak

    # Hook to the relevant layer (Usually 8th layer is C3 block)
    handle = yolo_model.model.model[layer_idx].register_forward_hook(hook_fn)

    # SINGLE PREDICT: Hook will capture data while model runs here
    results = yolo_model.predict(frame, imgsz=640, verbose=False, conf=0.3)

    # Remove hook (NECESSARY to prevent hook accumulation per frame)
    handle.remove()

    return results, features.get("map")


@torch.no_grad()
def get_roi(
        feature_map,  # [1, C, Hf, Wf]  (YOLO layer output)
        bbox_xywh,  # [x, y, w, h]    (ORIGINAL frame coordinate)
        orig_w,
        orig_h,
        expand_ratio=1.5,  # how many times to expand bbox
        max_size=128  # ROI max resolution
):
    device = feature_map.device
    x, y, w, h = bbox_xywh

    # -----------------------------
    #  Expand BBOX (center-based)
    # -----------------------------
    cx = x + w / 2
    cy = y + h / 2

    nw = w * expand_ratio
    nh = h * expand_ratio

    x1 = cx - nw / 2
    y2 = cy - nh / 2
    x2 = cx + nw / 2
    y2 = cy + nh / 2

    # clip to frame boundaries
    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(1, min(x2, orig_w))
    y2 = max(1, min(y2, orig_h))

    # -----------------------------
    # 2️⃣ ROI output size
    # -----------------------------
    roi_w = min(max_size, max(1, int(max_size * (x2 - x1) / orig_w)))
    roi_h = min(max_size, max(1, int(max_size * (y2 - y1) / orig_h)))

    # -----------------------------
    # 3️⃣ ROI Align
    # -----------------------------
    _, _, Hf, Wf = feature_map.shape
    spatial_scale = min(Wf / orig_w, Hf / orig_h)

    rois = torch.tensor(
        [[0, x1, y1, x2, y2]],
        dtype=torch.float32,
        device=device
    )

    roi_feat = roi_align(
        feature_map,
        rois,
        output_size=(roi_h, roi_w),
        spatial_scale=spatial_scale,
        aligned=True
    )

    return roi_feat  # [1, 512, h, w]


import torch


@torch.no_grad()
def roi_align_no_embedding(roi_feat):
    """
    roi_feat : torch.Tensor [1, C, H, W]  (ROI feature map)
    osnet    : pretrained OSNet model, linear head yerine Identity
    return   : np.array, normalized embedding [C_osnet]
    """
    # Global average pooling → [1, C]
    pooled = roi_feat.mean(dim=(2, 3))  # [1, C]
    emb = pooled
    # Normalize it
    emb = F.normalize(emb, dim=1)

    return emb.squeeze(0).cpu().numpy()  # [embedding_dim]


def load_resnet18_embedder(device="cuda"):
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    # Standard preprocessing for ResNet
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


@torch.no_grad()
def resnet_embedding(model, preprocess, frame, bbox_xywh, device="cuda"):
    x, y, w, h = map(int, bbox_xywh)
    # Crop the drone from the image
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return np.zeros(512, dtype=np.float32)

    # OpenCV (BGR) -> PIL (RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    # Preprocess and Inference
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    emb = model(input_tensor)

    # L2 Normalization (for Cosine similarity)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze().cpu().numpy()


def test_on_video_sota(video_path):
    # 1. Model, Embedder and Tracker Preparation
    without_model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\without3\weights\best.pt'
    motion_model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\3_channel14\weights\best.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    without_model = YOLO(without_model_path)
    motion_model = YOLO(motion_model_path)

    # resnet_model, resnet_preprocess = load_resnet18_embedder(device)
    tracker = Tracker(similarity_threshold=0.5, max_missing=10)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file could not be opened!")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- VIDEO RECORDING SETTINGS ---
    output_filename = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    fps = 30
    # Size of the video to be recorded
    out = cv2.VideoWriter(output_filename, fourcc, fps, (orig_w, orig_h))

    print(f"Processing starting... Recording: {output_filename} ({orig_w}x{orig_h} @ {fps}fps)")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Check if track exists
        use_motion = (
                len(tracker.tracks) > 0

        )
        # 2. Model + input selection
        if use_motion:
            pred_bbox = prediction_function(tracker.get_best_track())
            input_frame = apply_highlight_test(frame, pred_bbox)
            model_to_use = motion_model

        else:
            input_frame = frame
            model_to_use = without_model

        # 3. YOLO inference
        results, feat_map = detection_and_featuremap(model_to_use, input_frame)

        detections = []
        embeddings = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

                detections.append(bbox_xywh)
                # emb = resnet_embedding(resnet_model, resnet_preprocess, frame, bbox_xywh, device)   #use this for resnet embedding

                # Drone ROI feature
                roi_feat = get_roi(feat_map, bbox_xywh, orig_w, orig_h)  # disable this if using resnet embedder

                # Embedding (512-dim)
                emb = roi_align_no_embedding(roi_feat)  # disable this if using resnet embedder

                embeddings.append(emb)
        # 3. Tracker Update
        active_tracks = tracker.update(detections, embeddings, frame_idx, orig_w, orig_h)

        # 4. Drawing
        for t in active_tracks:
            # If LOST, show nothing
            if t.missing_frames > 0:
                continue

            tx, ty, tw, th = map(int, t.bbox)
            color = (0, 255, 0)  # only active DRONES

            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 2)
            label = f"ID:{t.track_id} DRONE"
            cv2.putText(frame, label, (tx, ty - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- STEP 5: WRITE FRAME TO VIDEO ---
        out.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"{frame_idx} frames processed...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Process completed. Video saved: {output_filename}")


def test_pure_detection(video_path, model_path):
    # 1. Load Model and Video
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Video Writing Settings (Optional: If you want to save the result)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("pure_detection_test.mp4", fourcc, 30, (orig_w, orig_h))

    print("Pure Detection Test Starting...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Only YOLO Inference (No tracking, no masking)
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.3,
            verbose=False
        )

        # 3. Loop through results and draw
        for r in results:
            for box in r.boxes:
                # Get coordinates (Raw YOLO output)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().item()

                # Draw only the box and confidence score
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"DRONE {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Save and Display Image
        out.write(frame)
        cv2.imshow("Pure Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Test finished. 'pure_detection_test.mp4' saved.")


if __name__ == "__main__":
    # test_on_video(r"C:\Users\USER\Desktop\DRONE\dataset\test\20190925_111757_1_2\visible.mp4")
    test_metrics()
