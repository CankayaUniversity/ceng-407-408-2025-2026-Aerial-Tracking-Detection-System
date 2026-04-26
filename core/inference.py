import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision.ops import roi_align

from core.tracker_rgb import Tracker, prediction_function
from core.utils import apply_highlight_test


# ======================================================
# YOLO FEATURE MAP
# ======================================================
def detection_and_featuremap(yolo_model, frame, layer_idx=8, conf=0.3):
    features = {}

    def hook_fn(module, inp, out):
        features["map"] = out.detach()

    handle = yolo_model.model.model[layer_idx].register_forward_hook(hook_fn)

    results = yolo_model.predict(frame, imgsz=640, verbose=False, conf=conf)

    handle.remove()
    return results, features.get("map")


# ======================================================
# ROI ALIGN
# ======================================================
@torch.no_grad()
def get_roi(feature_map, bbox_xywh, orig_w, orig_h, expand_ratio=1.5, max_size=128):
    device = feature_map.device
    x, y, w, h = bbox_xywh

    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * expand_ratio, h * expand_ratio

    x1, y1 = cx - nw / 2, cy - nh / 2
    x2, y2 = cx + nw / 2, cy + nh / 2

    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(1, min(x2, orig_w))
    y2 = max(1, min(y2, orig_h))

    roi_w = min(max_size, max(1, int(max_size * (x2 - x1) / orig_w)))
    roi_h = min(max_size, max(1, int(max_size * (y2 - y1) / orig_h)))

    _, _, Hf, Wf = feature_map.shape
    spatial_scale = min(Wf / orig_w, Hf / orig_h)

    rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=device)

    return roi_align(
        feature_map,
        rois,
        output_size=(roi_h, roi_w),
        spatial_scale=spatial_scale,
        aligned=True
    )


# ======================================================
# DEFAULT ROI EMBEDDING
# ======================================================
@torch.no_grad()
def roi_align_no_embedding(roi_feat):
    pooled = roi_feat.mean(dim=(2, 3))
    emb = F.normalize(pooled, dim=1)
    return emb.squeeze(0).cpu().numpy()


# ======================================================
# RESNET EMBEDDING
# ======================================================
from torchvision import models, transforms
from PIL import Image


def load_resnet18_embedder(device="cuda"):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return model, preprocess


@torch.no_grad()
def resnet_embedding(model, preprocess, frame, bbox_xywh, device="cuda"):
    x, y, w, h = map(int, bbox_xywh)

    crop = frame[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return np.zeros(512, dtype=np.float32)

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop)

    inp = preprocess(pil).unsqueeze(0).to(device)
    emb = model(inp)

    return F.normalize(emb, dim=1).squeeze().cpu().numpy()


# ======================================================
# MAIN SOTA TEST
# ======================================================
def test_on_video_sota(video_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    without_model_path = os.path.join(base_dir, 'runs', 'bestNormal', 'best.pt')
    motion_model_path = os.path.join(base_dir, 'runs', 'bestHigh', 'best.pt')

    without_model = YOLO(without_model_path)
    motion_model = YOLO(motion_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- SWITCH: RESNET ENABLE ----
    use_resnet = False  # True yaparsan ResNet embedding aktif olur

    if use_resnet:
        resnet_model, resnet_preprocess = load_resnet18_embedder(device)

    tracker = Tracker(similarity_threshold=0.5, max_missing=10)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı!")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        use_motion = len(tracker.tracks) > 0

        if use_motion:
            pred_bbox = prediction_function(tracker.get_best_track())
            input_frame = apply_highlight_test(frame, pred_bbox)
            model = motion_model
        else:
            input_frame = frame
            model = without_model

        results, feat_map = detection_and_featuremap(model, input_frame)

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

        tracker.update(detections, embeddings, frame_idx, orig_w, orig_h)

        for t in tracker.tracks:
            if t.missing_frames > 0:
                continue

            x, y, w, h = map(int, t.bbox)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{t.track_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        cv2.imshow("SOTA TEST", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()