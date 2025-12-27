import cv2
import torch
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from track import Track,Tracker,prediction_function
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.ops import roi_align
import torch.nn.functional as F
import torchreid
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

def x(yolo_model):
    for i, layer in enumerate(yolo_model.model.model):
        print(i, layer)

def detection_and_featuremap(yolo_model, frame, layer_idx=8):
    features = {}

    def hook_fn(module, inp, out):
        features["map"] = out.detach() # Memory leak olmaması için detach ediyoruz

    # Kancayı ilgili katmana tak (Genelde 8. katman C3 bloğudur)
    handle = yolo_model.model.model[layer_idx].register_forward_hook(hook_fn)

    # 🔥 TEK PREDICT: Model burada çalışırken kanca veriyi yakalayacak
    results = yolo_model.predict(frame, imgsz=640, verbose=False, conf=0.3)

    # Kancayı kaldır (Her karede kanca birikmemesi için ŞART)
    handle.remove()

    return results, features.get("map")

@torch.no_grad()
def get_roi(
    feature_map,        # [1, C, Hf, Wf]  (YOLO layer output)
    bbox_xywh,          # [x, y, w, h]    (ORİJİNAL frame koordinatı)
    orig_w,
    orig_h,
    expand_ratio=1.5,   # bbox kaç kat büyütülecek
    max_size=128        # ROI max çözünürlük
):
    device = feature_map.device
    x, y, w, h = bbox_xywh

    # -----------------------------
    # BBOX'u büyüt (center-based)
    # -----------------------------
    cx = x + w / 2
    cy = y + h / 2

    nw = w * expand_ratio
    nh = h * expand_ratio

    x1 = cx - nw / 2
    y1 = cy - nh / 2
    x2 = cx + nw / 2
    y2 = cy + nh / 2

    # frame sınırlarına kırp
    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(1, min(x2, orig_w))
    y2 = max(1, min(y2, orig_h))

    # -----------------------------
    # ROI output size
    # -----------------------------
    roi_w = min(max_size, max(1, int(max_size * (x2 - x1) / orig_w)))
    roi_h = min(max_size, max(1, int(max_size * (y2 - y1) / orig_h)))

    # -----------------------------
    # ROI Align
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
def embedding(roi_feat):
    """
    roi_feat : torch.Tensor [1, C, H, W]  (ROI feature map)
    osnet    : pretrained OSNet model, linear head yerine Identity
    return   : np.array, normalized embedding [C_osnet]
    """
    # Global average pooling → [1, C]
    pooled = roi_feat.mean(dim=(2, 3))  # [1, C]
    emb = pooled
    # Normalize et
    emb = F.normalize(emb, dim=1)

    return emb.squeeze(0).cpu().numpy()  # [embedding_dim]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea
    return interArea / union if union > 0 else 0.0


def apply_highlight_test(frame, pred_bbox, alpha=0.5, expand_ratio=3.0):
    """Görüntüyü takip edilen drone'un tahmin edilen konumuna göre karartır/parlatır."""
    h, w, _ = frame.shape
    highlighted = (frame * alpha).astype(np.uint8)

    if pred_bbox is None:
        return frame  # Takip yoksa orijinal görüntüyü döndür

    px, py, pw, ph = pred_bbox
    # Tahmini kutuyu genişlet (Modelin etrafı görmesi için)
    nx1 = int(max(0, px - (pw * (expand_ratio - 1) / 2)))
    ny1 = int(max(0, py - (ph * (expand_ratio - 1) / 2)))
    nx2 = int(min(w, px + pw + (pw * (expand_ratio - 1) / 2)))
    ny2 = int(min(h, py + ph + (ph * (expand_ratio - 1) / 2)))

    highlighted[ny1:ny2, nx1:nx2] = frame[ny1:ny2, nx1:nx2]
    return highlighted


# --- prediction_function, Track ve Tracker sınıflarını buraya dahil ettiğini varsayıyorum ---
# (Yukarıda paylaştığın kodun tamamını buraya yapıştırabilirsin)

# DINOv2 model yükle (1 defa)


def load_resnet18_embedder(device="cuda"):
    # Pretrained ResNet18 yükle
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Son katmanı (sınıflandırma) kaldırarak sadece özellik çıkarıcı yapıyoruz
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    # ResNet için standart ön işleme
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


@torch.no_grad()
def resnet_embedding(model, preprocess, frame, bbox_xywh, device="cuda"):
    x, y, w, h = map(int, bbox_xywh)
    # Görüntüden drone'u kes (Crop)
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return np.zeros(512, dtype=np.float32)

    # OpenCV (BGR) -> PIL (RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    # Preprocess ve Inference
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    emb = model(input_tensor)

    # L2 Normalizasyonu (Cosine similarity için kritik)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze().cpu().numpy()

def test_on_video_sota(video_path):
    # 1. Model, Embedder ve Tracker Hazırlığı
    without_model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\without3\weights\best.pt'
    motion_model_path = r'C:\Users\USER\Desktop\DRONE\runs\detect\3_channel14\weights\best.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    without_model = YOLO(without_model_path)
    motion_model = YOLO(motion_model_path)

    #resnet_model, resnet_preprocess = load_resnet18_embedder(device)
    tracker = Tracker(similarity_threshold=0.5, max_missing=10)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Hata: Video dosyası açılamadı!")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- VİDEO KAYIT AYARLARI ---
    output_filename = "output6.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 formatı için codec
    fps = 30
    # Kaydedilecek videonun boyutu (Orijinal boyutları korumak en iyisidir)
    out = cv2.VideoWriter(output_filename, fourcc, fps, (orig_w, orig_h))

    print(f"İşlem başlıyor... Kayıt: {output_filename} ({orig_w}x{orig_h} @ {fps}fps)")


    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Track var mı kontrolü
        use_motion = (
                len(tracker.tracks) > 0

        )
        # 2. Model + input seçimi
        if use_motion:
            pred_bbox = prediction_function(tracker.get_best_track())
            input_frame = apply_highlight_test(frame, pred_bbox)
            model_to_use = motion_model

        else:
            input_frame = frame
            model_to_use = without_model

        # 3. YOLO inference
        results, feat_map = detection_and_featuremap(model_to_use, frame)

        detections = []
        embeddings = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]



                detections.append(bbox_xywh)
                #emb = resnet_embedding(resnet_model, resnet_preprocess, frame, bbox_xywh, device)


                # Drone ROI feature
                roi_feat = get_roi(feat_map,bbox_xywh,orig_w,orig_h)

                #  Embedding (512-dim)
                emb = embedding(roi_feat)

                embeddings.append(emb)
        # 3. Tracker Update
        active_tracks = tracker.update(detections, embeddings, frame_idx, orig_w, orig_h)

        # 4. Çizim (Kayıt edilecek kareye çiziyoruz)
        for t in active_tracks:
            # LOST ise hiçbir şey gösterme
            if t.missing_frames > 0:
                continue

            tx, ty, tw, th = map(int, t.bbox)
            color = (0, 255, 0)  # sadece aktif DRONE'lar

            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 2)
            label = f"ID:{t.track_id} DRONE"
            cv2.putText(frame, label, (tx, ty - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- ADIM 5: KAREYİ VİDEOYA YAZ ---
        out.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"{frame_idx} kare işlendi...")

    # Kaynakları serbest bırak (Burası çok önemli!)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"İşlem tamamlandı. Video kaydedildi: {output_filename}")


def test_pure_detection(video_path, model_path):
    # 1. Modeli ve Videoyu Yükle
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Video Yazma Ayarları (Opsiyonel: Sonucu kaydetmek istersen)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("pure_detection_test.mp4", fourcc, 30, (orig_w, orig_h))

    print("Saf Deteksiyon Testi Başlıyor...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Sadece YOLO Inference (Tracking yok, maskeleme yok)
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.3,
            verbose=False
        )

        # 3. Sonuçları Döngüye Al ve Çiz
        for r in results:
            for box in r.boxes:
                # Koordinatları al (Ham YOLO çıktısı)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().item()

                # Sadece kutuyu ve güven skorunu çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"DRONE {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Görüntüyü Kaydet ve Göster
        out.write(frame)
        cv2.imshow("Pure Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Test bitti. 'pure_detection_test.mp4' kaydedildi.")

if __name__ == "__main__":

    test_on_video_sota("visible.mp4")

