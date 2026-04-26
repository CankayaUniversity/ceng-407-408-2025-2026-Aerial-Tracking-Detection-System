import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from core.tracker_ir import Tracker, prediction_function



def apply_highlight_test(frame, pred_bbox, intensity=0.5, expand_ratio=3.0):
    """
    IR Görüntüler için 1-Channel (Mavi) Highlight Fonksiyonu.
    Görüntüyü 3 kanala çevirir ve SADECE Channel 0'a (Mavi) fener tutar.
    Arka plan piksellerini (IR sinyallerini) KESİNLİKLE bozmaz.
    """

    # 1. KANALLARI AYARLA VE KOPYALA
    # Orijinal görüntüyü bozmamak için kopyasını alıyoruz
    # (Böylece ekranda izlerken mavi kutu gözümüzü yormaz, sadece YOLO görür)
    if len(frame.shape) == 2:
        # Eğer görüntü tek kanallıysa (Grayscale) 3 kanala (BGR) çevir
        input_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        # Zaten 3 kanallıysa (Örn: Video okuyucu BGR okumuşsa) sadece kopyala
        input_frame = frame.copy()

    if pred_bbox is None:
        return input_frame  # Takip yoksa RAW görüntüyü YOLO'ya yolla

    h, w, _ = input_frame.shape
    px, py, pw, ph = pred_bbox

    # 2. KUTUYU GENİŞLET (3x Oranında)
    # Merkez noktasını bulup kutuyu her yöne eşit genişletiyoruz
    cx, cy = px + (pw / 2), py + (ph / 2)
    new_w, new_h = pw * expand_ratio, ph * expand_ratio

    nx1 = int(max(0, cx - (new_w / 2)))
    ny1 = int(max(0, cy - (new_h / 2)))
    nx2 = int(min(w, cx + (new_w / 2)))
    ny2 = int(min(h, cy + (new_h / 2)))

    # 3. 1-CHANNEL HİGHLIGHT (SADECE MAVİ KANALA MÜDAHALE)
    # Sadece Kanal 0'ı (Blue) al, üzerine intensity (ışık) ekle
    blue_ch = input_frame[ny1:ny2, nx1:nx2, 0].astype(np.float32)
    blue_ch += (255 * intensity)

    # 255'i aşmaması için clip yap ve geri koy
    input_frame[ny1:ny2, nx1:nx2, 0] = np.clip(blue_ch, 0, 255).astype(np.uint8)

    return input_frame



# ======================================================
# MAIN SOTA TEST
# ======================================================
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Kendi yazdığın modüller (Tracker ve prediction_function güncellenmiş haliyle)
from core.tracker_rgb import Tracker, prediction_function
from core.utils import apply_highlight_test


# ======================================================
# MAIN SOTA TEST - IR OPTIMIZED (EDGE DEPLOYMENT READY)
# ======================================================
def test_on_video_sota(video_path):
    print("[INFO] Modeller Yükleniyor...")
    without_model = YOLO(r'C:\Users\USER\Desktop\DRONE\infrared\runs\detect\initial_startup_model\weights\best.pt')
    motion_model = YOLO(r'C:\Users\USER\Desktop\DRONE\infrared\runs\detect\final_43k_rect_model2\weights\best.pt')

    # Tracker başlat (Yeni versiyonda max_missing ve iou_threshold kullanıyoruz, embedding yok)
    tracker = Tracker(iou_threshold=0.3, max_missing=10)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[HATA] Video açılamadı!")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. HIGHLIGHT STRATEJİSİ
        best_track = tracker.get_best_track()

        # Eğer aktif ve kaybolmamış bir hedefimiz varsa motion_model kullan
        use_motion = best_track is not None and best_track.missing_frames == 0

        if use_motion:
            # Tracker'ın regresyon tahminini al
            pred_bbox = prediction_function(best_track)
            # Resme 3x boyutunda MAVİ HİGHLIGHT kutusunu uygula
            input_frame = apply_highlight_test(frame, pred_bbox)
            model = motion_model
        else:
            # Hedef yoksa veya ilk defa aranıyorsa saf görüntüyle RAW model kullan
            input_frame = frame
            model = without_model

        # 2. YOLO TAHMİNİ (Feature Map vs YOK, Sadece BBOX)
        # verbose=False yaparak terminali gereksiz yazılardan kurtarıyoruz (hızlandırır)
        results = model.predict(input_frame, imgsz=640, verbose=False, conf=0.3)

        detections = []

        # 3. KUTULARI TOPLA
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h] formatına çevir
                detections.append(bbox)

        # 4. TRACKER GÜNCELLEMESİ (Sadece konumlarla)
        tracker.update(detections, frame_idx, frame_width=orig_w, frame_height=orig_h)

        # 5. GÖRSELLEŞTİRME
        for t in tracker.tracks:
            # Eğer drone bulutun arkasına girdiyse (missing_frames > 0), ekrana kutu çizme
            # (Arka planda regresyon tahmin etmeye devam ediyor ama sahte kutu göstermiyoruz)
            if t.missing_frames > 0:
                continue

            x, y, w, h = map(int, t.bbox)

            # Hedefin etrafına yeşil takip kutusu çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{t.track_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        # İPUCU: Eğer "input_frame" yazarsan, Mavi Highlight Kutusunu da ekranda görürsün.
        # "frame" yazarsan mavi kutu gizli kalır, sadece yeşil sonuç kutusunu görürsün.
        cv2.imshow("SOTA TEST - IR OPTIMIZED", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()