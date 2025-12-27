import os
import json
import shutil
import cv2
import os
import shutil
from pathlib import Path

def organize_dataset(input_base_dir="dataset", output_base_dir="dataset_rgb"):
    """
    Anti-UAV datasetini YOLOv8 formatına dönüştürür.
    - Dinamik olarak her görüntünün çözünürlüğünü okur ve bbox'ları normalize eder.
    - images/train, labels/train şeklinde klasörleri oluşturur.
    """

    splits = ['train', 'val', 'test']

    for split in splits:
        input_dir = os.path.join(input_base_dir, split)
        image_out_dir = os.path.join(output_base_dir, f"images/{split}")
        label_out_dir = os.path.join(output_base_dir, f"labels/{split}")

        os.makedirs(image_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        print(f"\nStarting: {split.upper()} set")

        video_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

        for video_folder in sorted(video_folders):
            video_path = os.path.join(input_dir, video_folder)
            label_path = os.path.join(video_path, "visible.json")

            if not os.path.exists(label_path):
                print(f"[!] Missing annotation file: {label_path}")
                continue

            with open(label_path, "r") as f:
                label_data = json.load(f)

            exists = label_data.get("exist", [])
            gt_rect = label_data.get("gt_rect", [])
            rect_index = 0

            print(f"\nProcessing {split}/{video_folder} ({len(exists)} frames)...")

            for i in range(len(exists)):
                original_jpg = os.path.join(video_path, "visible", f"visibleI{i:04d}.jpg")
                new_basename = f"{video_folder}_{i:04d}"
                new_image_path = os.path.join(image_out_dir, f"{new_basename}.jpg")
                new_label_path = os.path.join(label_out_dir, f"{new_basename}.txt")

                if not os.path.exists(original_jpg):
                    print(f"Missing image: {original_jpg}")
                    continue

                # Görüntüyü oku ve çözünürlüğünü al
                img = cv2.imread(original_jpg)
                if img is None:
                    print(f"Could not read image: {original_jpg}")
                    continue
                IMG_HEIGHT, IMG_WIDTH = img.shape[:2]

                shutil.copy(original_jpg, new_image_path)

                if exists[i] == 1:
                    if rect_index < len(gt_rect) and gt_rect[rect_index]:
                        try:
                            x, y, w, h = gt_rect[rect_index]
                            rect_index += 1
                        except ValueError:
                            print(f"Empty bbox at frame {i}, skipping.")
                            continue
                    else:
                        print(f"No bbox for frame {i}, skipping.")
                        continue

                    # Dinamik çözünürlük ile normalize et
                    cx = (x + w / 2) / IMG_WIDTH
                    cy = (y + h / 2) / IMG_HEIGHT
                    nw = w / IMG_WIDTH
                    nh = h / IMG_HEIGHT

                    # YOLO formatında label dosyası
                    with open(new_label_path, "w") as lf:
                        lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            print(f"Finished: {split}/{video_folder}")

        print(f"\nCompleted {split.upper()} set.")

    print("\nAll dataset splits processed successfully.")


# Çalıştır
#organize_dataset()




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


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea
    return interArea / union if union > 0 else 0.0

def model_layer_debug(yolo_model):
    #see model layers
    for i, layer in enumerate(yolo_model.model.model):
        print(i, layer)

# ======================================================
# FRAME NAME PARSER
# ======================================================
def parse_frame_name(fname):
    base = fname.replace(".jpg", "")
    p = base.split("_")
    date = p[0]
    time = p[1]
    cam  = int(p[2])
    seq  = int(p[3])
    frame= int(p[4])
    video_key = f"{date}_{time}"
    return video_key, cam, seq, frame


# ======================================================
# LABEL LOADER
# ======================================================
def norm_cxcywh_to_xywh(norm, W, H):
    cx, cy, w, h = norm
    bw = w * W
    bh = h * H
    x = (cx * W) - bw / 2
    y = (cy * H) - bh / 2
    return [x, y, bw, bh]


def load_label_for_frame(img_name, label_root, W, H):
    label_path = os.path.join(
        label_root, img_name.replace(".jpg", ".txt")
    )

    if not os.path.exists(label_path):
        return {"exist": 0, "bbox": None}

    with open(label_path, "r") as f:
        line = f.readline().strip().split()

    if len(line) != 5:
        return {"exist": 0, "bbox": None}

    _, cx, cy, w, h = map(float, line)
    bbox = norm_cxcywh_to_xywh([cx, cy, w, h], W, H)

    return {"exist": 1, "bbox": bbox}

def finalize_metrics(stats, state_history):
    FP, FN, IDSW, GT, TP = stats["FP"], stats["FN"], stats["IDSW"], stats["GT"], stats["TP"]

    mota = 1 - (FP + FN + IDSW) / max(GT, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    state_acc = compute_state_accuracy(state_history)

    return {
        "MOTA": mota,
        "Precision": precision,
        "Recall": recall,
        "StateAcc": state_acc
    }

# ======================================================
# IOU
# ======================================================
def iou_xywh(b1, b2):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    xa = max(x1,x2)
    ya = max(y1,y2)
    xb = min(x1+w1, x2+w2)
    yb = min(y1+h1, y2+h2)
    inter = max(0,xb-xa) * max(0,yb-ya)
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0


# ======================================================
# STATE ACCURACY
# ======================================================
def compute_state_accuracy(history):
    T = len(history)
    T_star = 0
    acc_sum = 0.0
    empty_penalty = 0.0

    for h in history:
        exist = h["exist"]
        gt_bbox = h["gt_bbox"]
        pred = h["pred"]
        p = h["p"]

        if exist > 0:  # GT var
            T_star += 1
            if pred is not None:
                acc_sum += iou_xywh(pred["bbox"], gt_bbox)
            empty_penalty += p
        else:  # GT yok
            acc_sum += p

    return (acc_sum / max(T, 1)) - 0.2 * ((empty_penalty / max(T_star, 1)) ** 0.3)

def sample_dataset(source_root='dataset_rgb', target_root='sampledataset2', step=5):
    splits = ['train', 'val', 'test']
    max_limits = {
        'train': float('inf'),
        'val': 2000,
        'test': 2000
    }

    # Hedef boyutlar
    TARGET_SIZE = (640, 512)  # (Genişlik, Yükseklik)

    for split in splits:
        source_images = Path(source_root) / 'images' / split
        source_labels = Path(source_root) / 'labels' / split
        target_images = Path(target_root) / 'images' / split
        target_labels = Path(target_root) / 'labels' / split

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        all_images = sorted(source_images.glob('*.jpg'))

        videos = {}
        for img_path in all_images:
            name = img_path.stem
            frameid_str = name[-4:]
            if not frameid_str.isdigit(): continue
            frameid = int(frameid_str)
            videoname = name[:-5]
            if videoname not in videos: videos[videoname] = []
            videos[videoname].append((frameid, img_path))

        current_count = 0
        limit = max_limits[split]

        print(f"{split} işleniyor ve yeniden boyutlandırılıyor... (Limit: {limit})")

        for videoname, frames in videos.items():
            if current_count >= limit:
                break

            frames_sorted = sorted(frames, key=lambda x: x[0])
            for frameid, img_path in frames_sorted:
                if current_count >= limit:
                    break

                if frameid % step == 0:
                    # Görüntüyü oku ve Resize et
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        # Yeni konuma kaydet
                        cv2.imwrite(str(target_images / img_path.name), resized_img)

                        # Etiketi kopyala (Koordinatlar normalize olduğu için değişmez)
                        label_path = source_labels / f"{videoname}_{frameid:04d}.txt"
                        if label_path.exists():
                            shutil.copy(label_path, target_labels / label_path.name)

                        current_count += 1

        print(f"--- {split} tamamlandı. Toplam: {current_count}")

    print(f"Sample dataset oluşturuldu ve görüntüler {TARGET_SIZE} boyutuna getirildi: {target_root}")


# Kullanım
#sample_dataset()


def create_azdata_set(source_root='dataset_rgb', target_root='azdata', total_limit=1000):
    """
    source_root: Orijinal devasa datasetin yolu
    target_root: Yeni oluşturulacak küçük klasör
    total_limit: Toplamda alınacak maksimum resim sayısı
    """
    splits = ['train', 'val', 'test']

    # Her split (train/val/test) için toplam limitin bir kısmını ayıralım
    # Örn: 700 train, 150 val, 150 test
    split_limits = {
        'train': int(total_limit * 0.7),
        'val': int(total_limit * 0.15),
        'test': int(total_limit * 0.15)
    }

    for split in splits:
        source_images = Path(source_root) / 'images' / split
        source_labels = Path(source_root) / 'labels' / split

        target_images = Path(target_root) / 'images' / split
        target_labels = Path(target_root) / 'labels' / split

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        all_images = sorted(source_images.glob('*.jpg'))

        count = 0
        limit = split_limits[split]

        # Eğer kaynakta yeterli resim yoksa elindekini kullanması için step hesaplayalım
        # Çok hızlı dolmaması için her n resimde bir alalım (seyreltme)
        step = max(1, len(all_images) // limit)

        print(f"{split} klasörü işleniyor... Hedef: {limit} resim.")

        for i in range(0, len(all_images), step):
            if count >= limit:
                break

            img_path = all_images[i]
            # Resmi kopyala
            shutil.copy(img_path, target_images / img_path.name)

            # Etiketi bul ve kopyala
            label_name = img_path.stem + ".txt"
            label_path = source_labels / label_name
            if label_path.exists():
                shutil.copy(label_path, target_labels / label_path.name)

            count += 1

    print(f"İşlem tamam! 'azdata' klasörüne toplam yaklaşık {total_limit} resim ve etiket kopyalandı.")

# Çalıştırmak için:
#create_azdata_set()


def sample_dataset_by_video(source_root='dataset_rgb', target_root='sampledataset3', step=5, num_videos=15):
    splits = ['train', 'val', 'test']
    max_limits = {
        'train': float('inf'),  # Train için sınır koymuyoruz, ilk 15 videonun tamamını alır
        'val': 500,
        'test': 500
    }

    TARGET_SIZE = (640, 512)

    for split in splits:
        source_images = Path(source_root) / 'images' / split
        source_labels = Path(source_root) / 'labels' / split
        target_images = Path(target_root) / 'images' / split
        target_labels = Path(target_root) / 'labels' / split

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        all_images = sorted(source_images.glob('*.jpg'))

        # 1. Videoları grupla
        videos = {}
        for img_path in all_images:
            name = img_path.stem
            videoname = name[:-5]  # Son 4 hane ve '_' hariç video adı
            if videoname not in videos:
                videos[videoname] = []
            videos[videoname].append(img_path)

        # 2. Sadece ilk 15 videoyu seç (Alfabetik sıraya göre)
        sorted_video_names = sorted(list(videos.keys()))
        selected_video_names = sorted_video_names[:num_videos]

        current_count = 0
        limit = max_limits[split]

        print(f"{split} işleniyor... Seçilen {len(selected_video_names)} video taranacak. (Limit: {limit})")

        for v_name in selected_video_names:
            if current_count >= limit:
                break

            # O videoya ait kareleri frame id'ye göre sırala
            frames = sorted(videos[v_name], key=lambda x: int(x.stem[-4:]))

            for img_path in frames:
                if current_count >= limit:
                    break

                frameid = int(img_path.stem[-4:])

                # Step kontrolü
                if frameid % step == 0:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Resize
                        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(target_images / img_path.name), resized_img)

                        # Etiket kopyala
                        label_path = source_labels / f"{img_path.stem}.txt"
                        if label_path.exists():
                            shutil.copy(label_path, target_labels / label_path.name)

                        current_count += 1

        print(f"--- {split} bitti. {current_count} kare hazır.")

    print(f"İşlem tamam! İlk {num_videos} video kullanılarak {target_root} oluşturuldu.")

import random
# Çalıştır
#sample_dataset_by_video(num_videos=15)
def sample_dataset_range(source_root='dataset_rgb', target_root='sampledataset10',
                         step=5, start_idx=0, end_idx=90):
    """
    Video sınırları aşılırsa rastgele önceki videoları seçer.
    Val Sınırı: 67 video | Test Sınırı: 90 video
    """
    splits = ['train', 'val', 'test']
    max_limits = {'train': float('inf'), 'val': 500, 'test': 500}
    TARGET_SIZE = (640, 512)

    for split in splits:
        source_images = Path(source_root) / 'images' / split
        source_labels = Path(source_root) / 'labels' / split
        target_images = Path(target_root) / 'images' / split
        target_labels = Path(target_root) / 'labels' / split

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        # 1. Klasördeki tüm resimleri bul ve grupla
        all_images = sorted(source_images.glob('*.jpg'))
        videos = {}
        for img_path in all_images:
            v_name = img_path.stem[:-5]
            if v_name not in videos: videos[v_name] = []
            videos[v_name].append(img_path)

        sorted_v_names = sorted(list(videos.keys()))
        total_v_in_folder = len(sorted_v_names)

        # 2. Seçim Mantığı
        # Eğer start_idx mevcut video sayısından büyükse veya aralık çok darsa rastgele seç
        if start_idx >= total_v_in_folder:
            print(f"--- {split.upper()}: start_idx ({start_idx}) sınırı aşıyor. Rastgele seçim yapılıyor...")
            # İstenen miktar kadar (end_idx - start_idx) veya mevcut kadar rastgele seç
            count_to_pick = min(end_idx - start_idx, total_v_in_folder)
            if count_to_pick <= 0: count_to_pick = 10  # Default küçük bir havuz
            selected_video_names = random.sample(sorted_v_names, count_to_pick)
        else:
            # Normal aralık seçimi (Sınırı aşmayacak şekilde)
            actual_end = min(end_idx, total_v_in_folder)
            selected_video_names = sorted_v_names[start_idx:actual_end]

        # 3. Kopyalama İşlemi
        current_count = 0
        print(f"{split.upper()} işleniyor... Seçilen Video Sayısı: {len(selected_video_names)}")

        for v_name in selected_video_names:
            frames = sorted(videos[v_name], key=lambda x: int(x.stem[-4:]))
            for img_path in frames:
                if current_count >= max_limits[split]: break

                frameid = int(img_path.stem[-4:])
                if frameid % step == 0:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(target_images / img_path.name), resized)

                        label_path = source_labels / f"{img_path.stem}.txt"
                        if label_path.exists():
                            shutil.copy(label_path, target_labels / label_path.name)
                        current_count += 1

        print(f"--- {split} bitti. Toplam {current_count} kare.")

    print(f"\nİşlem tamam! '{target_root}' hazır.")

# Kullanım:
#sample_dataset_range(start_idx=144, end_idx=160, step=5)


class DummyTrack:
    def __init__(self, history, bbox):
        self.history = history  # [(x, y, w, h), ...]
        self.bbox = bbox  # [x, y, w, h]

import numpy as np
def prediction_function(track, max_history=5):
    if len(track.history) == 0: return track.bbox
    if len(track.history) == 1: return track.history[-1]

    N = min(max_history, len(track.history))
    xs, ys, ws, hs = [], [], [], []

    weights = np.linspace(1, N, N)
    weights = weights / weights.sum()

    # History'nin son N elemanını al
    for i in range(-N, 0):
        x, y, w, h = track.history[i]
        xs.append(x);
        ys.append(y);
        ws.append(w);
        hs.append(h)

    def weighted_linreg(t, values, weights):
        X = np.vstack([t, np.ones_like(t)]).T
        W = np.diag(weights)
        # Çözüm: (X^T * W * X)^-1 * X^T * W * Y
        theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ values)
        return theta

    t = np.arange(N)
    a_x, b_x = weighted_linreg(t, np.array(xs), weights)
    a_y, b_y = weighted_linreg(t, np.array(ys), weights)

    t_next = N
    return [a_x * t_next + b_x, a_y * t_next + b_y, ws[-1], hs[-1]]


def get_previous_labels_matrix_with_none(image_path, dataset_labels_dir, num_prev=5):
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    try:
        frameid = int(name[-4:])
        videoname = name[:-5]
    except:
        return [None] * num_prev

    labels_list = []
    # Not: Eğitim sırasında sadece train/val içinde ararız
    for i in range(1, num_prev + 1):
        prev_frameid = frameid - i
        target_filename = f"{videoname}_{prev_frameid:04d}.txt"
        found_data = None

        # Etiketi ara
        for split in ["test"]:
            full_path = os.path.join(dataset_labels_dir, split, target_filename)
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    lines = f.readlines()
                    if lines: found_data = [float(p) for p in lines[0].split()]
                break
        labels_list.append(found_data)

    labels_list.reverse()  # ÖNEMLİ: Zamanı [eski -> yeni] sırasına sokar
    return labels_list


def apply_highlight(frame, pred_bbox, alpha=0.5, expand_ratio=3.0):
    """
    pred_bbox: [cx, cy, w, h]
    expand_ratio: Kutunun kaç kat büyüyeceği (3.0 = 3 kat genişlik ve yükseklik)
    """
    h, w, _ = frame.shape
    # Arka planı biraz daha görünür yapalım (0.3 çok karanlıktı, 0.5 daha ideal)
    highlighted = (frame * alpha).astype(np.uint8)

    if pred_bbox is None:
        return highlighted

    cx, cy, bw, bh = pred_bbox

    # Kutuyu genişletiyoruz
    new_w = bw * expand_ratio
    new_h = bh * expand_ratio

    # Normalize -> Pixel (Genişletilmiş koordinatlar)
    ix1 = int((cx - new_w / 2) * w)
    iy1 = int((cy - new_h / 2) * h)
    ix2 = int((cx + new_w / 2) * w)
    iy2 = int((cy + new_h / 2) * h)

    # Görüntü sınırlarını aşmamak için kırpma
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(w, ix2), min(h, iy2)

    # Genişletilmiş alanı orijinal parlaklığına getir
    highlighted[iy1:iy2, ix1:ix2] = frame[iy1:iy2, ix1:ix2]

    return highlighted



def prepare_highlighted_dataset(input_img_dir, input_lbl_dir, output_img_dir):
    # Bu fonksiyonu bir kez çalıştırarak tüm resimleri highlight edilmiş halleriyle kaydet
    for img_name in os.listdir(input_img_dir):
        img_path = os.path.join(input_img_dir, img_name)

        # 1. Geçmişi bul
        history = get_previous_labels_matrix_with_none(img_path, input_lbl_dir)
        valid_history = []
        if history is not None:
            for h in history:
                # h'nin bir liste olduğunu ve içinde veri olduğunu doğrula
                if isinstance(h, list) and len(h) >= 5:
                    # Sınıf ID'sini (h[0]) atla, sadece x,y,w,h al
                    valid_history.append(h[1:5])

        # 2. Tahmin yap (Eğer geçmiş varsa)
        frame = cv2.imread(img_path)
        # 640x512 resize işlemini burada yapmayı unutma!
        frame = cv2.resize(frame, (640, 512))

        if valid_history:
            dummy = DummyTrack(history=valid_history, bbox=valid_history[-1])
            pred = prediction_function(dummy)
            final_img = apply_highlight(frame, pred)
        else:
            final_img = frame  # Geçmiş yoksa olduğu gibi bırak veya tamamen karart

        cv2.imwrite(os.path.join(output_img_dir, img_name), final_img)



# ---------------------------------------------------------------------------
#os.makedirs("last_test3")
#prepare_highlighted_dataset("dataset_rgb/images/test","dataset_rgb/labels", "last_test3")

import os

def parse_name(name):
    # 20190925_134301_1_5_0203
    parts = name.split("_")
    date_time = parts[0] + "_" + parts[1]
    cam = int(parts[2])
    seq = int(parts[3])
    frame = int(parts[4])
    return date_time, cam, seq, frame

ranges = [
("20190925_111757_1_1_0777","20190925_111757_1_1_0999"),
("20190925_111757_1_7_0237","20190925_111757_1_8_0999"),
("20190925_111757_1_9_0759","20190925_111757_1_9_0999"),
("20190925_124612_1_1_0370","20190925_124612_1_1_0999"),
("20190925_124612_1_2_0144","20190925_124612_1_2_0999"),
("20190925_124612_1_3_0268","20190925_124612_1_3_0999"),
("20190925_124612_1_7_0756","20190925_124612_1_7_0999"),
("20190925_134301_1_2_0736","20190925_134301_1_2_0999"),
("20190925_134301_1_3_0171","20190925_134301_1_3_0999"),
("20190925_134301_1_4_0626","20190925_134301_1_4_0999"),
("20190925_134301_1_5_0203","20190925_134301_1_7_0999"),
("20190925_134301_1_8_0368","20190925_134301_1_9_0618"),
("20190925_193610_1_7_0322","20190925_193610_1_7_0938"),
("20190926_095902_1_1_0731","20190926_095902_1_1_0741"),
("20190926_095902_1_1_0798","20190926_095902_1_1_0999"),
("20190926_095902_1_3_0344","20190926_095902_1_3_0999"),
("20190926_095902_1_4_0865","20190926_095902_1_4_0999"),
("20190926_095902_1_8_0426","20190926_095902_1_8_0999"),
("20190926_102042_1_2_0491","20190926_102042_1_3_0999"),
("20190926_102042_1_4_0105","20190926_102042_1_4_0999"),
("20190926_102042_1_5_0251","20190926_102042_1_6_0999"),
("20190926_102042_1_8_0907","20190926_102042_1_8_0999"),
("20190926_111509_1_3_0734","20190926_111509_1_3_0999"),
("20190926_111509_1_8_0094","20190926_111509_1_8_0999"),
("20190926_134054_1_1_0075","20190926_134054_1_1_0999"),
("20190926_134054_1_7_0156","20190926_134054_1_7_0999"),
]
def delete_frame_ranges(
    image_root="dataset_rgb/images/test",
    ranges=None
):
    if ranges is None:
        raise ValueError("ranges listesi boş olamaz")

    deleted = 0

    for fname in os.listdir(image_root):
        if not fname.endswith(".jpg"):
            continue

        base = fname.replace(".jpg", "")
        try:
            date_time, cam, seq, frame = parse_name(base)
        except:
            continue

        for start, end in ranges:
            s_dt, s_cam, s_seq, s_fr = parse_name(start)
            e_dt, e_cam, e_seq, e_fr = parse_name(end)

            # date & cam eşleşmeli
            if date_time != s_dt or cam != s_cam:
                continue

            # seq aralığında mı?
            if not (s_seq <= seq <= e_seq):
                continue

            # frame koşulu (seq'e göre)
            if seq == s_seq and frame < s_fr:
                continue
            if seq == e_seq and frame > e_fr:
                continue

            # aradaki seq'ler full silinir
            path = os.path.join(image_root, fname)
            os.remove(path)
            deleted += 1
            break

    print(f"🗑️ Silinen jpg sayısı: {deleted}")
#delete_frame_ranges(image_root="dataset_rgb/images/test",ranges=ranges)