import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(boxA, boxB):
    if boxA[2] <= 0 or boxA[3] <= 0 or boxB[2] <= 0 or boxB[3] <= 0:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea
    return interArea / union if union > 0 else 0.0


def prediction_function(track, max_history=5):
    """ Weighted Regression: Son 5 konuma bakarak t+1 tahmini yapar """
    if len(track.history) < 2:
        return track.bbox

    N = min(max_history, len(track.history))
    xs, ys, ws, hs = [], [], [], []
    weights = np.linspace(1, N, N)
    weights = weights / weights.sum()

    for i in range(-N, 0):
        x, y, w, h = track.history[i]
        xs.append(x);
        ys.append(y);
        ws.append(w);
        hs.append(h)

    t = np.arange(N)

    def weighted_linreg(t, values, w_vec):
        X = np.vstack([t, np.ones_like(t)]).T
        W = np.diag(w_vec)
        theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ values)
        return theta  # [eğim, kayma]

    a_x, b_x = weighted_linreg(t, np.array(xs), weights)
    a_y, b_y = weighted_linreg(t, np.array(ys), weights)

    pred_x = a_x * N + b_x
    pred_y = a_y * N + b_y
    return [pred_x, pred_y, ws[-1], hs[-1]]


class Track:
    def __init__(self, track_id, bbox, last_seen_frame):
        self.track_id = track_id
        self.bbox = bbox
        self.last_seen_frame = last_seen_frame
        self.missing_frames = 0
        self.history = []
        self.sim = 0
    def is_stationary(self, frame_window=30, pixel_threshold=5.0):
        """Son 'frame_window' kadar karede kutu merkezinin ne kadar hareket ettiğine bakar."""
        if len(self.history) < frame_window:
            return False  # Henüz yeterince geçmiş yoksa hareketli varsay
        
        curr_box = self.history[-1]
        past_box = self.history[-frame_window]
        
        # Merkez noktalarını bul
        curr_cx = curr_box[0] + curr_box[2] / 2
        curr_cy = curr_box[1] + curr_box[3] / 2
        past_cx = past_box[0] + past_box[2] / 2
        past_cy = past_box[1] + past_box[3] / 2
        
        # Merkezler arası mesafe
        dist = np.sqrt((curr_cx - past_cx)**2 + (curr_cy - past_cy)**2)
        
        # Eğer mesafe eşikten küçükse (kutu aynı yerde sayıyorsa) True dön
        return dist < pixel_threshold

class Tracker:
    def __init__(self, iou_threshold=0.1, max_missing=10):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    def update(self, detections, frame_idx, frame_width=640, frame_height=512, bg_dx=0.0, bg_dy=0.0):
        assigned_tracks = set()
        assigned_detections = set()

        if len(self.tracks) == 0:
            for det in detections:
                t = Track(self.next_id, det, frame_idx)
                t.history.append(det)
                self.tracks.append(t)
                self.next_id += 1
            return self.tracks

        # Maliyet Matrisi (Sadece IoU üzerinden)
        cost_matrix = np.ones((len(self.tracks), len(detections)), dtype=np.float32)

        for t_idx, track in enumerate(self.tracks):
            # Mevcut frame tahmini
            pred_bbox = prediction_function(track)
            for d_idx, det_bbox in enumerate(detections):
                # Alan büyümesi kontrolü (Area Growth Check)
                pred_area = pred_bbox[2] * pred_bbox[3]
                det_area = det_bbox[2] * det_bbox[3]
                if pred_area > 0 and det_area > 3.0 * pred_area:
                    cost_matrix[t_idx, d_idx] = 1e6
                    continue
                    
                score = iou(pred_bbox, det_bbox)
                # IoU maliyeti = 1 - IoU
                cost_matrix[t_idx, d_idx] = 1.0 - score

        # Macar Algoritması (Hungarian Algorithm) ile Eşleştirme
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for t_idx, d_idx in zip(row_ind, col_ind):
            current_cost = cost_matrix[t_idx, d_idx]
            # Eğer IoU yeterince yüksekse (Maliyet eşikten düşükse)
            if current_cost < (1.0 - self.iou_threshold):
                track = self.tracks[t_idx]
                det = detections[d_idx]
                
                # Arka Plan Hareketi (Camera Panning) Kontrolü
                if len(track.history) > 0:
                    old_det = track.history[-1]
                    old_cx = old_det[0] + old_det[2] / 2
                    old_cy = old_det[1] + old_det[3] / 2
                    new_cx = det[0] + det[2] / 2
                    new_cy = det[1] + det[3] / 2
                    
                    track_dx = new_cx - old_cx
                    track_dy = new_cy - old_cy
                    
                    bg_mag = np.hypot(bg_dx, bg_dy)
                    track_mag = np.hypot(track_dx, track_dy)
                    
                    if bg_mag > 0.5: # Kamera belirgin hareket ediyorsa
                        dot_product = track_dx * bg_dx + track_dy * bg_dy
                        cos_sim = dot_product / (track_mag * bg_mag + 1e-6)
                        speed_ratio = track_mag / (bg_mag + 1e-6)
                        
                        if cos_sim > 0.95 and 0.90 < speed_ratio < 1.10:
                            # Arka plan objesi! Öldür.
                            track.missing_frames = self.max_missing + 1
                            continue
                            
                track.bbox = det
                track.last_seen_frame = frame_idx
                track.missing_frames = 0
                track.sim = 1 - current_cost
                track.history.append(det)
                assigned_tracks.add(t_idx)
                assigned_detections.add(d_idx)

        # Eşleşmeyenleri Tahminle Devam Ettir
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in assigned_tracks:
                track.missing_frames += 1
                track.sim = 0
                pred = prediction_function(track)
                track.bbox = [
                    max(0, min(pred[0], frame_width - pred[2])),
                    max(0, min(pred[1], frame_height - pred[3])),
                    pred[2], pred[3]
                ]
                track.history.append(track.bbox)

        # Yeni Çıkan Drone'ları Ekle
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_detections:
                t = Track(self.next_id, det, frame_idx)
                t.history.append(det)
                self.tracks.append(t)
                self.next_id += 1

        # Temizlik
        self.tracks = [t for t in self.tracks if t.missing_frames <= self.max_missing]
        self.remove_duplicate_tracks()
        return self.tracks

    def remove_duplicate_tracks(self):
        """Çok yakın (IoU > 0.80) track'leri temizler, sadece eskisini bırakır."""
        if len(self.tracks) < 2:
            return
        tracks_to_remove = []
        for i, track_a in enumerate(self.tracks):
            for j, track_b in enumerate(self.tracks):
                if i >= j:
                    continue
                track_iou = iou(track_a.bbox, track_b.bbox)
                if track_iou > 0.80:
                    if track_a.track_id > track_b.track_id:
                        tracks_to_remove.append(track_a)
                    else:
                        tracks_to_remove.append(track_b)
        self.tracks = [t for t in self.tracks if t not in tracks_to_remove]

    def reset(self):
        """Tüm tracking state'ini sıfırlar."""
        self.tracks.clear()
        self.next_id = 0
    
    def get_best_track(self):
        active_tracks = self.tracks

        if not active_tracks:
            return None

        best_t = None
        max_score = -1.0

        for t in active_tracks:
            history_score = min(len(t.history), 50) / 50.0

            current_sim = t.sim

            total_score = (history_score * 0.5) + (current_sim * 0.5)

            if total_score > max_score:
                max_score = total_score
                best_t = t

        return best_t