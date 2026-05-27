import numpy as np
from scipy.spatial.distance import cosine
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

    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - interArea
    return interArea / union if union > 0 else 0.0

def safe_cosine(a, b):
    if a is None or b is None:
        return 1.0

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-6 or nb < 1e-6:
        return 1.0  # max distance

    return cosine(a, b)

def vectorized_iou(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    b1_x1, b1_y1 = boxes1[:, 0], boxes1[:, 1]
    b1_x2, b1_y2 = boxes1[:, 0] + boxes1[:, 2], boxes1[:, 1] + boxes1[:, 3]
    b2_x1, b2_y1 = boxes2[:, 0], boxes2[:, 1]
    b2_x2, b2_y2 = boxes2[:, 0] + boxes2[:, 2], boxes2[:, 1] + boxes2[:, 3]
    xA = np.maximum(b1_x1[:, np.newaxis], b2_x1[np.newaxis, :])
    yA = np.maximum(b1_y1[:, np.newaxis], b2_y1[np.newaxis, :])
    xB = np.minimum(b1_x2[:, np.newaxis], b2_x2[np.newaxis, :])
    yB = np.minimum(b1_y2[:, np.newaxis], b2_y2[np.newaxis, :])
    interW = np.maximum(0, xB - xA)
    interH = np.maximum(0, yB - yA)
    interArea = interW * interH
    area1 = (boxes1[:, 2] * boxes1[:, 3])[:, np.newaxis]
    area2 = (boxes2[:, 2] * boxes2[:, 3])[np.newaxis, :]
    unionArea = area1 + area2 - interArea
    return interArea / np.maximum(unionArea, 1e-6)

def vectorized_cosine(embs1, embs2):
    if len(embs1) == 0 or len(embs2) == 0:
        return np.ones((len(embs1), len(embs2)))
    # If all embs in a list are None, return all ones
    if all(e is None for e in embs1) or all(e is None for e in embs2):
        return np.ones((len(embs1), len(embs2)))
    
    # Get a dummy shape from the first valid embedding
    valid_shape = next((e.shape for e in embs1 + embs2 if e is not None), (1,))
    
    e1 = np.array([e if e is not None else np.zeros(valid_shape) for e in embs1])
    e2 = np.array([e if e is not None else np.zeros(valid_shape) for e in embs2])
    
    # Squeeze out extra dimensions if necessary
    e1 = np.reshape(e1, (e1.shape[0], -1))
    e2 = np.reshape(e2, (e2.shape[0], -1))
    
    norm1 = np.linalg.norm(e1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(e2, axis=1, keepdims=True)
    norm1[norm1 < 1e-6] = 1.0
    norm2[norm2 < 1e-6] = 1.0
    
    return 1.0 - np.dot(e1 / norm1, (e2 / norm2).T)

def prediction_function(track, max_history=5):
    """ Weighted Regression: Son N konuma bakarak t+1 tahmini yapar """
    hist_len = len(track.history)
    if hist_len < 2:
        return track.bbox

    N = min(max_history, hist_len)
    hist_arr = np.array(track.history[-N:])
    xs, ys, ws, hs = hist_arr[:, 0], hist_arr[:, 1], hist_arr[:, 2], hist_arr[:, 3]

    weights = np.linspace(1, N, N)
    weights /= weights.sum()
    t = np.arange(N)

    def fast_weighted_linreg(t, values, w):
        mean_t = np.sum(w * t)
        mean_y = np.sum(w * values)
        var_t = np.sum(w * (t - mean_t)**2)
        if var_t == 0:
            return 0.0, mean_y
        cov_ty = np.sum(w * (t - mean_t) * (values - mean_y))
        a = cov_ty / var_t
        return a, mean_y - a * mean_t

    a_x, b_x = fast_weighted_linreg(t, xs, weights)
    a_y, b_y = fast_weighted_linreg(t, ys, weights)

    return [a_x * N + b_x, a_y * N + b_y, ws[-1], hs[-1]]



def update_embedding(old_emb, new_emb, momentum=0.9):
    if old_emb is None:
        return new_emb

    emb = momentum * old_emb + (1 - momentum) * new_emb
    emb /= np.linalg.norm(emb) + 1e-6
    return emb

class Track:
    def __init__(self, track_id, bbox, embedding, last_seen_frame, fps=30):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.last_seen_frame = last_seen_frame
        self.missing_frames = 0
        self.history = []
        self.sim = 0
        self.fps = fps
    def is_stationary(self, frame_window=None, pixel_threshold=5.0):
        """Son 'frame_window' kadar karede kutu merkezinin ne kadar hareket ettiğine bakar."""
        if frame_window is None:
            frame_window = int(2 * self.fps)
            
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

    """
    Tracker class will be used for ID assignment
    """
    def __init__(self, similarity_threshold=0.7, max_missing=10, fps=30):
        self.tracks = []
        self.next_id = 0
        self.similarity_threshold = similarity_threshold
        self.max_missing = max_missing
        self.fps = fps

    def remove_duplicate_tracks(self):
        """
        If tracks are very similar, delete these tracks, leave only one
        """
        if len(self.tracks) < 2:
            return

        bboxes = np.array([t.bbox for t in self.tracks])
        iou_matrix = vectorized_iou(bboxes, bboxes)
        
        tracks_to_remove = []
        for i in range(len(self.tracks)):
            for j in range(i + 1, len(self.tracks)):
                if iou_matrix[i, j] > 0.80:
                    if self.tracks[i].track_id > self.tracks[j].track_id:
                        tracks_to_remove.append(self.tracks[i])
                    else:
                        tracks_to_remove.append(self.tracks[j])

        # Listeden temizle
        self.tracks = [t for t in self.tracks if t not in tracks_to_remove]

    def update(self, detections, embeddings, frame_idx, frame_width=640, frame_height=512, bg_dx=0.0, bg_dy=0.0):

        assigned_tracks = set()
        assigned_detections = set()

        # -------------------------------
        # INIT
        # -------------------------------
        if len(self.tracks) == 0:
            for d_idx, det in enumerate(detections):
                emb = embeddings[d_idx]

                t = Track(self.next_id, det, emb, frame_idx, fps=self.fps)
                t.history.append(det)
                self.tracks.append(t)
                self.next_id += 1
            return self.tracks

        # -------------------------------
        # NO DETECTION → PURE PREDICTION
        # -------------------------------
        if len(detections) == 0:
            for track in self.tracks:
                track.missing_frames += 1
                px, py, pw, ph = prediction_function(track)

                px = max(0, min(px, frame_width - pw))
                py = max(0, min(py, frame_height - ph))

                track.bbox = [px, py, pw, ph]
                track.history.append(track.bbox)
            return self.tracks

        alpha = 0.5
        iou_threshold = 0.00

        # -------------------------------
        # VECTORIZED COST MATRIX
        # -------------------------------
        pred_bboxes = np.array([prediction_function(t) for t in self.tracks])
        det_bboxes = np.array(detections)
        
        iou_matrix = vectorized_iou(pred_bboxes, det_bboxes)
        iou_cost_matrix = 1.0 - iou_matrix
        
        pred_areas = (pred_bboxes[:, 2] * pred_bboxes[:, 3])[:, np.newaxis]
        det_areas = (det_bboxes[:, 2] * det_bboxes[:, 3])[np.newaxis, :]
        area_growth_mask = (pred_areas > 0) & (det_areas > 3.0 * pred_areas)
        
        track_embs = [t.embedding for t in self.tracks]
        cosine_dist_matrix = vectorized_cosine(track_embs, embeddings)
        
        cost_matrix = alpha * cosine_dist_matrix + (1 - alpha) * iou_cost_matrix
        
        emb_missing_mask = np.array([e is None for e in track_embs])[:, np.newaxis] | np.array([e is None for e in embeddings])[np.newaxis, :]
        cost_matrix = np.where(emb_missing_mask, iou_cost_matrix, cost_matrix)
        
        invalid_mask = area_growth_mask | (iou_matrix < iou_threshold)
        cost_matrix[invalid_mask] = 1e6

        cost_matrix = np.nan_to_num(
            cost_matrix,
            nan=1e6,
            posinf=1e6,
            neginf=1e6
        )
        # -------------------------------
        # HUNGARIAN MATCHING
        # -------------------------------
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        for t_idx, d_idx in zip(row_ind, col_ind):
            cost = cost_matrix[t_idx, d_idx]
            matched_pairs.append((t_idx, d_idx, cost))
        matched_pairs.sort(key=lambda x: x[2])

        for t_idx, d_idx, current_cost in matched_pairs:


            if current_cost < (1 - self.similarity_threshold):


                track = self.tracks[t_idx]
                det = detections[d_idx]

                if len(track.history) >= 3:
                    pred_bbox = prediction_function(track)
                    pred_cx = pred_bbox[0] + pred_bbox[2] / 2
                    pred_cy = pred_bbox[1] + pred_bbox[3] / 2
                    det_cx = det[0] + det[2] / 2
                    det_cy = det[1] + det[3] / 2

                    error_dist = np.sqrt((pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2)

                    # Eğer hata 25 pikselden büyükse, drone ani manevra yapmıştır.
                    if error_dist > 25.0:
                        # Geçmişi silip sadece son konumu bırakıyoruz.
                        track.history = track.history[-1:]

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

                track.embedding = update_embedding(
                    track.embedding,
                    embeddings[d_idx]
                )
                track.last_seen_frame = frame_idx
                track.missing_frames = 0
                track.sim = 1 - current_cost
                track.history.append(det)

                assigned_tracks.add(t_idx)
                assigned_detections.add(d_idx)
            else:
                pass  # No match for track id: {t_idx}
        # -------------------------------
        # UNMATCHED TRACKS → PREDICT
        # -------------------------------
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in assigned_tracks:
                track.missing_frames += 1
                track.sim = 0
                px, py, pw, ph = prediction_function(track)

                px = max(0, min(px, frame_width - pw))
                py = max(0, min(py, frame_height - ph))

                track.bbox = [px, py, pw, ph]
                track.history.append(track.bbox)

        # -------------------------------
        # NEW TRACKS
        # -------------------------------
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_detections:
                emb = embeddings[d_idx]
                if emb is not None and np.linalg.norm(emb) < 1e-6:
                    emb = None

                t = Track(self.next_id, det, emb, frame_idx, fps=self.fps)
                t.history.append(det)
                self.tracks.append(t)
                self.next_id += 1

        # -------------------------------
        # CLEANUP
        # -------------------------------
        self.tracks = [
            t for t in self.tracks
            if t.missing_frames <= self.max_missing
        ]
        self.remove_duplicate_tracks()
        return self.tracks

    def reset(self):
        """

        Tüm tracking state'ini sıfırlar.
        """
        self.tracks.clear()
        self.next_id = 0

    def get_best_track(self):
        active_tracks = self.tracks

        if not active_tracks:
            return None

        best_t = None
        max_score = -1.0

        for t in active_tracks:
            history_window = int(2 * self.fps)
            history_score = min(len(t.history), history_window) / float(history_window)

            current_sim = t.sim

            total_score = (history_score * 0.5) + (current_sim * 0.5)

            if total_score > max_score:
                max_score = total_score
                best_t = t

        return best_t