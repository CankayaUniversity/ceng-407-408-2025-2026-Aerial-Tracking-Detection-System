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

def prediction_function(track, max_history=4):
    """
    track.history: [(x, y, w, h), ...]
    track.bbox: [x, y, w, h]
    """

    if len(track.history) == 0:
        return track.bbox

    if len(track.history) == 1:
        return track.history[-1]

    N = min(max_history, len(track.history))

    xs, ys, ws, hs = [], [], [], []

    weights = np.linspace(1, N, N)
    weights = weights / weights.sum()

    for i in range(-N, 0):
        x, y, w, h = track.history[i]
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

    xs = np.array(xs)
    ys = np.array(ys)

    def weighted_linreg(t, values, weights):
        X = np.vstack([t, np.ones_like(t)]).T
        W = np.diag(weights)
        theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ values)
        return theta  # [a, b]

    t = np.arange(N)

    a_x, b_x = weighted_linreg(t, xs, weights)
    a_y, b_y = weighted_linreg(t, ys, weights)

    t_next = N
    pred_x = a_x * t_next + b_x
    pred_y = a_y * t_next + b_y

    w = ws[-1]
    h = hs[-1]

    return [pred_x, pred_y, w, h]



def update_embedding(old_emb, new_emb, momentum=0.9):
    if old_emb is None:
        return new_emb

    emb = momentum * old_emb + (1 - momentum) * new_emb
    emb /= np.linalg.norm(emb) + 1e-6
    return emb

class Track:

    def __init__(self, track_id, bbox, embedding, last_seen_frame):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.last_seen_frame = last_seen_frame
        self.missing_frames = 0
        self.history = []
        self.sim = 0

class Tracker:

    """
    Tracker class will be used for ID assignment
    """
    def __init__(self, similarity_threshold=0.7, max_missing=10):
        self.tracks = []
        self.next_id = 0
        self.similarity_threshold = similarity_threshold
        self.max_missing = max_missing

    def remove_duplicate_tracks(self):
        """
        If tracks are very similar, delete these tracks, leave only one
        """
        if len(self.tracks) < 2:
            return

        tracks_to_remove = []
        for i, track_a in enumerate(self.tracks):
            for j, track_b in enumerate(self.tracks):
                if i >= j: continue


                track_iou = iou(track_a.bbox, track_b.bbox)

                # Delete very close tracks
                if track_iou > 0.80:

                    if track_a.track_id > track_b.track_id:
                        tracks_to_remove.append(track_a)
                    else:
                        tracks_to_remove.append(track_b)

        # Listeden temizle
        self.tracks = [t for t in self.tracks if t not in tracks_to_remove]

    def update(self, detections, embeddings, frame_idx, frame_width=640, frame_height=512):

        assigned_tracks = set()
        assigned_detections = set()

        # -------------------------------
        # INIT
        # -------------------------------
        if len(self.tracks) == 0:
            for d_idx, det in enumerate(detections):
                emb = embeddings[d_idx]

                t = Track(self.next_id, det, emb, frame_idx)
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

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        # -------------------------------
        # COST MATRIX (PREDICTION-BASED)
        # -------------------------------
        for t_idx, track in enumerate(self.tracks):

            # 🔥 prediction sadece 1 kere
            pred_bbox = prediction_function(track)

            for d_idx, det_bbox in enumerate(detections):

                iou_score = iou(pred_bbox, det_bbox)
                iou_cost = 1.0 - iou_score

                if iou_score < iou_threshold:
                    cost_matrix[t_idx, d_idx] = 1e6
                    continue

                # 🔑 Appearance var mı
                if track.embedding is None or embeddings[d_idx] is None:
                    # SADECE IoU
                    cost = iou_cost

                else:
                    appearance_cost = safe_cosine(track.embedding, embeddings[d_idx])
                    cost = alpha * appearance_cost + (1 - alpha) * iou_cost

                cost_matrix[t_idx, d_idx] = cost

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
                print(f"Eşleşmedi id: {t_idx} ")
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

                t = Track(self.next_id, det, emb, frame_idx)
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
            history_score = min(len(t.history), 50) / 50.0

            current_sim = t.sim

            total_score = (history_score * 0.5) + (current_sim * 0.5)

            if total_score > max_score:
                max_score = total_score
                best_t = t

        return best_t