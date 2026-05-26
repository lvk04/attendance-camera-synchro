import logging
import cv2

# Import torch FIRST so its DLL definitions take priority in memory
import torch
import torchreid
from ultralytics import YOLO
import faiss  
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonTracker:
    def __init__(
        self,
        model_path="yolo11s.pt",
        reid_weights="osnet_x1_0_msmt17.pth",
        tracker_config="custom_tracker.yaml",
        sim_threshold=0.65,
        occlusion_iou_threshold=0.35,
        stability_frames=3,
        reid_top_k=3,
        track_ttl=30,
        ema_alpha=0.1,
    ):
        self.model = YOLO(model_path, task="detect")
        self.tracker_config = tracker_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== LOAD OSNET =====
        self.reid_model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=False
        )
        torchreid.utils.load_pretrained_weights(self.reid_model, reid_weights)
        self.reid_model.to(self.device)
        self.reid_model.eval()
        self.reid_model = self.reid_model.float()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # ===== FAISS & REID LOGIC =====
        self.dim = 512
        self.index = faiss.IndexFlatIP(self.dim)
        # id_map[i] -> global_id for the i-th FAISS vector
        self.id_map: list[int] = []
        # Stores the raw embedding for each FAISS index position (for EMA updates)
        self.embeddings: list[np.ndarray] = []
        self.next_global_id = 0

        # --- Tunable thresholds (all now constructor params) ---
        self.SIM_THRESHOLD = sim_threshold
        self.OCCLUSION_IOU_THRESHOLD = occlusion_iou_threshold
        self.STABILITY_FRAMES = stability_frames
        self.REID_TOP_K = reid_top_k
        self.EMA_ALPHA = ema_alpha

        self.trackid_to_global: dict[int, int] = {}

        # FIX 4: track_history also records the last frame a track was seen
        # so stale entries can be evicted after track_ttl missed frames.
        self.track_history: dict[int, int] = {}       # track_id -> stable frame count
        self.track_last_seen: dict[int, int] = {}     # track_id -> frame index
        self.TRACK_TTL = track_ttl

        # ROI zone — person must enter this to trigger identity link
        self.ROI = (400, 50, 560, 350)  # (x1, y1, x2, y2)

        # Target tracking state — set by ZMQ signal, linked on ROI entry
        self.target_name: str | None = None
        self.target_global_id: int | None = None

        self.frame_idx = 0

        self.COLOR_PALETTE = np.array([
            (255, 50, 50), (50, 255, 50), (50, 50, 255),
            (255, 255, 50), (50, 255, 255), (255, 50, 255),
            (255, 150, 50), (150, 50, 255), (50, 150, 255), (150, 255, 50),
        ], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_roi(self, point: tuple[int, int]) -> bool:
        x, y = point
        rx1, ry1, rx2, ry2 = self.ROI
        return rx1 <= x <= rx2 and ry1 <= y <= ry2

    def get_iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def get_embedding(self, frame, box) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            # FIX 10: warn instead of silently returning None
            logger.warning("Empty crop for box %s — skipping embedding.", box)
            return None

        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.reid_model(img)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        feat = feat.view(feat.size(0), -1)
        feat = feat / feat.norm(p=2, dim=1, keepdim=True)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def _evict_stale_tracks(self):
        """FIX 4: Remove track IDs that haven't been seen for TRACK_TTL frames."""
        stale = [
            t_id
            for t_id, last in self.track_last_seen.items()
            if self.frame_idx - last > self.TRACK_TTL
        ]
        for t_id in stale:
            self.trackid_to_global.pop(t_id, None)
            self.track_history.pop(t_id, None)
            self.track_last_seen.pop(t_id, None)

    def _update_faiss_embedding(self, faiss_idx: int, new_emb: np.ndarray):
        """FIX 8: EMA-update the stored embedding and rebuild that FAISS slot."""
        old_emb = self.embeddings[faiss_idx]
        updated = (1 - self.EMA_ALPHA) * old_emb + self.EMA_ALPHA * new_emb
        updated /= np.linalg.norm(updated) + 1e-6
        self.embeddings[faiss_idx] = updated

        # faiss.IndexFlatIP has no in-place update; rebuild from stored embeddings.
        self.index.reset()
        all_embs = np.stack(self.embeddings, axis=0)
        self.index.add(all_embs)

    def _match_or_register(self, emb: np.ndarray) -> tuple[int, int | None]:
        """
        FIX 6: Search top-k and use majority-vote among matches above threshold.
        Returns (global_id, faiss_index_of_match | None).
        faiss_index_of_match is None when a brand-new ID is created.
        """
        if self.index.ntotal == 0:
            gid = self.next_global_id
            self.next_global_id += 1
            self.id_map.append(gid)
            self.embeddings.append(emb.copy())
            self.index.add(emb.reshape(1, -1))
            return gid, None

        k = min(self.REID_TOP_K, self.index.ntotal)
        D, I = self.index.search(emb.reshape(1, -1), k)

        # Collect all candidates above the similarity threshold
        candidates: dict[int, list[float]] = {}  # global_id -> list of similarities
        candidate_faiss_idx: dict[int, int] = {}  # global_id -> best faiss slot
        for sim, faiss_idx in zip(D[0], I[0]):
            if sim > self.SIM_THRESHOLD:
                gid = self.id_map[int(faiss_idx)]
                candidates.setdefault(gid, []).append(sim)
                # Keep track of the faiss slot with the best similarity per gid
                if gid not in candidate_faiss_idx or sim > max(
                    candidates[gid][:-1], default=-1
                ):
                    candidate_faiss_idx[gid] = int(faiss_idx)

        if candidates:
            # Majority vote: pick the gid with the most votes; break ties by avg sim
            best_gid = max(
                candidates,
                key=lambda g: (len(candidates[g]), sum(candidates[g]) / len(candidates[g])),
            )
            return best_gid, candidate_faiss_idx[best_gid]

        # No match — register new identity
        gid = self.next_global_id
        self.next_global_id += 1
        self.id_map.append(gid)
        self.embeddings.append(emb.copy())
        self.index.add(emb.reshape(1, -1))
        return gid, None

    # ------------------------------------------------------------------
    # Main tracking loop
    # ------------------------------------------------------------------

    def track(self, source_url, imgsz=480, conf=0.3):
        results = self.model.track(
            source=source_url,
            conf=conf,
            imgsz=imgsz,
            stream=True,
            tracker=self.tracker_config,
            persist=True,
            classes=[0],
        )

        for r in results:
            self.frame_idx += 1
            frame = r.orig_img
            current_detections = []

            # FIX 4: clean up stale track state
            self._evict_stale_tracks()

            # FIX 3: build a confidence-sorted list so the most confident box
            # wins when two detections collide on the same global ID.
            assigned_in_frame: dict[int, float] = {}  # global_id -> winning score

            if r.boxes is not None and r.boxes.id is not None:
                data = r.boxes.data.cpu().numpy()

                # FIX 1: use explicit indexing instead of fragile unpacking
                all_boxes = data[:, :4]

                # FIX 3: sort by confidence descending so higher-conf box wins
                order = np.argsort(-data[:, 5])
                data = data[order]

                for row in data:
                    # FIX 1: safe explicit indexing
                    x1, y1, x2, y2 = row[:4]
                    track_id = int(row[4])
                    score = float(row[5])
                    box = [int(x1), int(y1), int(x2), int(y2)]

                    self.track_last_seen[track_id] = self.frame_idx
                    global_id = None

                    if track_id in self.trackid_to_global:
                        global_id = self.trackid_to_global[track_id]

                        # FIX 8: opportunistically update the gallery embedding
                        emb = self.get_embedding(frame, box)
                        if emb is not None:
                            emb /= np.linalg.norm(emb) + 1e-6
                            # Find the faiss slot for this global_id
                            faiss_idx = next(
                                (i for i, g in enumerate(self.id_map) if g == global_id),
                                None,
                            )
                            if faiss_idx is not None:
                                self._update_faiss_embedding(faiss_idx, emb)
                    else:
                        # Check occlusion with the configurable threshold (FIX 5)
                        is_occluded = any(
                            not np.array_equal(box, other.astype(int))
                            and self.get_iou(box, other) > self.OCCLUSION_IOU_THRESHOLD
                            for other in all_boxes
                        )

                        if not is_occluded:
                            self.track_history[track_id] = (
                                self.track_history.get(track_id, 0) + 1
                            )

                            if self.track_history[track_id] >= self.STABILITY_FRAMES:
                                emb = self.get_embedding(frame, box)
                                if emb is not None:
                                    emb /= np.linalg.norm(emb) + 1e-6
                                    # FIX 6: majority-vote k-NN match
                                    best_gid, faiss_idx = self._match_or_register(emb)
                                    # FIX 8: EMA update if this was an existing identity
                                    if faiss_idx is not None:
                                        self._update_faiss_embedding(faiss_idx, emb)
                                    self.trackid_to_global[track_id] = best_gid
                                    global_id = best_gid

                    # FIX 3: highest-confidence box wins per global_id
                    if global_id is not None:
                        if global_id in assigned_in_frame:
                            if score <= assigned_in_frame[global_id]:
                                continue  # a better box already claimed this ID
                        assigned_in_frame[global_id] = score

                        base_point = (int((x1 + x2) / 2), int(y2))
                        current_detections.append({
                            "id": track_id,
                            "global_id": global_id,
                            "bbox": box,
                            "base_point": base_point,
                        })

                        # ═══════ ROI LINKING ═══════
                        if (self.target_name is not None
                                and self.target_global_id is None
                                and self._in_roi(base_point)):
                            self.target_global_id = global_id
                            logger.info("TRACK LINKED: '%s' → global_id=%d at ROI zone",
                                        self.target_name, global_id)
                        # ═══════════════════════════

                        color = self.COLOR_PALETTE[global_id % len(self.COLOR_PALETTE)].tolist()
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                        cv2.putText(
                            frame,
                            f"ID:{global_id}",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                        # ═══════ TARGET HIGHLIGHT ═══════
                        if self.target_name and global_id == self.target_global_id:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)
                            label = f"TRACKING: {self.target_name}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3)
                            cv2.rectangle(frame, (box[0], box[1]-th-10), (box[0]+tw, box[1]), (0, 255, 0), -1)
                            cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        # ════════════════════════════════
                    else:
                        cv2.rectangle(
                            frame, (box[0], box[1]), (box[2], box[3]), (100, 100, 100), 1
                        )

            # Draw ROI rectangle
            rx1, ry1, rx2, ry2 = self.ROI
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(frame, "TRACK ZONE", (rx1, ry1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            yield frame, current_detections