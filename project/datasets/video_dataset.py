"""Phase 3 — PyTorch Dataset that turns a manifest split CSV into (T,C,H,W)
frame tensors for frozen VideoMAE/CLIP feature extraction.

Frames are returned as un-normalized float32 in [0,1] -- per-encoder mean/std
normalization is applied in `models/encoders.py`, not here, so the same
decoded clip can feed both encoders without re-decoding.
"""
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("emomv.video_dataset")
_LOGGING_CONFIGURED = False


def configure_logging(log_path: Optional[Path] = None, level=logging.WARNING):
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logger.setLevel(level)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(stream)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
    _LOGGING_CONFIGURED = True


class EmoMVVideoDataset(Dataset):
    """Reads a split CSV (train.csv/val.csv/test.csv) produced by Phase 1/2.

    mode="train": random temporal sampling (one random frame per uniform segment).
    mode in {"val","test"}: deterministic uniform sampling (segment midpoints) --
    same item always returns the same frames, required for reproducible eval.
    """

    def __init__(
        self,
        csv_path,
        mode: str = "train",
        num_frames: int = 16,
        resize_shorter_edge: int = 256,
        crop_size: int = 224,
        log_path: Optional[Path] = None,
    ):
        assert mode in ("train", "val", "test"), f"invalid mode: {mode}"
        configure_logging(log_path)

        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.mode = mode
        self.num_frames = num_frames
        self.resize_shorter_edge = resize_shorter_edge
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.df)

    # ── temporal sampling ────────────────────────────────────────────
    def _sample_frame_indices(self, start_frame: int, end_frame: int) -> List[int]:
        n = max(end_frame - start_frame, 1)
        seg_size = n / self.num_frames
        indices = []
        for i in range(self.num_frames):
            seg_start, seg_end = i * seg_size, (i + 1) * seg_size
            if self.mode == "train":
                pos = seg_start + np.random.uniform(0, max(seg_end - seg_start, 1e-6))
            else:
                pos = (seg_start + seg_end) / 2.0
            indices.append(start_frame + int(min(pos, n - 1)))
        return indices

    # ── spatial transform: resize shorter edge -> crop_size center crop ──
    def _spatial_transform(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if h <= w:
            new_h = self.resize_shorter_edge
            new_w = max(self.crop_size, round(w * (new_h / h)))
        else:
            new_w = self.resize_shorter_edge
            new_h = max(self.crop_size, round(h * (new_w / w)))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (new_h - self.crop_size) // 2
        left = (new_w - self.crop_size) // 2
        cropped = resized[top:top + self.crop_size, left:left + self.crop_size]
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # ── decoding: single sequential pass over the requested frame indices ──
    def _decode_clip(self, video_path: str, frame_indices: List[int]) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        captured = {}
        if not cap.isOpened():
            return captured
        target_set = sorted(set(frame_indices))
        max_target = target_set[-1]
        ti, idx = 0, 0
        while ti < len(target_set):
            ok, frame = cap.read()
            if not ok:
                break
            if idx == target_set[ti]:
                captured[idx] = self._spatial_transform(frame)
                ti += 1
            idx += 1
            if idx > max_target:
                break
        cap.release()
        return captured

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        video_id = row["video_id"]
        video_path = row["video_path"]
        start_frame = int(row.get("start_frame", 0))
        end_frame = int(row.get("end_frame", row.get("orig_frame_count", start_frame + 1)))

        frame_indices = self._sample_frame_indices(start_frame, end_frame)

        try:
            captured = self._decode_clip(video_path, frame_indices)
        except Exception as exc:  # corrupt/unreadable file at read time
            logger.warning(f"{video_id}: exception during decode ({video_path}): {exc}")
            captured = {}

        frames, n_missing = [], 0
        for idx in frame_indices:
            if idx in captured:
                frames.append(captured[idx])
            else:
                n_missing += 1
                frames.append(np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8))

        if not captured:
            logger.warning(f"{video_id}: clip failed to decode entirely, returning all-zero tensor ({video_path})")
        elif n_missing:
            logger.warning(f"{video_id}: {n_missing}/{self.num_frames} frames failed to decode, "
                            f"zero-padded ({video_path})")

        clip = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,C) in [0,1]
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()  # (T,C,H,W)
        return clip, int(row["emotion_id"]), video_id
