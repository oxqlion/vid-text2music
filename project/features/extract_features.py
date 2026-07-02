"""Phase 4 — batched, idempotent embedding extraction for frozen VideoMAE + CLIP.

Saves one tensor per clip:
  features/videomae/<video_id>.pt  (768-d)
  features/clip/<video_id>.pt      (512-d)

Extraction always uses deterministic (val-style) uniform temporal sampling --
even for the train split -- because embeddings are computed once and cached,
not re-sampled per epoch. Re-running is safe: clips whose .pt files already
exist for both encoders are skipped.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.video_dataset import EmoMVVideoDataset  # noqa: E402
from project.models.encoders import FrozenCLIPEncoder, FrozenVideoMAEEncoder, get_device  # noqa: E402


def pending_indices(ds: EmoMVVideoDataset, videomae_dir: Path, clip_dir: Path) -> list:
    pending = []
    for i in range(len(ds)):
        vid = ds.df.iloc[i]["video_id"]
        if not ((videomae_dir / f"{vid}.pt").exists() and (clip_dir / f"{vid}.pt").exists()):
            pending.append(i)
    return pending


def extract_split(
    csv_path: Path,
    split_name: str,
    cfg: dict,
    videomae_enc: FrozenVideoMAEEncoder,
    clip_enc: FrozenCLIPEncoder,
    videomae_dir: Path,
    clip_dir: Path,
):
    ds = EmoMVVideoDataset(
        csv_path,
        mode="val",  # deterministic sampling for cached, one-shot embeddings (see module docstring)
        num_frames=cfg["dataset"]["num_frames"],
        resize_shorter_edge=cfg["dataset"]["resize_shorter_edge"],
        crop_size=cfg["dataset"]["crop_size"],
        log_path=resolve_path(cfg, "paths.logs_dir") / "dataset_errors.log",
    )

    todo = pending_indices(ds, videomae_dir, clip_dir)
    print(f"[{split_name}] {len(todo)}/{len(ds)} clips pending extraction")
    if not todo:
        return

    loader = DataLoader(
        Subset(ds, todo),
        batch_size=cfg["encoders"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    for clips, _labels, video_ids in tqdm(loader, desc=f"Extracting [{split_name}]"):
        vmae_emb = videomae_enc(clips)  # (B, 768)
        clip_emb = clip_enc(clips)      # (B, 512)
        for i, vid in enumerate(video_ids):
            torch.save(vmae_emb[i].clone(), videomae_dir / f"{vid}.pt")
            torch.save(clip_emb[i].clone(), clip_dir / f"{vid}.pt")


def main():
    cfg = load_config()
    device = get_device(cfg["encoders"]["device_preference"])
    print(f"Device: {device}")

    videomae_enc = FrozenVideoMAEEncoder(cfg["encoders"]["videomae_model"], device=device)
    clip_enc = FrozenCLIPEncoder(cfg["encoders"]["clip_model"], device=device)

    videomae_dir = resolve_path(cfg, "paths.videomae_features_dir")
    clip_dir = resolve_path(cfg, "paths.clip_features_dir")
    videomae_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    for split_name, path_key in (("train", "paths.train_csv"), ("val", "paths.val_csv"), ("test", "paths.test_csv")):
        csv_path = resolve_path(cfg, path_key)
        extract_split(csv_path, split_name, cfg, videomae_enc, clip_enc, videomae_dir, clip_dir)


if __name__ == "__main__":
    main()
