"""Stage B (data prep) — cache the 512-d penultimate FusionClassifier
embedding for every clip, mirroring the existing videomae/clip/slowfast
feature-caching pattern (see features/extract_features.py,
features/extract_slowfast_baseline.py).

This is the frozen, task-supervised video representation Stage B's
cross-modal alignment projects from (see project/docs/cross_modal_alignment_plan.pdf,
Section "Which embedding to project") -- not the raw 1280-d VideoMAE+CLIP
concat, and not the 5-class logits.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.feature_dataset import CachedEmbeddingDataset  # noqa: E402
from project.models.encoders import get_device  # noqa: E402
from project.models.fusion_classifier import FusionClassifier  # noqa: E402
from project.training.utils import load_checkpoint  # noqa: E402


def extract_video_embeddings(cfg: dict):
    device = get_device(cfg["encoders"]["device_preference"])
    videomae_dir = resolve_path(cfg, "paths.videomae_features_dir")
    clip_dir = resolve_path(cfg, "paths.clip_features_dir")
    out_dir = resolve_path(cfg, "paths.features_dir") / "video_embedding_512"
    out_dir.mkdir(parents=True, exist_ok=True)

    fusion_ckpt = resolve_path(cfg, "paths.checkpoints_dir") / "fusion_classifier_best.pt"
    assert fusion_ckpt.exists(), (
        f"{fusion_ckpt} not found -- train the fusion classifier (Phase 5) before Stage B."
    )
    model = FusionClassifier(
        hidden_dim=cfg["training"]["fusion_hidden_dim"],
        num_classes=len(cfg["emotion_to_id"]),
        dropout=cfg["training"]["fusion_dropout"],
    )
    load_checkpoint(model, fusion_ckpt)
    model = model.to(device).eval()

    for split_name, path_key in (("train", "paths.train_csv"), ("val", "paths.val_csv"), ("test", "paths.test_csv")):
        csv_path = resolve_path(cfg, path_key)
        ds = CachedEmbeddingDataset(csv_path, [(videomae_dir, "video_id"), (clip_dir, "video_id")])

        todo = [i for i in range(len(ds)) if not (out_dir / f"{ds.df.iloc[i]['video_id']}.pt").exists()]
        print(f"[{split_name}] {len(todo)}/{len(ds)} clips pending 512-d embedding extraction")
        if not todo:
            continue

        loader = DataLoader(torch.utils.data.Subset(ds, todo), batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, _labels, video_ids in tqdm(loader, desc=f"Video-512 [{split_name}]"):
                emb = model(x.to(device), return_embedding=True).cpu()
                for i, vid in enumerate(video_ids):
                    torch.save(emb[i].clone(), out_dir / f"{vid}.pt")

    print(f"512-d video embeddings -> {out_dir}")


def main():
    cfg = load_config()
    extract_video_embeddings(cfg)


if __name__ == "__main__":
    main()
