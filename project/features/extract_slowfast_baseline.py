"""Phase 5 (baseline) — align the dataset's shipped SlowFast .h5 features with
our MATCH-only manifest and cache them as features/slowfast/<parent_video_id>.pt.

The shipped h5 files store one row per line of the *original* MATCH+MISMATCH
annotation CSV, in the same order (verified directly: row i of
SlowFast_DS1_TRAIN_MATCH_MISMATCH.h5 == row i of
DS1_TRAIN_MATCH_MISMATCH_labels.csv). We re-walk those CSVs, keep only
match_lbl==1 rows, and slice out the matching h5 row.

SlowFast features exist only for whole original clips, not per-window -- a
windowed manifest row's `parent_video_id` maps back to the same cached
feature as its parent, via `datasets.feature_dataset.CachedEmbeddingDataset`.
"""
import csv
import sys
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402


def h5_path_for(base_dir: Path, dataset: str, split: str) -> Path:
    folder = dataset.replace("DS", "Dataset")  # DS1 -> Dataset1
    return base_dir / folder / "extracted_features" / f"SlowFast_{dataset}_{split.upper()}_MATCH_MISMATCH.h5"


def extract_slowfast_features(cfg: dict):
    base_dir = resolve_path(cfg, "paths.base_dir")
    out_dir = resolve_path(cfg, "paths.features_dir") / "slowfast"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_lbl = cfg["match_lbl_keep"]

    total_saved, total_skipped = 0, 0
    for entry in cfg["annotation_files"]:
        dataset, split, csv_rel = entry["dataset"], entry["split"], entry["csv"]
        csv_path = base_dir / csv_rel
        h5_path = h5_path_for(base_dir, dataset, split)
        if not csv_path.exists() or not h5_path.exists():
            print(f"  [WARN] missing csv or h5 for {dataset}/{split}, skipping")
            continue

        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        with h5py.File(h5_path, "r") as f:
            feats = f["default"][:]

        assert len(rows) == feats.shape[0], (
            f"{dataset}/{split}: csv rows ({len(rows)}) != h5 rows ({feats.shape[0]}) -- "
            "row-order alignment assumption violated"
        )

        n_saved = 0
        for i, line in enumerate(tqdm(rows, desc=f"SlowFast [{dataset}/{split}]", leave=False)):
            if len(line) < 5:
                continue
            match_lbl_raw = line[2].strip()
            match_lbl = int(match_lbl_raw) if match_lbl_raw.lstrip("-").isdigit() else -1
            if match_lbl != keep_lbl:
                continue
            stem = line[1].strip()
            video_id = f"{dataset}__{stem}"
            out_path = out_dir / f"{video_id}.pt"
            if out_path.exists():
                total_skipped += 1
                continue
            torch.save(torch.from_numpy(feats[i]).clone(), out_path)
            n_saved += 1

        total_saved += n_saved
        print(f"  {dataset}/{split}: {n_saved} SlowFast features saved")

    print(f"SlowFast features -> {out_dir}  (saved {total_saved}, already-cached {total_skipped})")


def main():
    cfg = load_config()
    extract_slowfast_features(cfg)


if __name__ == "__main__":
    main()
