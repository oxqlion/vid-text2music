"""Cached-embedding dataset consumed by Phase 5 training loops.

Reads one or more pre-extracted `<key>.pt` embedding directories and
concatenates them per row. `key_column` lets a feature source be keyed by
`video_id` (VideoMAE/CLIP -- extracted per window) or by `parent_video_id`
(SlowFast -- extracted once per original clip, shared by all of its windows).
"""
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedEmbeddingDataset(Dataset):
    def __init__(self, csv_path, feature_specs: List[Tuple[str, str]]):
        """feature_specs: list of (feature_dir, key_column) pairs, concatenated in order."""
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.feature_specs = [(Path(d), col) for d, col in feature_specs]

        def has_all_features(row) -> bool:
            return all((d / f"{row[col]}.pt").exists() for d, col in self.feature_specs)

        mask = self.df.apply(has_all_features, axis=1)
        n_missing = int((~mask).sum())
        if n_missing:
            print(f"  [WARN] {n_missing}/{len(self.df)} rows missing cached features, dropping "
                  f"(run feature extraction first)")
        self.df = self.df[mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        feats = [torch.load(d / f"{row[col]}.pt") for d, col in self.feature_specs]
        x = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        return x.float(), int(row["emotion_id"]), row["video_id"]
