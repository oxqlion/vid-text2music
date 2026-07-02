"""Phase 2 — group-aware, emotion-stratified 70/15/15 split.

The dataset's *official* train/val/test assignment has confirmed group
leakage: DS1 clips segmented from the same source video ("...-2-of-8.mp4"
style) land in different splits (37 groups span train/val, 23 train/test, 34
val/test in the official split). We therefore rebuild the split from scratch,
grouped on `source_group_id` (so a source video's segments/windows always land
in the same split) and stratified on `emotion_id` (so class proportions track
the global distribution in every split).

Algorithm: greedy iterative stratification. Groups are shuffled deterministically
then, one at a time, assigned to whichever split currently has the largest
"deficit" against its target fraction *for that group's emotion*. This gives
precise 70/15/15 control without the awkward fold-size rounding of
StratifiedGroupKFold, while still guaranteeing zero cross-split group leakage
by construction (each group is assigned to exactly one split).
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402

SPLITS = ("train", "val", "test")


def assign_groups_to_splits(manifest: pd.DataFrame, cfg: dict) -> dict:
    group_col = cfg["split"]["group_col"]
    stratify_col = cfg["split"]["stratify_col"]
    target_frac = {
        "train": cfg["split"]["train_frac"],
        "val": cfg["split"]["val_frac"],
        "test": cfg["split"]["test_frac"],
    }
    assert abs(sum(target_frac.values()) - 1.0) < 1e-6, "split fractions must sum to 1.0"

    groups = manifest.groupby(group_col).agg(
        size=("video_id", "count"),
        emotion_id=(stratify_col, lambda s: s.mode().iloc[0]),
        n_unique_emotions=(stratify_col, "nunique"),
    ).reset_index()

    impure = groups[groups["n_unique_emotions"] > 1]
    if len(impure):
        print(f"  [WARN] {len(impure)} group(s) contain rows with different emotion_id "
              f"(unexpected -- using majority label per group): {impure[group_col].tolist()}")

    rng = np.random.RandomState(cfg["seed"])
    groups = groups.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    counts = {s: defaultdict(int) for s in SPLITS}     # counts[split][emotion_id] = rows assigned so far
    totals_per_emotion = defaultdict(int)               # rows seen so far for this emotion (any split)
    assignment = {}

    for _, g in groups.iterrows():
        e, size, gid = g["emotion_id"], g["size"], g[group_col]
        totals_per_emotion[e] += size
        best_split, best_deficit = None, -np.inf
        for s in SPLITS:
            target = target_frac[s] * totals_per_emotion[e]
            deficit = target - counts[s][e]
            if deficit > best_deficit:
                best_deficit, best_split = deficit, s
        assignment[gid] = best_split
        counts[best_split][e] += size

    return assignment


def verify_split(manifest: pd.DataFrame, cfg: dict):
    group_col = cfg["split"]["group_col"]

    group_splits = manifest.groupby(group_col)["split"].nunique()
    leaking = group_splits[group_splits > 1]
    assert len(leaking) == 0, f"Group leakage detected across splits: {leaking.index.tolist()}"
    print("  [OK] no source_group_id crosses a split boundary")

    print("\nSplit sizes (rows):")
    print(manifest["split"].value_counts().reindex(list(SPLITS)))

    print("\nPer-split emotion distribution (%):")
    print((manifest.groupby(["split", "emotion_label"]).size()
           .unstack(fill_value=0)
           .reindex(list(SPLITS))
           .pipe(lambda t: 100 * t.div(t.sum(axis=1), axis=0))
           .round(1)))

    print("\nPer-split dataset_source distribution (rows):")
    print(manifest.groupby(["split", "dataset_source"]).size().unstack(fill_value=0).reindex(list(SPLITS)))


def build_splits(cfg: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("PHASE 2 — Grouped + stratified 70/15/15 split")
    print("=" * 70)

    assignment = assign_groups_to_splits(manifest, cfg)
    manifest = manifest.copy()
    manifest["split"] = manifest[cfg["split"]["group_col"]].map(assignment)
    assert manifest["split"].notna().all(), "every manifest row must receive a split"

    verify_split(manifest, cfg)
    print("=" * 70)
    return manifest


def main():
    cfg = load_config()
    manifest_path = resolve_path(cfg, "paths.manifest_csv")
    manifest = pd.read_csv(manifest_path)

    manifest = build_splits(cfg, manifest)

    # Persist the split assignment back into the manifest (single source of truth)...
    manifest.to_csv(manifest_path, index=False)
    # ...and export the three flat CSVs consumers actually read from.
    for split_name, path_key in (("train", "paths.train_csv"), ("val", "paths.val_csv"), ("test", "paths.test_csv")):
        out_path = resolve_path(cfg, path_key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest[manifest["split"] == split_name].drop(columns=["split"]).to_csv(out_path, index=False)
        print(f"Saved {split_name} split -> {out_path}")


if __name__ == "__main__":
    main()
