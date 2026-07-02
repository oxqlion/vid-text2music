#!/usr/bin/env python3
"""
EmoMV Dataset — Comprehensive Video EDA
Analyzes label distribution, temporal/spatial metrics, and data health
across all three EmoMV sub-datasets (DS1/DS2/DS3).

Metadata is extracted via ffprobe (no frames loaded).
"""

import os
import re
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/rafiabhista/Documents/code/apple/c1_audio/emomv/media/EmoMV")
OUT_DIR  = Path("/Users/rafiabhista/Documents/code/apple/c1_audio/emomv")

ANNOTATION_FILES = [
    BASE_DIR / "Dataset1/annotation/DS1_TRAIN_MATCH_MISMATCH_labels.csv",
    BASE_DIR / "Dataset1/annotation/DS1_VAL_MATCH_MISMATCH_labels.csv",
    BASE_DIR / "Dataset1/annotation/DS1_TEST_MATCH_MISMATCH_labels.csv",
    BASE_DIR / "Dataset2/annotation/DS2_TRAIN_MATCH_MISMATCH_labels.csv",
    BASE_DIR / "Dataset2/annotation/DS2_VAL_MATCH_MISMATCH_labels.csv",
    BASE_DIR / "Dataset3/annotation/DS3_TRAIN_MATCH_MISMATCH_labels.csv",
    BASE_DIR / "Dataset3/annotation/DS3_VAL_MATCH_MISMATCH_labels.csv",
]

# Known emotion classes (normalised to lower-case)
KNOWN_EMOTIONS = {"exciting", "sad", "relax", "fear", "tense"}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def normalise_emotion(raw: str) -> str:
    s = raw.strip().lower()
    # Map common aliases
    aliases = {"happy": "exciting", "contentment": "relax", "relaxing": "relax",
               "excitement": "exciting", "fearful": "fear", "tension": "tense"}
    return aliases.get(s, s)


def parse_csvs() -> pd.DataFrame:
    """Parse all annotation CSVs into a unified DataFrame."""
    rows = []
    for csvf in ANNOTATION_FILES:
        if not csvf.exists():
            print(f"  [WARN] CSV not found: {csvf}")
            continue
        dataset = csvf.parts[-3]           # e.g. Dataset1
        split   = csvf.stem.split("_")[1]  # TRAIN / VAL / TEST
        with open(csvf, newline="", encoding="utf-8") as fh:
            for line in csv.reader(fh):
                if len(line) < 5:
                    continue
                rel_dir, stem, match_lbl, vid_emo, mus_emo = (
                    line[0].strip(), line[1].strip(), line[2].strip(),
                    line[3].strip(), line[4].strip()
                )
                full_path = BASE_DIR / rel_dir / (stem + ".mp4")
                rows.append({
                    "dataset":    dataset,
                    "split":      split.lower(),
                    "match_lbl":  int(match_lbl),
                    "vid_emotion": normalise_emotion(vid_emo),
                    "mus_emotion": normalise_emotion(mus_emo),
                    "rel_path":   str(Path(rel_dir) / (stem + ".mp4")),
                    "abs_path":   str(full_path),
                    "exists":     full_path.exists(),
                })
    return pd.DataFrame(rows)


def scan_disk_videos() -> list[Path]:
    """Walk the EmoMV tree and collect every .mp4 file."""
    return sorted(BASE_DIR.rglob("*.mp4"))


def extract_emotion_from_filename(stem: str) -> str:
    """
    DS1 MISMATCH filenames follow: {vid_emotion}_{code}_PAIR_{mus_emotion}_{code}
    Extract the leading emotion word.
    """
    m = re.match(r"^([a-zA-Z]+)_", stem)
    if m:
        return normalise_emotion(m.group(1))
    return "unknown"


def ffprobe_metadata(path: str) -> dict | None:
    """Return video metadata dict via ffprobe (no frame decoding)."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None

    meta = {"corrupt": False}

    # Duration from format container (most reliable)
    fmt = data.get("format", {})
    meta["duration_s"]  = float(fmt.get("duration", 0)) or 0.0
    meta["file_size_mb"] = round(float(fmt.get("size", 0)) / 1_048_576, 3)

    # Find the video stream
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        meta["width"]  = int(stream.get("width", 0))
        meta["height"] = int(stream.get("height", 0))

        # FPS from avg_frame_rate (e.g., "30/1") or r_frame_rate
        fps_str = stream.get("avg_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            meta["fps"] = round(float(num) / float(den), 4) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            meta["fps"] = 0.0

        nb = stream.get("nb_frames")
        if nb and nb != "N/A":
            meta["frame_count"] = int(nb)
        else:
            # Estimate from duration × fps
            meta["frame_count"] = int(meta["duration_s"] * meta["fps"])

        meta["codec"] = stream.get("codec_name", "unknown")
        break

    if "width" not in meta:
        meta["corrupt"] = True  # no video stream found

    return meta


def aspect_ratio_str(w: int, h: int) -> str:
    if w == 0 or h == 0:
        return "unknown"
    d = gcd(w, h)
    return f"{w // d}:{h // d}"


def orientation(w: int, h: int) -> str:
    if w == 0 or h == 0:
        return "unknown"
    if w > h:
        return "landscape"
    if h > w:
        return "portrait"
    return "square"


def print_section(title: str):
    width = 72
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_bar(label: str, count: int, total: int, bar_width: int = 30):
    pct   = 100 * count / total if total else 0
    filled = int(bar_width * pct / 100)
    bar   = "█" * filled + "░" * (bar_width - filled)
    print(f"  {label:<22s} {bar}  {count:>5d}  ({pct:>5.1f}%)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("  EmoMV Dataset — Comprehensive Video EDA")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # ── 1. Load annotations ──────────────────────────────────────────────────
    print_section("1 · PARSING ANNOTATION FILES")
    ann_df = parse_csvs()
    print(f"  Total annotation entries : {len(ann_df):,}")
    print(f"  Entries with file on disk: {ann_df['exists'].sum():,}")
    print(f"  Entries with MISSING file: {(~ann_df['exists']).sum():,}")

    # Break down missing by dataset
    missing = ann_df[~ann_df["exists"]]
    if len(missing):
        print(f"\n  Missing breakdown by dataset:")
        for ds, cnt in missing.groupby("dataset").size().items():
            print(f"    {ds}: {cnt:,} entries missing "
                  f"(match_lbl=1: {missing[(missing['dataset']==ds)&(missing['match_lbl']==1)].shape[0]:,}, "
                  f"match_lbl=0: {missing[(missing['dataset']==ds)&(missing['match_lbl']==0)].shape[0]:,})")
        print("  NOTE: Dataset1 MATCH subdirs (DS1_*_MATCH/) are absent from disk —")
        print("        their SlowFast/VGGish features are in extracted_features/ instead.")

    # ── 2. Scan all physical video files ────────────────────────────────────
    print_section("2 · DISK INVENTORY")
    disk_files = scan_disk_videos()
    print(f"  Physical .mp4 files found: {len(disk_files):,}")
    ds_counts = Counter(p.parts[p.parts.index("EmoMV") + 1] for p in disk_files)
    for ds, cnt in sorted(ds_counts.items()):
        print(f"    {ds}: {cnt} files")

    # ── 3. Resolve label for every physical file ─────────────────────────────
    # Build a path → emotion map from the CSV (DS2/DS3 where files exist)
    path_to_emotion = {}
    for _, row in ann_df[ann_df["exists"]].iterrows():
        path_to_emotion[row["abs_path"]] = row["vid_emotion"]

    file_records = []
    for p in disk_files:
        abs_str = str(p)
        if abs_str in path_to_emotion:
            emo = path_to_emotion[abs_str]
        else:
            # DS1 MISMATCH files: infer from filename prefix
            emo = extract_emotion_from_filename(p.stem)
        file_records.append({"abs_path": abs_str, "vid_emotion": emo})

    files_df = pd.DataFrame(file_records)

    # ── 4. Label distribution ────────────────────────────────────────────────
    print_section("3 · LABEL DISTRIBUTION  (physical video files on disk)")
    total_videos = len(files_df)
    emo_counts = files_df["vid_emotion"].value_counts()
    for emo, cnt in emo_counts.items():
        print_bar(emo, cnt, total_videos)

    max_cls  = emo_counts.max()
    min_cls  = emo_counts.min()
    imbalance_ratio = max_cls / min_cls if min_cls else float("inf")
    print(f"\n  Imbalance ratio (max/min): {imbalance_ratio:.2f}×")
    if imbalance_ratio > 3.0:
        print("  ⚠  SEVERE class imbalance detected (>3×). Weighting recommended.")
    elif imbalance_ratio > 1.5:
        print("  ⚠  Moderate imbalance detected. Consider soft weighting.")
    else:
        print("  ✓  Classes are roughly balanced.")

    # Also show per-dataset split distribution
    print_section("3b · LABEL DISTRIBUTION  (from annotations, by dataset)")
    ann_exist = ann_df[ann_df["exists"]]
    if len(ann_exist):
        pivot = ann_exist.groupby(["dataset", "vid_emotion"]).size().unstack(fill_value=0)
        print(pivot.to_string())
    # Full annotation (including missing files) to show intended distribution
    print("\n  Intended distribution (all annotation rows incl. missing files):")
    full_emo_counts = ann_df["vid_emotion"].value_counts()
    for emo, cnt in full_emo_counts.items():
        print_bar(emo, cnt, len(ann_df))

    # ── 5. Extract video metadata via ffprobe ────────────────────────────────
    print_section("4 · EXTRACTING VIDEO METADATA  (ffprobe, no frames loaded)")
    print(f"  Processing {len(disk_files):,} files ... ", end="", flush=True)

    metadata_rows = []
    corrupt_files = []
    for i, p in enumerate(disk_files):
        meta = ffprobe_metadata(str(p))
        if meta is None or meta.get("corrupt"):
            corrupt_files.append(str(p))
            metadata_rows.append({
                "abs_path": str(p),
                "vid_emotion": files_df.loc[files_df["abs_path"] == str(p), "vid_emotion"].values[0]
                              if str(p) in files_df["abs_path"].values else "unknown",
                "corrupt": True,
                "duration_s": None, "fps": None, "frame_count": None,
                "width": None, "height": None, "file_size_mb": None, "codec": None,
            })
        else:
            emo = files_df.loc[files_df["abs_path"] == str(p), "vid_emotion"].values
            emo = emo[0] if len(emo) else "unknown"
            metadata_rows.append({
                "abs_path":    str(p),
                "vid_emotion": emo,
                "corrupt":     False,
                **meta,
            })
        if (i + 1) % 100 == 0:
            print(f"{i + 1}", end=" ", flush=True)

    print("done.")
    meta_df = pd.DataFrame(metadata_rows)

    # Add derived columns
    valid = meta_df[~meta_df["corrupt"]].copy()
    valid["aspect_ratio"] = valid.apply(
        lambda r: aspect_ratio_str(int(r["width"] or 0), int(r["height"] or 0)), axis=1)
    valid["orientation"] = valid.apply(
        lambda r: orientation(int(r["width"] or 0), int(r["height"] or 0)), axis=1)
    valid["resolution"] = valid.apply(
        lambda r: f"{int(r['width'] or 0)}×{int(r['height'] or 0)}", axis=1)

    # ── 6. Health checks ─────────────────────────────────────────────────────
    print_section("5 · HEALTH & VALIDATION")

    print(f"  Corrupt / unreadable files   : {len(corrupt_files)}")
    if corrupt_files:
        for f in corrupt_files[:10]:
            print(f"    ✗ {f}")
        if len(corrupt_files) > 10:
            print(f"    ... and {len(corrupt_files) - 10} more")

    # Duplicate paths in annotations
    dup_ann = ann_df[ann_df.duplicated("rel_path", keep=False)]
    print(f"  Duplicate entries in CSV     : {len(dup_ann)}")
    if len(dup_ann):
        print(f"    (first 5 duplicated paths:)")
        for p in dup_ann["rel_path"].unique()[:5]:
            print(f"    {p}")

    # Missing files (CSV entries with no disk file)
    print(f"  CSV entries with no disk file: {(~ann_df['exists']).sum():,}")

    # Files on disk with no CSV annotation
    disk_paths_set = {str(p) for p in disk_files}
    ann_paths_set  = set(ann_df["abs_path"].tolist())
    unannotated    = disk_paths_set - ann_paths_set
    print(f"  Disk files with no CSV entry : {len(unannotated):,}")
    if unannotated:
        for p in sorted(unannotated)[:5]:
            print(f"    {p}")

    # Zero-duration videos
    zero_dur = valid[valid["duration_s"] <= 0]
    print(f"  Zero-duration videos         : {len(zero_dur)}")

    # ── 7. Temporal profiling ────────────────────────────────────────────────
    print_section("6 · TEMPORAL PROFILING")
    dur = valid["duration_s"].dropna()
    fps = valid["fps"].dropna()
    fc  = valid["frame_count"].dropna()

    print(f"\n  Duration (seconds):")
    print(f"    Min    : {dur.min():.2f}s")
    print(f"    Max    : {dur.max():.2f}s")
    print(f"    Mean   : {dur.mean():.2f}s")
    print(f"    Median : {dur.median():.2f}s")
    print(f"    Std    : {dur.std():.2f}s")

    # Histogram bins
    bins = [0, 5, 10, 15, 20, 30, 45, 60, 120, float("inf")]
    labels_bins = ["0-5s","5-10s","10-15s","15-20s","20-30s","30-45s","45-60s","60-120s",">120s"]
    dur_binned = pd.cut(dur, bins=bins, labels=labels_bins, right=False)
    print(f"\n  Duration histogram:")
    for lbl, cnt in dur_binned.value_counts().sort_index().items():
        print_bar(str(lbl), cnt, len(dur))

    print(f"\n  FPS:")
    fps_counts = fps.round(2).value_counts().head(10)
    for fval, cnt in fps_counts.sort_index().items():
        print_bar(f"{fval:.2f} fps", cnt, len(fps))

    print(f"\n  Frame count:")
    print(f"    Min    : {int(fc.min()):,}")
    print(f"    Max    : {int(fc.max()):,}")
    print(f"    Mean   : {fc.mean():.0f}")
    print(f"    Median : {fc.median():.0f}")
    print(f"    Std    : {fc.std():.0f}")

    # Per-emotion duration
    print(f"\n  Median duration per emotion class:")
    emo_dur = valid.groupby("vid_emotion")["duration_s"].agg(["median","mean","std","count"])
    print(emo_dur.round(2).to_string())

    # ── 8. Spatial profiling ─────────────────────────────────────────────────
    print_section("7 · SPATIAL PROFILING")

    print(f"\n  Unique resolutions (top 15):")
    res_counts = valid["resolution"].value_counts().head(15)
    for res, cnt in res_counts.items():
        print_bar(res, cnt, len(valid))

    print(f"\n  Unique aspect ratios:")
    ar_counts = valid["aspect_ratio"].value_counts()
    for ar, cnt in ar_counts.items():
        print_bar(ar, cnt, len(valid))

    print(f"\n  Orientation:")
    ori_counts = valid["orientation"].value_counts()
    for ori, cnt in ori_counts.items():
        print_bar(ori, cnt, len(valid))

    # ── 9. Save metadata ─────────────────────────────────────────────────────
    print_section("8 · SAVING OUTPUTS")

    # Full metadata CSV
    out_csv = OUT_DIR / "emomv_video_metadata.csv"
    meta_out = meta_df.copy()
    if "aspect_ratio" not in meta_out.columns:
        meta_out = meta_out.merge(
            valid[["abs_path","aspect_ratio","orientation","resolution"]],
            on="abs_path", how="left"
        )
    meta_out.to_csv(out_csv, index=False)
    print(f"  Saved metadata CSV → {out_csv}")

    # Summary JSON
    summary = {
        "run_at": datetime.now().isoformat(),
        "total_disk_videos": len(disk_files),
        "total_annotation_rows": len(ann_df),
        "annotation_files_resolved": int(ann_df["exists"].sum()),
        "annotation_files_missing": int((~ann_df["exists"]).sum()),
        "corrupt_videos": len(corrupt_files),
        "corrupt_video_paths": corrupt_files,
        "unannotated_disk_files": len(unannotated),
        "duplicate_annotation_entries": len(dup_ann),
        "label_distribution_disk": emo_counts.to_dict(),
        "label_distribution_annotations": full_emo_counts.to_dict(),
        "imbalance_ratio": round(imbalance_ratio, 3),
        "duration": {
            "min_s":    round(float(dur.min()), 3),
            "max_s":    round(float(dur.max()), 3),
            "mean_s":   round(float(dur.mean()), 3),
            "median_s": round(float(dur.median()), 3),
            "std_s":    round(float(dur.std()), 3),
        },
        "fps_distribution": {str(k): int(v) for k, v in fps.round(2).value_counts().items()},
        "frame_count": {
            "min":    int(fc.min()),
            "max":    int(fc.max()),
            "mean":   round(float(fc.mean()), 1),
            "median": round(float(fc.median()), 1),
        },
        "resolution_distribution": res_counts.to_dict(),
        "aspect_ratio_distribution": ar_counts.to_dict(),
        "orientation_distribution": ori_counts.to_dict(),
    }
    out_json = OUT_DIR / "emomv_eda_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved summary JSON  → {out_json}")

    # ── 10. Recommendations ──────────────────────────────────────────────────
    print_section("9 · RECOMMENDATIONS REPORT")

    # (a) Duration / temporal
    dur_range = dur.max() - dur.min()
    med_dur   = dur.median()
    pct_short = (dur < 5).mean() * 100
    pct_long  = (dur > 30).mean() * 100

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  (a)  TEMPORAL HANDLING STRATEGY                                    │
  └─────────────────────────────────────────────────────────────────────┘""")
    print(f"  Duration range: {dur.min():.1f}s – {dur.max():.1f}s  "
          f"(median {med_dur:.1f}s, {pct_short:.1f}% under 5s, {pct_long:.1f}% over 30s)")
    if dur_range > 20:
        print("""
  → RECOMMENDATION: Use a SLIDING WINDOW approach (uniform temporal sampling):
      • Sample T=16 frames uniformly from each video (VideoMAE default).
        This handles variable-length videos without padding/truncation artefacts.
      • For very short clips (<5s at ≥24fps → <120 frames), repeat frames
        cyclically to reach T=16 (torch.index_select trick).
      • If you need fixed-duration clips (e.g., for a contrastive pair pipeline),
        truncate to the median duration (~{:.0f}s) and pad shorter videos
        by looping.  Do NOT zero-pad — it corrupts the temporal gradient.
      • Save the sampled-frame indices so experiments are reproducible.""".format(med_dur))
    else:
        print("""
  → Duration variance is low. Simple uniform sampling of T=16 frames is sufficient.
    No padding/truncation needed.""")

    # (b) Spatial
    landscape_pct = ori_counts.get("landscape", 0) / len(valid) * 100
    portrait_pct  = ori_counts.get("portrait", 0)  / len(valid) * 100
    top_ar        = ar_counts.index[0] if len(ar_counts) else "16:9"

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  (b)  SPATIAL RESIZING STRATEGY                                     │
  └─────────────────────────────────────────────────────────────────────┘""")
    print(f"  Dominant aspect ratio: {top_ar}  "
          f"(landscape {landscape_pct:.1f}% | portrait {portrait_pct:.1f}%)")
    print(f"""
  → RECOMMENDATION:
      • Resize shorter edge to 256, then center-crop to 224×224 (standard
        for VideoMAE ViT-B/16 and CLIP ViT-B/32).
      • Use torchvision transforms.Resize(256) then CenterCrop(224).
      • For portrait videos (H > W), resize width to 256 first to avoid
        extreme letterboxing — apply Resize((256, 256)) then CenterCrop(224).
      • Normalise with ImageNet μ/σ (both VideoMAE and CLIP were trained on it):
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      • Do NOT resize to 224 directly without intermediate step — aliasing
        artifacts degrade CLIP visual features noticeably.""")

    # (c) Class imbalance
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  (c)  CLASS IMBALANCE WEIGHTING STRATEGY                            │
  └─────────────────────────────────────────────────────────────────────┘""")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}×")
    print(f"  Per-class counts: {emo_counts.to_dict()}")
    if imbalance_ratio > 3.0:
        print(f"""
  → SEVERE IMBALANCE — use BOTH of these strategies:

    1. Inverse-frequency loss weights (for CrossEntropyLoss):
         n_samples = {total_videos}
         class_weights = n_samples / (n_classes × count_per_class)
         Plug into: nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    2. Weighted random sampler (ensures balanced mini-batches):
         from torch.utils.data import WeightedRandomSampler
         sample_weights = [1/count[label] for each sample]
         sampler = WeightedRandomSampler(sample_weights, len(dataset))

    3. For contrastive (InfoNCE) pre-training:
         Balanced sampling per class is more important than loss weighting.
         Build your batch to have equal class representation.

    4. Avoid oversampling duplicate videos — use random temporal augmentations
       (random clip start, horizontal flip, colour jitter) on minority classes
       instead.  This effectively oversamples without exact duplication.""")
    elif imbalance_ratio > 1.5:
        print("""
  → MODERATE IMBALANCE — inverse-frequency loss weights are sufficient.
    Weighted sampler is optional but recommended if validation F1 on minority
    classes drops significantly during training.""")
    else:
        print("""
  → Classes are balanced. Standard uniform sampling is fine.
    Monitor per-class F1 on validation set and apply mild label smoothing
    (label_smoothing=0.1) to reduce overconfidence.""")

    print_section("EDA COMPLETE")
    print(f"  Outputs saved to: {OUT_DIR}")
    print(f"    • emomv_video_metadata.csv  — per-video metrics")
    print(f"    • emomv_eda_summary.json    — aggregated statistics\n")


if __name__ == "__main__":
    main()
