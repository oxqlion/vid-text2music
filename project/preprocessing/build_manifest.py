"""Phase 1 — build the clean MATCH-only manifest that is the source of truth
for every downstream phase (splitting, dataset loading, feature extraction).

Pipeline: parse annotations -> filter MATCH-only -> drop missing/corrupt ->
hash + exact-dedup -> derive source_group_id (for leakage-free splitting) ->
flag duration outliers -> expand long clips into windows -> write manifest.csv.
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.preprocessing.duration_windows import compute_windows  # noqa: E402

# DS1: "..._name-2-of-8.mp4" -> strip "-2-of-8" to recover the shared source id.
SEGMENT_SUFFIX_RE = re.compile(r"-\d+-of-\d+$")
# DS3: "<youtube_id>_02.mp4" -> strip the trailing 2-digit segment index.
TRAILING_INDEX_RE = re.compile(r"_\d{2,}$")


def normalise_emotion(raw: str, aliases: dict) -> str:
    s = raw.strip().lower()
    return aliases.get(s, s)


def derive_group_id(dataset: str, stem: str) -> str:
    if dataset == "DS1":
        return SEGMENT_SUFFIX_RE.sub("", stem)
    if dataset == "DS3":
        return TRAILING_INDEX_RE.sub("", stem)
    return stem  # DS2: bare YouTube ID, no observed segmentation


def parse_annotations(cfg: dict) -> pd.DataFrame:
    """Read every annotation CSV (headerless, 7 cols) and keep MATCH-only rows."""
    base_dir = resolve_path(cfg, "paths.base_dir")
    aliases = cfg["emotion_aliases"]
    emotion_to_id = cfg["emotion_to_id"]
    keep_lbl = cfg["match_lbl_keep"]

    rows = []
    for entry in cfg["annotation_files"]:
        dataset, split, csv_rel = entry["dataset"], entry["split"], entry["csv"]
        csv_path = base_dir / csv_rel
        if not csv_path.exists():
            print(f"  [WARN] annotation file not found: {csv_path}")
            continue
        n_rows, n_kept = 0, 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for line in csv.reader(fh):
                if len(line) < 5:
                    continue
                n_rows += 1
                match_lbl_raw = line[2].strip()
                match_lbl = int(match_lbl_raw) if match_lbl_raw.lstrip("-").isdigit() else -1
                if match_lbl != keep_lbl:
                    continue
                stem = line[1].strip()
                vid_emo = normalise_emotion(line[3], aliases)
                if vid_emo not in emotion_to_id:
                    print(f"  [WARN] unrecognised emotion '{vid_emo}' for {stem} in {csv_path.name}, skipping")
                    continue
                rel_dir = line[0].strip()
                abs_path = base_dir / rel_dir / f"{stem}.mp4"
                rows.append({
                    "dataset_source": dataset,
                    "orig_split": split,
                    "stem": stem,
                    "emotion_label": vid_emo,
                    "emotion_id": emotion_to_id[vid_emo],
                    "video_path": str(abs_path),
                })
                n_kept += 1
        print(f"  {dataset}/{split}: {n_rows} rows -> {n_kept} MATCH rows ({csv_path.name})")

    df = pd.DataFrame(rows)
    print(f"Total MATCH annotation rows parsed: {len(df)}")
    return df


def drop_missing_files(df: pd.DataFrame) -> pd.DataFrame:
    exists = df["video_path"].apply(lambda p: Path(p).exists())
    n_missing = (~exists).sum()
    if n_missing:
        print(f"  [WARN] {n_missing} MATCH clips missing on disk, dropping")
    return df[exists].reset_index(drop=True)


def probe_video(path: str):
    """Cheap container-level probe via OpenCV: fps, frame_count, decodability."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, _ = cap.read()
    cap.release()
    if not ok or fps <= 0 or frame_count <= 0:
        return None
    return {"fps": fps, "frame_count": frame_count, "duration": frame_count / fps}


def probe_all(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for path in tqdm(df["video_path"], desc="Probing videos (OpenCV)"):
        records.append(probe_video(path))
    probed = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, probed], axis=1)

    corrupt = df[df["fps"].isna()]
    if len(corrupt):
        print(f"  [WARN] {len(corrupt)} clips unreadable/corrupt, dropping")
    df = df.dropna(subset=["fps", "frame_count", "duration"]).reset_index(drop=True)
    df["frame_count"] = df["frame_count"].astype(int)
    return df


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_all(df: pd.DataFrame) -> pd.DataFrame:
    hashes = [sha256_of_file(p) for p in tqdm(df["video_path"], desc="Hashing videos (SHA-256)")]
    df = df.copy()
    df["video_hash"] = hashes
    return df


def dedupe_exact(df: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    seen: dict[str, tuple[str, str]] = {}  # hash -> (path, emotion_label)
    keep_mask = []
    dup_rows = []
    for _, row in df.iterrows():
        h = row["video_hash"]
        if h in seen:
            kept_path, kept_label = seen[h]
            keep_mask.append(False)
            dup_rows.append({
                "video_hash": h,
                "kept_path": kept_path,
                "kept_label": kept_label,
                "dropped_path": row["video_path"],
                "dropped_label": row["emotion_label"],
                "label_conflict": kept_label != row["emotion_label"],
            })
        else:
            seen[h] = (row["video_path"], row["emotion_label"])
            keep_mask.append(True)

    n_dupes = sum(not k for k in keep_mask)
    print(f"Exact duplicates (identical file hash): {n_dupes} found, removed")
    dup_df = pd.DataFrame(dup_rows)
    dup_df.to_csv(report_path, index=False)
    print(f"  -> duplicate report: {report_path}")

    n_conflicts = int(dup_df["label_conflict"].sum()) if len(dup_df) else 0
    if n_conflicts:
        print(f"  [WARN] {n_conflicts} duplicate pair(s) have CONFLICTING emotion labels "
              f"(identical video content, different label) -- resolved by keeping the "
              f"first-seen annotation. Review these rows in {report_path.name} manually:")
        for _, r in dup_df[dup_df["label_conflict"]].iterrows():
            print(f"      kept='{r['kept_label']}' ({Path(r['kept_path']).name})  vs  "
                  f"dropped='{r['dropped_label']}' ({Path(r['dropped_path']).name})")

    return df[keep_mask].reset_index(drop=True)


def base_fields(row: pd.Series) -> dict:
    return {
        "video_id": row["video_id"],
        "video_path": row["video_path"],
        "emotion_label": row["emotion_label"],
        "emotion_id": row["emotion_id"],
        "dataset_source": row["dataset_source"],
        "match_type": "match",
        "video_hash": row["video_hash"],
        "source_group_id": row["source_group_id"],
        "orig_duration_s": round(float(row["duration"]), 4),
        "orig_frame_count": int(row["frame_count"]),
        "fps": round(float(row["fps"]), 4),
    }


def expand_with_windows(df: pd.DataFrame, cfg: dict, outliers_path: Path) -> pd.DataFrame:
    threshold = cfg["duration"]["max_duration_s"] + cfg["duration"]["tolerance_s"]
    window_s = cfg["duration"]["window_s"]
    min_remainder_s = cfg["duration"]["min_window_remainder_s"]

    outliers = df[df["duration"] > threshold].copy()
    outliers.to_csv(outliers_path, index=False)
    print(f"Duration outliers (> {threshold}s): {len(outliers)} clips -> inspect at {outliers_path}")

    manifest_rows = []
    n_windowed_parents = 0
    for _, row in df.iterrows():
        base = base_fields(row)
        if row["duration"] <= threshold:
            manifest_rows.append({
                **base,
                "window_index": 0,
                "parent_video_id": row["video_id"],
                "start_frame": 0,
                "end_frame": int(row["frame_count"]),
                "window_duration_s": round(float(row["duration"]), 4),
                "split": "",
            })
            continue

        n_windowed_parents += 1
        windows = compute_windows(row["duration"], row["fps"], row["frame_count"], window_s, min_remainder_s)
        for w in windows:
            manifest_rows.append({
                **base,
                "video_id": f'{row["video_id"]}__win{w["window_index"]}',
                "window_index": w["window_index"],
                "parent_video_id": row["video_id"],
                "start_frame": w["start_frame"],
                "end_frame": w["end_frame"],
                "window_duration_s": round(w["duration_s"], 4),
                "split": "",
            })

    n_window_rows = sum(1 for r in manifest_rows if r["parent_video_id"] != r["video_id"])
    print(f"Long clips windowed: {n_windowed_parents} parent videos -> {n_window_rows} window rows")
    return pd.DataFrame(manifest_rows)


def build_manifest(cfg: dict) -> pd.DataFrame:
    print("=" * 70)
    print("PHASE 1 — Building manifest")
    print("=" * 70)

    df = parse_annotations(cfg)
    df = drop_missing_files(df)
    df = probe_all(df)
    df = hash_all(df)

    duplicates_path = resolve_path(cfg, "paths.duplicates_csv")
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    df = dedupe_exact(df, duplicates_path)

    df["video_id"] = df["dataset_source"] + "__" + df["stem"]
    df["source_group_id"] = df.apply(
        lambda r: f'{r["dataset_source"]}__{derive_group_id(r["dataset_source"], r["stem"])}', axis=1
    )

    outliers_path = resolve_path(cfg, "paths.duration_outliers_csv")
    manifest = expand_with_windows(df, cfg, outliers_path)

    manifest_path = resolve_path(cfg, "paths.manifest_csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    print("-" * 70)
    print(f"Manifest rows       : {len(manifest)}")
    print(f"Unique source videos : {manifest['parent_video_id'].nunique()}")
    print(f"Per-dataset counts  :\n{manifest.groupby('dataset_source').size()}")
    print(f"Per-emotion counts  :\n{manifest.groupby('emotion_label').size()}")
    print(f"Saved -> {manifest_path}")
    print("=" * 70)
    return manifest


def main():
    cfg = load_config()
    build_manifest(cfg)


if __name__ == "__main__":
    main()
