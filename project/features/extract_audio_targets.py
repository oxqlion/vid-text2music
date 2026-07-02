"""Stage C (data prep) — extract each clip's audio track as a waveform,
resampled to MusicGen's expected format, respecting windowed sub-clips'
start/end frame boundaries.

This is the real (mood-embedding, music) pairing Stage C trains on: EmoMV's
MATCH clips have video emotion == music emotion by the dataset's own
construction (see project/docs/cross_modal_alignment_plan.pdf), so every
clip's own soundtrack is already a mood-consistent generation target -- no
new data collection needed, just extraction.

Audio is used as-is (may include vocals/dialogue from the source music
videos, by deliberate choice for this first milestone -- see plan doc).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402


def extract_one(video_path: str, start_frame: int, end_frame: int, fps: float,
                 sample_rate: int, channels: int, out_path: Path) -> bool:
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_time:.4f}",
        "-i", str(video_path),
        "-t", f"{duration:.4f}",
        "-vn", "-ac", str(channels), "-ar", str(sample_rate),
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and out_path.exists()


def extract_audio_targets(cfg: dict):
    sc_cfg = cfg["stage_c"]
    sample_rate = sc_cfg["audio_sample_rate"]
    channels = sc_cfg["audio_channels"]
    out_dir = resolve_path(cfg, "paths.features_dir") / "audio_targets"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(resolve_path(cfg, "paths.manifest_csv"))
    manifest = manifest[manifest["split"].notna() & (manifest["split"] != "")]

    n_ok, n_fail, n_skip = 0, 0, 0
    failures = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting audio"):
        out_path = out_dir / f"{row['video_id']}.wav"
        if out_path.exists():
            n_skip += 1
            continue
        ok = extract_one(
            row["video_path"], int(row["start_frame"]), int(row["end_frame"]), float(row["fps"]),
            sample_rate, channels, out_path,
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append(row["video_id"])

    print(f"Audio targets -> {out_dir}  (extracted {n_ok}, already-cached {n_skip}, failed {n_fail})")
    if failures:
        print(f"  [WARN] failed extractions (first 10): {failures[:10]}")


def main():
    cfg = load_config()
    extract_audio_targets(cfg)


if __name__ == "__main__":
    main()
