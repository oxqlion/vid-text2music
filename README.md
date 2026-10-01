# vid-text2music — EmoMV video emotion → music generation

Pipeline built on the EmoMV video/music emotion dataset. It works in three stages:

| Stage | Goal | Entry point |
|---|---|---|
| **A. Classify** | Predict a video's emotion (`exciting`, `fear`, `tense`, `sad`, `relax`) using frozen VideoMAE + CLIP embeddings and a small fusion head | `project/training/train_fusion.py` |
| **B. Align** | Project video embeddings and 5 text mood anchors into a shared space with a label-anchored contrastive loss | `project/training/train_alignment.py` |
| **C. Generate** | Train a bridge adapter that conditions a frozen MusicGen on the video's mood embedding (optionally combined with MusicGen's own T5 text conditioning) | `project/training/train_bridge.py` |

Stage A results (VideoMAE+CLIP fusion: **80.1% test accuracy, 0.794 macro-F1**, vs. 50.6% for the SlowFast baseline) are written up in [`project/RESULTS.md`](project/RESULTS.md). Design notes for Stages B and C are in [`project/docs/`](project/docs/).

## Repository layout

```
.
├── requirements.txt            # Python dependencies
├── emomv_eda.ipynb             # exploratory data analysis
├── emomv_video_eda_2.py        # video metadata EDA script
├── eda_outputs/                # EDA figures and CSVs
└── project/
    ├── configs/config.yaml     # single source of truth: paths, hyperparameters, thresholds
    ├── preprocessing/          # manifest building, dedup, duration windowing, grouped split
    ├── features/               # frozen-encoder / SlowFast / audio-target extraction
    ├── datasets/               # video and cached-feature datasets
    ├── models/                 # baseline MLP, fusion classifier, adapters, bridge, conditioning
    ├── training/               # training scripts, loop, losses, utils
    ├── evaluation/             # evaluate.py, evaluate_alignment.py, generate_samples.py
    ├── text_mood_clf_encoder/  # text → mood classifier (see USE_text_mood_clf.md)
    ├── notebooks/              # 01–08 notebooks walking through each phase
    ├── data/                   # manifest and train/val/test CSVs
    └── docs/                   # plans and paper summary (.tex / .pdf)
```

## Setup

Requires Python 3.11+ (developed on macOS Apple Silicon, using the MPS backend) and FFmpeg for video metadata.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg            # provides ffprobe
```

## Data and large files (not in git)

The following are git-ignored because they are large or regenerable:

- **Raw EmoMV dataset** (`aria/`, `EmoMV.z*`, `EmoMV.zip`) — download it and set `paths.base_dir` in `project/configs/config.yaml` to its location.
- **Cached features** (`project/features/{videomae,clip,slowfast,video_embedding_512,audio_targets}/`) — regenerate with the scripts in `project/features/`.
- **Checkpoints** (`project/training/checkpoints/`) — regenerate with the training scripts.
- **Text mood classifier weights** (`project/text_mood_clf_encoder/text_mood_clf*`) — see [`USE_text_mood_clf.md`](project/text_mood_clf_encoder/USE_text_mood_clf.md).
- **Generated audio** (`*.wav`, including `project/evaluation/results/generated_samples/`) and logs.

## Running the pipeline

Run everything from the repository root. All scripts read `project/configs/config.yaml`.

```bash
# 1. Preprocessing: MATCH-only manifest, dedup, windowing, then a group-aware 70/15/15 split
python project/preprocessing/build_manifest.py
python project/preprocessing/split.py

# 2. Feature extraction (frozen encoders; idempotent)
python project/features/extract_features.py             # VideoMAE (768-d) + CLIP (512-d)
python project/features/extract_slowfast_baseline.py    # shipped SlowFast features (baseline)

# 3. Stage A: train and evaluate classifiers
python project/training/train_baseline.py
python project/training/train_fusion.py
python project/evaluation/evaluate.py --model baseline
python project/evaluation/evaluate.py --model fusion

# 4. Stage B: cross-modal alignment
python project/features/extract_video_embedding_512.py
python project/training/train_alignment.py
python project/evaluation/evaluate_alignment.py

# 5. Stage C: MusicGen bridge
python project/features/extract_audio_targets.py
python project/training/train_bridge.py
python project/evaluation/generate_samples.py           # writes .wav samples for listening
```

The notebooks in `project/notebooks/` (`01_manifest` … `08_bridge`) run the same phases interactively.

## Key design decisions

- **MATCH-only clips.** Only `match_lbl == 1` rows are used, which avoids the label ambiguity of audio-swapped MISMATCH pairs.
- **Rebuilt split.** The official split leaks groups: segments of one source video appear in train, val and test. The split is rebuilt grouped on `source_group_id` and stratified on emotion.
- **Frozen encoders.** VideoMAE, CLIP, MusicGen and the text encoder are never fine-tuned. Only small heads and adapters are trained.
- **Domain shift.** DS1 is about 83% of clips, and DS2/DS3 test slices are small. Read per-dataset numbers with that in mind (see `project/RESULTS.md`).
