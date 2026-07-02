# EmoMV Video Emotion Classification — Frozen-Encoder Fusion: Results

**Date:** 2026-07-01
**Pipeline:** `project/` (see `configs/config.yaml` for the exact run configuration)

## 1. Objective

Classify video emotion (5 classes: `exciting`, `fear`, `tense`, `sad`, `relax`) from
the EmoMV dataset by extracting embeddings from **frozen, pretrained VideoMAE
(ViT-B) and CLIP (ViT-B/32) encoders** and training only a lightweight fusion
classifier on top — no end-to-end fine-tuning. A SlowFast-feature MLP baseline
(using the dataset's own shipped features) was trained first as a reference
point.

## 2. Data & preprocessing

Source: EmoMV (DS1/DS2/DS3), **MATCH-only** clips (`match_lbl==1`), to avoid the
label ambiguity/leakage risk of MISMATCH audio-swapped pairs.

| Step | Result |
|---|---|
| MATCH annotation rows parsed | 3,178 |
| Exact duplicate clips removed (SHA-256 hash) | 34 (2 had conflicting emotion labels between the duplicate pair — resolved by keeping the first-seen annotation) |
| Genuine duration outliers (>35s, vs. the ~30s nominal length) | 20 clips, windowed into 30s chunks (65 window rows total) |
| **Final manifest rows** | **3,189** |

**Split:** train/val/test rebuilt from scratch (70/15/15) — the dataset's
*official* split was found to leak groups (same source video's segments
spread across train/val/test in DS1). The new split is grouped on
`source_group_id` (a source video's segments/windows always land together)
and stratified on `emotion_id`.

| Split | Rows | DS1 | DS2 | DS3 |
|---|---|---|---|---|
| train | 2,232 | 1,870 | 190 | 172 |
| val | 479 | 395 | 59 | 25 |
| test | 478 | 388 | 59 | 31 |

DS1 dominates (83% of clips) — relevant when interpreting per-dataset results below.

## 3. Models

- **Baseline:** shipped SlowFast features (2304-d) → `Linear(2304,256) → ReLU → Dropout(0.3) → Linear(256,5)`.
- **Fusion:** frozen `VideoMAE-base` (768-d, mean-pooled) + frozen `CLIP ViT-B/32` (512-d, mean-pooled over 16 frames) → concat(1280-d) → `Linear(1280,512) → GELU → Dropout(0.3) → Linear(512,5)`.

Both trained with class-weighted cross-entropy (inverse-frequency weights),
Adam (lr=3e-4, wd=1e-4), batch size 32, early stopping on validation macro-F1
(patience 10), seed 42.

## 4. Results

### 4.1 Headline

| Model | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | Best epoch (val macro-F1) |
|---|---|---|---|---|
| SlowFast baseline | 50.6% | 0.479 | 0.503 | 44 (0.525) |
| **VideoMAE+CLIP fusion** | **80.1%** | **0.794** | **0.803** | 10 (0.807) |

The fusion model beats the baseline by **+29.5 points accuracy / +0.32 macro-F1**,
and converges in far fewer epochs (10 vs. 44) — frozen VideoMAE/CLIP embeddings
carry substantially more emotion-discriminative signal than the shipped SlowFast
features, even feeding an identically-sized MLP head. Val and test macro-F1 for
fusion are close (0.807 vs 0.794), so there's no obvious overfitting to the
validation split.

### 4.2 Per-class (fusion, overall test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| exciting | 0.896 | 0.827 | 0.860 | 104 |
| fear | 0.655 | 0.889 | 0.754 | 81 |
| tense | 0.783 | 0.691 | 0.734 | 68 |
| sad | 0.766 | 0.809 | 0.787 | 89 |
| relax | 0.898 | 0.779 | 0.835 | 136 |

`tense` is the weakest class; `exciting`/`relax` are strongest. The baseline's
per-class F1 (not shown in full) ranges 0.35–0.65, uniformly worse and with
`sad`/`tense` particularly weak (F1 0.35, 0.36) — consistent with those being
lower-arousal, less visually-distinctive classes for a simple 2304-d motion
descriptor.

### 4.3 Per-dataset breakdown (domain shift)

| Dataset | n (test) | Fusion Accuracy | Fusion Macro-F1 |
|---|---|---|---|
| DS1 | 388 | 86.6% | 0.860 |
| DS2 | 59 | 52.5% | 0.473 |
| DS3 | 31 | 51.6% | 0.366 |

This reproduces the domain-shift signal flagged in the original EDA (KS tests
on duration distributions were significant for every DS1/DS2/DS3 pair). **However,
part of this gap is a small-sample artifact, not purely generalization failure:**
DS2/DS3 test slices are tiny and per-class support is very uneven —

| Class | DS1 | DS2 | DS3 |
|---|---|---|---|
| exciting | 74 | 18 | 12 |
| fear | 76 | 5 | **0** |
| tense | 54 | 10 | 4 |
| sad | 65 | 17 | 7 |
| relax | 119 | 9 | 8 |

DS3's test set has **zero** true `fear` examples and only 4 `tense` examples (all
4 misclassified as `fear` — see 4.4). A macro-F1 averaged over such thin classes
is high-variance; the DS2/DS3 numbers should be read as "consistent with a real
domain gap, but not statistically conclusive at this sample size," not as a firm
generalization failure.

### 4.4 Confusion analysis (fusion, overall)

```
              pred:  exciting  fear  tense  sad  relax
true: exciting          86      12     2     1     3
      fear                0     72     6     3     0
      tense                2     17    47     2     0
      sad                  1      4     3    72     9
      relax                7      5     2    16   106
```

| True class | % correct | Top confusion |
|---|---|---|
| exciting | 82.7% | → fear (11.5%) |
| fear | 88.9% | → tense (7.4%) |
| tense | 69.1% | → **fear (25.0%)** |
| sad | 80.9% | → relax (10.1%) |
| relax | 77.9% | → sad (11.8%) |

Errors cluster between **adjacent** regions of the valence-arousal plane
(`tense`/`fear` are both negative-valence; `relax`/`sad` are both low-arousal)
and essentially never cross to opposite quadrants (e.g. `exciting` is almost
never confused with `sad`). This is a positive signal that the fused embeddings
capture real affective structure rather than spurious correlations. The single
largest error mode is `tense → fear` (25% of true `tense` clips), which is the
main lever for a follow-up modeling iteration.

## 5. Limitations

- DS2/DS3 evaluation is statistically underpowered (see 4.3) — conclusions
  about cross-dataset generalization need more data or a resampling-based
  confidence interval before being stated firmly.
- Encoders are fully frozen; embeddings are mean-pooled over 16 frames with no
  temporal modeling beyond VideoMAE's own internal attention — no explicit
  motion/dynamics fusion between VideoMAE and CLIP streams.
- MISMATCH clips and the `for_retrieval` subset (466 clips) are excluded by
  design (see project plan) — not a limitation of this experiment, but scope
  to note if comparing against the original EmoMV paper's numbers.

## 6. Artifacts

- Metrics: `evaluation/results/{baseline_slowfast,fusion_videomae_clip}_metrics.json`
- Confusion matrices: `evaluation/results/*_confusion_matrix_{overall,DS1,DS2,DS3}.png`
- Checkpoints: `training/checkpoints/{baseline_slowfast_mlp,fusion_classifier}_best.pt`
- Manifest / splits: `data/{manifest,train,val,test}.csv`

Checkpoints, cached features (`features/{videomae,clip,slowfast,video_embedding_512,audio_targets}/`), and
`text_mood_clf_encoder/text_mood_clf/` are gitignored (large binaries, cheap to regenerate — see
`.gitignore` for the full list). To regenerate a checkpoint, rerun the corresponding training script
against `configs/config.yaml`: `training/train_baseline.py`, `training/train_fusion.py`,
`training/train_alignment.py` (Stage B), or `training/train_bridge.py` (Stage C, Milestone 1) — each
writes its `*_best.pt` back to `training/checkpoints/`.
