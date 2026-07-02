"""Phase 6 — evaluation: overall metrics + the per-DS1/DS2/DS3 breakdown the
EDA's confirmed domain shift (KS tests, all p~0 across DS1/DS2/DS3 duration
distributions) makes necessary -- a model that only fits DS1 should show up
here as a DS2/DS3 gap, not just a mediocre pooled number.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.training.utils import load_checkpoint  # noqa: E402

ID_TO_EMOTION_DEFAULT = {0: "exciting", 1: "fear", 2: "tense", 3: "sad", 4: "relax"}


@torch.no_grad()
def predict(model, dataset, device, batch_size: int = 32, num_workers: int = 4) -> pd.DataFrame:
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = []
    for x, y, video_ids in loader:
        logits = model(x.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
        for vid, true_id, pred_id in zip(video_ids, y.numpy(), preds):
            rows.append({"video_id": vid, "true_id": int(true_id), "pred_id": int(pred_id)})
    return pd.DataFrame(rows)


def compute_metrics(preds_df: pd.DataFrame, id_to_emotion: dict) -> dict:
    y_true, y_pred = preds_df["true_id"], preds_df["pred_id"]
    labels = sorted(id_to_emotion.keys())
    target_names = [id_to_emotion[i] for i in labels]

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n_samples": len(preds_df),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {name: report[name] for name in target_names},
        "confusion_matrix": cm.tolist(),
        "labels": target_names,
    }


def plot_confusion_matrix(cm: list, labels: list, title: str, out_path: Path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(model, dataset, test_csv_path, cfg: dict, device, model_name: str, results_dir: Path) -> dict:
    id_to_emotion = {v: k for k, v in cfg["emotion_to_id"].items()}

    preds_df = predict(model, dataset, device, cfg["training"]["batch_size"], cfg["training"]["num_workers"])

    test_meta = pd.read_csv(test_csv_path)[["video_id", "dataset_source"]]
    preds_df = preds_df.merge(test_meta, on="video_id", how="left")

    results = {"overall": compute_metrics(preds_df, id_to_emotion)}
    print(f"\n[{model_name}] OVERALL  n={results['overall']['n_samples']}  "
          f"acc={results['overall']['accuracy']:.4f}  macro_f1={results['overall']['macro_f1']:.4f}  "
          f"weighted_f1={results['overall']['weighted_f1']:.4f}")
    plot_confusion_matrix(results["overall"]["confusion_matrix"], results["overall"]["labels"],
                           f"{model_name} — overall confusion matrix",
                           results_dir / f"{model_name}_confusion_matrix_overall.png")

    for ds in sorted(preds_df["dataset_source"].dropna().unique()):
        sub = preds_df[preds_df["dataset_source"] == ds]
        if sub.empty:
            continue
        results[ds] = compute_metrics(sub, id_to_emotion)
        print(f"[{model_name}] {ds:>4}     n={results[ds]['n_samples']}  "
              f"acc={results[ds]['accuracy']:.4f}  macro_f1={results[ds]['macro_f1']:.4f}  "
              f"weighted_f1={results[ds]['weighted_f1']:.4f}")
        plot_confusion_matrix(results[ds]["confusion_matrix"], results[ds]["labels"],
                               f"{model_name} — {ds} confusion matrix",
                               results_dir / f"{model_name}_confusion_matrix_{ds}.png")

    out_path = results_dir / f"{model_name}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[{model_name}] metrics -> {out_path}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "fusion"], required=True)
    args = parser.parse_args()

    cfg = load_config()
    from project.models.encoders import FrozenCLIPEncoder, FrozenVideoMAEEncoder, get_device
    device = get_device(cfg["encoders"]["device_preference"])
    results_dir = resolve_path(cfg, "paths.results_dir")

    if args.model == "baseline":
        from project.datasets.feature_dataset import CachedEmbeddingDataset
        from project.models.baseline_mlp import SlowFastBaselineMLP

        slowfast_dir = resolve_path(cfg, "paths.features_dir") / "slowfast"
        test_csv = resolve_path(cfg, "paths.test_csv")
        test_ds = CachedEmbeddingDataset(test_csv, [(slowfast_dir, "parent_video_id")])
        model = SlowFastBaselineMLP(
            input_dim=2304, hidden_dim=cfg["training"]["baseline_hidden_dim"],
            num_classes=len(cfg["emotion_to_id"]), dropout=cfg["training"]["baseline_dropout"],
        )
        ckpt_path = resolve_path(cfg, "paths.checkpoints_dir") / "baseline_slowfast_mlp_best.pt"
        model_name = "baseline_slowfast"
    else:
        from project.datasets.feature_dataset import CachedEmbeddingDataset
        from project.models.fusion_classifier import FusionClassifier

        videomae_dir = resolve_path(cfg, "paths.videomae_features_dir")
        clip_dir = resolve_path(cfg, "paths.clip_features_dir")
        test_csv = resolve_path(cfg, "paths.test_csv")
        test_ds = CachedEmbeddingDataset(test_csv, [(videomae_dir, "video_id"), (clip_dir, "video_id")])
        model = FusionClassifier(
            videomae_dim=FrozenVideoMAEEncoder.EMBED_DIM, clip_dim=FrozenCLIPEncoder.EMBED_DIM,
            hidden_dim=cfg["training"]["fusion_hidden_dim"], num_classes=len(cfg["emotion_to_id"]),
            dropout=cfg["training"]["fusion_dropout"],
        )
        ckpt_path = resolve_path(cfg, "paths.checkpoints_dir") / "fusion_classifier_best.pt"
        model_name = "fusion_videomae_clip"

    load_checkpoint(model, ckpt_path)
    evaluate_model(model, test_ds, test_csv, cfg, device, model_name, results_dir)


if __name__ == "__main__":
    main()
