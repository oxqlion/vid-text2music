"""Stage B evaluation — video-to-text Recall@1 (overall + per-DS1/DS2/DS3),
the modality-gap diagnostic, and a t-SNE view of the shared space.

Note on scope: with only 5 fixed text anchors (no text training corpus --
see project/docs/cross_modal_alignment_plan.pdf), video->text R@1 reduces to
5-way classification accuracy using the text anchors as prototypes, and
text->video retrieval isn't meaningful to report (there is no diversity on
the text side to retrieve *among*). Both are documented here rather than
silently only reporting the flattering half.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.feature_dataset import CachedEmbeddingDataset  # noqa: E402
from project.models.adapters import AlignmentModel  # noqa: E402
from project.models.encoders import get_device  # noqa: E402
from project.training.alignment_common import load_text_anchors  # noqa: E402
from project.training.utils import load_checkpoint  # noqa: E402


@torch.no_grad()
def embed_split(model: AlignmentModel, video_dir: Path, csv_path: Path, device, batch_size: int = 128):
    ds = CachedEmbeddingDataset(csv_path, [(video_dir, "video_id")])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_z, all_labels, all_ids = [], [], []
    for x, y, video_ids in loader:
        all_z.append(model.encode_video(x.to(device)).cpu())
        all_labels.append(y)
        all_ids.extend(video_ids)
    return torch.cat(all_z), torch.cat(all_labels), all_ids


def compute_metrics(video_z: torch.Tensor, labels: torch.Tensor, text_z: torch.Tensor, id2label: dict) -> dict:
    logits = video_z @ text_z.t()
    preds = logits.argmax(dim=-1).numpy()
    labels_arr = labels.numpy()
    n_classes = text_z.shape[0]

    per_class_recall, modality_gap = {}, {}
    for c in range(n_classes):
        mask = labels_arr == c
        if mask.sum() == 0:
            per_class_recall[id2label[c]] = None
            modality_gap[id2label[c]] = None
            continue
        per_class_recall[id2label[c]] = float((preds[mask] == c).mean())
        centroid = video_z[mask].mean(dim=0)
        centroid = centroid / centroid.norm()
        modality_gap[id2label[c]] = float((centroid @ text_z[c]).item())

    return {
        "n_samples": int(len(labels_arr)),
        "v2t_accuracy": float((preds == labels_arr).mean()),
        "per_class_recall": per_class_recall,
        "modality_gap_cosine_sim": modality_gap,
    }


def plot_tsne(video_z: torch.Tensor, labels: torch.Tensor, text_z: torch.Tensor, id2label: dict, out_path: Path):
    n_classes = text_z.shape[0]
    all_points = torch.cat([video_z, text_z], dim=0).numpy()
    perplexity = min(30, max(5, len(video_z) // 20))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca").fit_transform(all_points)
    video_coords, text_coords = coords[: len(video_z)], coords[len(video_z):]

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    labels_arr = labels.numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(n_classes):
        mask = labels_arr == c
        ax.scatter(video_coords[mask, 0], video_coords[mask, 1], s=10, alpha=0.5,
                   color=colors[c], label=f"video: {id2label[c]}")
    for c in range(n_classes):
        ax.scatter(text_coords[c, 0], text_coords[c, 1], s=400, marker="*", color=colors[c],
                   edgecolor="black", linewidth=1.5, label=f"text: {id2label[c]}", zorder=5)
    ax.set_title("Stage B shared latent space (t-SNE)\n(stars = 5 fixed text anchors, dots = video embeddings)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = load_config()
    a_cfg = cfg["alignment"]
    device = get_device(cfg["encoders"]["device_preference"])
    results_dir = resolve_path(cfg, "paths.results_dir")

    model = AlignmentModel(
        shared_dim=a_cfg["shared_dim"], dropout=a_cfg["adapter_dropout"],
        logit_scale_init=a_cfg["logit_scale_init"], logit_scale_max=a_cfg["logit_scale_max"],
    ).to(device)
    ckpt_path = resolve_path(cfg, "paths.checkpoints_dir") / "alignment_adapters_best.pt"
    load_checkpoint(model, ckpt_path, map_location=device)
    model.eval()

    text_z_raw, id2label = load_text_anchors(cfg, device)
    with torch.no_grad():
        text_z = model.encode_text(text_z_raw.to(device)).cpu()

    video_dir = resolve_path(cfg, "paths.video_embedding_512_dir")
    test_csv = resolve_path(cfg, "paths.test_csv")
    video_z, labels, video_ids = embed_split(model, video_dir, test_csv, device)

    results = {"overall": compute_metrics(video_z, labels, text_z, id2label)}
    print(f"[alignment] OVERALL  n={results['overall']['n_samples']}  "
          f"v2t_acc={results['overall']['v2t_accuracy']:.4f}")
    print(f"  per-class recall: {results['overall']['per_class_recall']}")
    print(f"  modality gap (cosine sim to own class's text anchor, higher = closer): "
          f"{results['overall']['modality_gap_cosine_sim']}")

    test_meta = pd.read_csv(test_csv).set_index("video_id")["dataset_source"]
    ds_labels = np.array([test_meta[vid] for vid in video_ids])
    for ds in sorted(set(ds_labels.tolist())):
        mask = ds_labels == ds
        results[ds] = compute_metrics(video_z[mask], labels[mask], text_z, id2label)
        print(f"[alignment] {ds:>4}  n={results[ds]['n_samples']}  v2t_acc={results[ds]['v2t_accuracy']:.4f}")

    out_json = results_dir / "alignment_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[alignment] metrics -> {out_json}")

    tsne_path = results_dir / "alignment_tsne.png"
    plot_tsne(video_z, labels, text_z, id2label, tsne_path)
    print(f"[alignment] t-SNE plot -> {tsne_path}")


if __name__ == "__main__":
    main()
