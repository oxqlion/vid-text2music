"""Stage B — train the video/text projection adapters with the label-anchored
symmetric contrastive loss (project/training/losses.py). Both frozen encoders
that produced the cached embeddings (FusionClassifier backbone, text_mood_clf)
are never touched here -- only AlignmentModel's two adapters + temperature
are trainable.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.feature_dataset import CachedEmbeddingDataset  # noqa: E402
from project.models.adapters import AlignmentModel  # noqa: E402
from project.models.encoders import get_device  # noqa: E402
from project.training.alignment_common import load_text_anchors  # noqa: E402
from project.training.losses import label_anchored_contrastive_loss  # noqa: E402
from project.training.utils import EarlyStopping, save_checkpoint, set_seed  # noqa: E402


@torch.no_grad()
def evaluate_v2t_accuracy(model: AlignmentModel, text_z_raw: torch.Tensor, loader: DataLoader, device) -> float:
    """video-to-text Recall@1 -- with exactly 5 fixed text anchors this is
    identical to 5-way classification accuracy using the text anchors as
    class prototypes (see project/docs/cross_modal_alignment_plan.pdf)."""
    model.eval()
    text_z = model.encode_text(text_z_raw.to(device))  # (5, D)
    correct, total = 0, 0
    for x, y, _video_ids in loader:
        video_z = model.encode_video(x.to(device))
        preds = (video_z @ text_z.t()).argmax(dim=-1).cpu()
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def main():
    cfg = load_config()
    a_cfg = cfg["alignment"]
    set_seed(cfg["seed"])
    device = get_device(cfg["encoders"]["device_preference"])
    print(f"Device: {device}")

    video_dir = resolve_path(cfg, "paths.video_embedding_512_dir")
    train_ds = CachedEmbeddingDataset(resolve_path(cfg, "paths.train_csv"), [(video_dir, "video_id")])
    val_ds = CachedEmbeddingDataset(resolve_path(cfg, "paths.val_csv"), [(video_dir, "video_id")])
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    text_z_raw, id2label = load_text_anchors(cfg, device)
    print(f"text anchors (fixed, 5 class-name strings, no training corpus available): {id2label}")

    model = AlignmentModel(
        shared_dim=a_cfg["shared_dim"],
        dropout=a_cfg["adapter_dropout"],
        logit_scale_init=a_cfg["logit_scale_init"],
        logit_scale_max=a_cfg["logit_scale_max"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a_cfg["lr"], weight_decay=a_cfg["weight_decay"])
    train_loader = DataLoader(train_ds, batch_size=a_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=a_cfg["batch_size"], shuffle=False)

    early_stopping = EarlyStopping(patience=a_cfg["early_stopping_patience"], mode="max")
    checkpoint_path = resolve_path(cfg, "paths.checkpoints_dir") / "alignment_adapters_best.pt"

    for epoch in range(1, a_cfg["max_epochs"] + 1):
        model.train()
        running = {"total": 0.0, "loss_v2t": 0.0, "loss_t2v": 0.0}
        n_batches = 0
        for x, y, _video_ids in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            video_z = model.encode_video(x)
            text_z = model.encode_text(text_z_raw.to(device))
            loss, parts = label_anchored_contrastive_loss(video_z, text_z, y, model.logit_scale())
            loss.backward()
            optimizer.step()
            for k in running:
                running[k] += parts[k]
            n_batches += 1

        train_parts = {k: v / max(n_batches, 1) for k, v in running.items()}
        val_acc = evaluate_v2t_accuracy(model, text_z_raw, val_loader, device)
        temperature = 1.0 / model.logit_scale().item()
        print(f"[alignment] epoch {epoch:03d}  train_loss={train_parts['total']:.4f} "
              f"(v2t={train_parts['loss_v2t']:.4f} t2v={train_parts['loss_t2v']:.4f})  "
              f"val_v2t_acc={val_acc:.4f}  temperature={temperature:.4f}")

        if early_stopping.step(val_acc):
            save_checkpoint(model, optimizer, epoch, {"val_v2t_accuracy": val_acc}, checkpoint_path)
        if early_stopping.should_stop:
            print(f"[alignment] early stopping at epoch {epoch} (best val_v2t_acc={early_stopping.best_score:.4f})")
            break


if __name__ == "__main__":
    main()
