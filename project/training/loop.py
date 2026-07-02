"""Shared supervised training loop for Phase 5 classifiers (baseline SlowFast
MLP and VideoMAE+CLIP fusion classifier) -- both only differ in model/input,
not in the optimization loop, so the loop is written once here.
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from project.training.utils import EarlyStopping, compute_class_weights, save_checkpoint, set_seed


def make_train_loader(dataset, cfg: dict) -> DataLoader:
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    if cfg["training"]["class_imbalance_strategy"] == "weighted_sampler":
        labels = dataset.df["emotion_id"].values
        class_weights = compute_class_weights(labels, len(cfg["emotion_to_id"])).numpy()
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, criterion=None) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for x, y, _video_ids in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)
            n += len(y)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def train_classifier(model, train_ds, val_ds, cfg: dict, checkpoint_path, device, model_name: str = "model") -> list:
    set_seed(cfg["seed"])
    model = model.to(device)

    train_loader = make_train_loader(train_ds, cfg)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"]
    )

    if cfg["training"]["class_imbalance_strategy"] == "class_weights":
        weight = compute_class_weights(train_ds.df["emotion_id"].values, len(cfg["emotion_to_id"])).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    early_stopping = EarlyStopping(patience=cfg["training"]["early_stopping_patience"], mode="max")

    history = []
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        model.train()
        running_loss, n = 0.0, 0
        for x, y, _video_ids in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            n += len(y)
        train_loss = running_loss / max(n, 1)

        val_metrics = evaluate(model, val_loader, device, criterion)
        print(f"[{model_name}] epoch {epoch:03d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if early_stopping.step(val_metrics["macro_f1"]):
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
        if early_stopping.should_stop:
            print(f"[{model_name}] early stopping at epoch {epoch} "
                  f"(best val_macro_f1={early_stopping.best_score:.4f})")
            break

    return history
