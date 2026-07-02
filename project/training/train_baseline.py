"""Phase 5 (baseline) — train an MLP on the shipped SlowFast (N,2304)
features. Reference number to compare the fusion classifier against.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.feature_dataset import CachedEmbeddingDataset  # noqa: E402
from project.models.baseline_mlp import SlowFastBaselineMLP  # noqa: E402
from project.models.encoders import get_device  # noqa: E402
from project.training.loop import train_classifier  # noqa: E402


def main():
    cfg = load_config()
    device = get_device(cfg["encoders"]["device_preference"])
    print(f"Device: {device}")

    slowfast_dir = resolve_path(cfg, "paths.features_dir") / "slowfast"
    # SlowFast features are keyed by parent_video_id: extracted once per original
    # clip, shared by all of that clip's windows (see extract_slowfast_baseline.py).
    train_ds = CachedEmbeddingDataset(resolve_path(cfg, "paths.train_csv"), [(slowfast_dir, "parent_video_id")])
    val_ds = CachedEmbeddingDataset(resolve_path(cfg, "paths.val_csv"), [(slowfast_dir, "parent_video_id")])
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    model = SlowFastBaselineMLP(
        input_dim=2304,
        hidden_dim=cfg["training"]["baseline_hidden_dim"],
        num_classes=len(cfg["emotion_to_id"]),
        dropout=cfg["training"]["baseline_dropout"],
    )

    checkpoint_path = resolve_path(cfg, "paths.checkpoints_dir") / "baseline_slowfast_mlp_best.pt"
    train_classifier(model, train_ds, val_ds, cfg, checkpoint_path, device, model_name="baseline-slowfast")


if __name__ == "__main__":
    main()
