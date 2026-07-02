"""Phase 5 (fusion) — train the fusion classifier on cached, frozen
VideoMAE (768-d) + CLIP (512-d) embeddings extracted in Phase 4.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.feature_dataset import CachedEmbeddingDataset  # noqa: E402
from project.models.encoders import FrozenCLIPEncoder, FrozenVideoMAEEncoder, get_device  # noqa: E402
from project.models.fusion_classifier import FusionClassifier  # noqa: E402
from project.training.loop import train_classifier  # noqa: E402


def main():
    cfg = load_config()
    device = get_device(cfg["encoders"]["device_preference"])
    print(f"Device: {device}")

    videomae_dir = resolve_path(cfg, "paths.videomae_features_dir")
    clip_dir = resolve_path(cfg, "paths.clip_features_dir")
    feature_specs = [(videomae_dir, "video_id"), (clip_dir, "video_id")]

    train_ds = CachedEmbeddingDataset(resolve_path(cfg, "paths.train_csv"), feature_specs)
    val_ds = CachedEmbeddingDataset(resolve_path(cfg, "paths.val_csv"), feature_specs)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    model = FusionClassifier(
        videomae_dim=FrozenVideoMAEEncoder.EMBED_DIM,
        clip_dim=FrozenCLIPEncoder.EMBED_DIM,
        hidden_dim=cfg["training"]["fusion_hidden_dim"],
        num_classes=len(cfg["emotion_to_id"]),
        dropout=cfg["training"]["fusion_dropout"],
    )

    checkpoint_path = resolve_path(cfg, "paths.checkpoints_dir") / "fusion_classifier_best.pt"
    train_classifier(model, train_ds, val_ds, cfg, checkpoint_path, device, model_name="fusion-videomae-clip")


if __name__ == "__main__":
    main()
