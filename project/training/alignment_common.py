"""Shared helpers for Stage B training/evaluation scripts."""
import torch

from project.config import resolve_path
from project.models.encoders import FrozenTextMoodEncoder


def load_text_anchors(cfg: dict, device):
    """Returns (text_z_raw, id2label): text_z_raw is a (5,768) tensor of frozen
    penultimate embeddings, row i = class id i's anchor text (in the same
    0..4 order as cfg['emotion_to_id']). Asserts the text model's own label
    ids line up with the video side's, since Stage B relies on that ordering
    matching without a remapping table."""
    model_path = resolve_path(cfg, "paths.text_model_path")
    encoder = FrozenTextMoodEncoder(str(model_path), device=device)
    id2label = encoder.id2label

    n_classes = len(cfg["emotion_to_id"])
    assert set(id2label.keys()) == set(range(n_classes)), (
        f"text model id2label keys {sorted(id2label.keys())} do not match expected class ids "
        f"0..{n_classes - 1} -- Stage B assumes identical label ordering between video and text."
    )

    labels_sorted = [id2label[i] for i in range(n_classes)]
    with torch.no_grad():
        text_z_raw = encoder(labels_sorted)  # (n_classes, 768)
    return text_z_raw, id2label
