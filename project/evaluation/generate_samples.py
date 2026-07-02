"""Stage C, Milestone 1 — generate a handful of .wav samples per emotion
class for manual listening validation.

Two sources, on purpose:
  1. TEXT anchors (the 5 fixed class-name strings) -- out-of-distribution for
     the bridge, since it only ever trained on video embeddings. Tests
     whether Stage B's cross-modal alignment actually transfers to steering
     MusicGen from text alone.
  2. Real VIDEO embeddings from held-out test clips -- in-distribution, what
     the bridge was actually trained on.

guidance_scale=1.0 is set explicitly: MusicGen's default classifier-free
guidance path duplicates the batch against a null/unconditional embedding
internally when conditioning via raw input_ids, which doesn't apply when we
hand it custom encoder_outputs directly -- disabling it avoids a shape
mismatch rather than silently producing wrong output.
"""
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import MusicgenForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.feature_dataset import CachedEmbeddingDataset  # noqa: E402
from project.models.adapters import AlignmentModel  # noqa: E402
from project.models.bridge import EmbeddingToConditioningBridge  # noqa: E402
from project.models.encoders import get_device  # noqa: E402
from project.training.alignment_common import load_text_anchors  # noqa: E402
from project.training.utils import load_checkpoint  # noqa: E402


@torch.no_grad()
def generate_from_conditioning(musicgen, cond, attn_mask, max_new_tokens):
    return musicgen.generate(
        encoder_outputs=(cond,), attention_mask=attn_mask,
        max_new_tokens=max_new_tokens, do_sample=True, guidance_scale=1.0,
    )


def main():
    cfg = load_config()
    sc_cfg = cfg["stage_c"]
    a_cfg = cfg["alignment"]
    device = get_device(sc_cfg["device_preference"])
    print(f"Device: {device}")

    musicgen = MusicgenForConditionalGeneration.from_pretrained(sc_cfg["musicgen_model"]).to(device).eval()
    cond_dim = musicgen.config.text_encoder.d_model

    alignment_model = AlignmentModel(
        shared_dim=a_cfg["shared_dim"], dropout=a_cfg["adapter_dropout"],
        logit_scale_init=a_cfg["logit_scale_init"], logit_scale_max=a_cfg["logit_scale_max"],
    ).to(device)
    load_checkpoint(alignment_model, resolve_path(cfg, "paths.checkpoints_dir") / "alignment_adapters_best.pt",
                     map_location=device)
    alignment_model.eval()

    bridge = EmbeddingToConditioningBridge(
        shared_dim=a_cfg["shared_dim"], seq_len=sc_cfg["bridge_seq_len"], cond_dim=cond_dim,
        hidden_dim=sc_cfg["bridge_hidden_dim"], n_heads=sc_cfg["bridge_n_heads"],
        n_layers=sc_cfg["bridge_n_layers"], dropout=sc_cfg["bridge_dropout"],
    ).to(device)
    load_checkpoint(bridge, resolve_path(cfg, "paths.checkpoints_dir") / "bridge_best.pt", map_location=device)
    bridge.eval()

    out_dir = resolve_path(cfg, "paths.results_dir") / "generated_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = sc_cfg["audio_sample_rate"]
    max_new_tokens = int(sc_cfg["clip_duration_s"] * 50)  # EnCodec's ~50 Hz frame rate at this config

    print("\n--- Generating from TEXT anchors (out-of-distribution for the bridge) ---")
    text_z_raw, id2label = load_text_anchors(cfg, device)
    with torch.no_grad():
        text_z = alignment_model.encode_text(text_z_raw.to(device))
        cond = bridge(text_z)
        attn_mask = bridge.attention_mask(cond.shape[0], device)
        audio = generate_from_conditioning(musicgen, cond, attn_mask, max_new_tokens)
    for i, label in id2label.items():
        path = out_dir / f"text_{label}.wav"
        torchaudio.save(str(path), audio[i].cpu(), sample_rate)
        print(f"  saved {path}")

    print("\n--- Generating from real VIDEO embeddings (in-distribution) ---")
    video_dir = resolve_path(cfg, "paths.video_embedding_512_dir")
    test_ds = CachedEmbeddingDataset(resolve_path(cfg, "paths.test_csv"), [(video_dir, "video_id")])
    picked = {}
    for i in range(len(test_ds)):
        x, label, vid = test_ds[i]
        if label not in picked:
            picked[label] = (x, vid)
        if len(picked) == len(id2label):
            break

    for label_id, (x, vid) in picked.items():
        with torch.no_grad():
            z = alignment_model.encode_video(x.unsqueeze(0).to(device))
            cond = bridge(z)
            attn_mask = bridge.attention_mask(1, device)
            audio = generate_from_conditioning(musicgen, cond, attn_mask, max_new_tokens)
        path = out_dir / f"video_{id2label[label_id]}_{vid}.wav"
        torchaudio.save(str(path), audio[0].cpu(), sample_rate)
        print(f"  saved {path}")

    print(f"\nAll samples -> {out_dir}  (listen manually to assess mood-consistency)")


if __name__ == "__main__":
    main()
