"""Stage C, Milestone 1 (+ dual conditioning) — train the bridge adapter on
(video embedding, real audio) pairs from EmoMV MATCH clips.

Everything except the bridge is frozen: the Stage B AlignmentModel (already
trained), and all of MusicGen (text encoder, audio encoder/EnCodec, LM
decoder). Training uses MusicGen's own teacher-forced next-token loss over
EnCodec audio codes -- the same objective MusicGen's text-conditioning
pathway was originally trained with; only the conditioning *source* changes,
from "T5 embedding of a caption" to "bridge-projected shared-space embedding".

API note (confirmed by direct inspection of the installed transformers
source, not assumed): `MusicgenForConditionalGeneration.forward` accepts
`encoder_outputs=(tensor,)` to bypass its internal T5 call entirely, and
`labels` (pre-computed EnCodec codes, shape (B*num_codebooks, seq_len)) to
compute the LM loss via teacher forcing.

Dual conditioning (`stage_c.dual_conditioning`, see
project/docs/text_conditioning_bridge_plan.pdf): instead of the bridge output
alone replacing T5's conditioning, MusicGen's own frozen T5 encoder is run on
raw text (a class-name template, since EmoMV clips have no per-clip captions)
and concatenated with the bridge's mood sequence along the time axis. Only
the bridge is trained either way -- T5 stays frozen and is run under
`torch.no_grad()`, so this adds no optimizer state versus Milestone 1.
"""
import sys
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import MusicgenForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import load_config, resolve_path  # noqa: E402
from project.datasets.feature_dataset import CachedEmbeddingDataset  # noqa: E402
from project.models.adapters import AlignmentModel  # noqa: E402
from project.models.bridge import EmbeddingToConditioningBridge  # noqa: E402
from project.models.conditioning import build_musicgen_tokenizer, concat_conditioning, \
    label_template_text, t5_encode  # noqa: E402
from project.models.encoders import get_device  # noqa: E402
from project.training.utils import EarlyStopping, save_checkpoint, set_seed  # noqa: E402


class VideoEmbeddingAudioDataset(Dataset):
    """Pairs a clip's 512-d video embedding with its own extracted audio
    track. Both come from the clip's `video_id` -- no extra pairing needed,
    since a MATCH clip's video and audio are already the same file.

    Also carries each row's raw text for the dual-conditioning T5 branch:
    since EmoMV videos have no per-clip captions, this is a fixed template
    built from the clip's own emotion label (see `label_template_text`)."""

    def __init__(self, csv_path, video_dir: Path, audio_dir: Path, sample_rate: int, channels: int,
                 clip_duration_s: float, id2label: dict = None, text_template: str = None,
                 subset_size: int = None, seed: int = 42):
        self.base = CachedEmbeddingDataset(csv_path, [(video_dir, "video_id")])
        has_audio = self.base.df["video_id"].apply(lambda vid: (audio_dir / f"{vid}.wav").exists())
        n_missing = int((~has_audio).sum())
        if n_missing:
            print(f"  [WARN] {n_missing}/{len(self.base.df)} rows missing extracted audio, dropping "
                  f"(run extract_audio_targets.py first)")
        self.base.df = self.base.df[has_audio].reset_index(drop=True)

        if subset_size is not None and subset_size < len(self.base.df):
            self.base.df = self.base.df.sample(n=subset_size, random_state=seed).reset_index(drop=True)

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.n_samples = int(clip_duration_s * sample_rate)
        self.id2label = id2label
        self.text_template = text_template

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        video_embedding, label, video_id = self.base[i]
        waveform, sr = torchaudio.load(str(self.audio_dir / f"{video_id}.wav"))  # (channels, n)
        assert sr == self.sample_rate, f"{video_id}.wav sample rate {sr} != expected {self.sample_rate}"
        if waveform.shape[0] != self.channels:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] < self.n_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, : self.n_samples]

        if self.id2label is not None:
            text = label_template_text([label], self.id2label, self.text_template)[0]
            return video_embedding, waveform, label, video_id, text
        return video_embedding, waveform, label, video_id


def audio_to_labels(musicgen: MusicgenForConditionalGeneration, waveform: torch.Tensor) -> torch.Tensor:
    """waveform: (B, channels, n_samples) -> EnCodec codes as the (B, seq_len,
    num_codebooks) shape MusicgenForConditionalGeneration.forward expects for
    `labels` (per its docstring; codebooks last, distinct from the
    (B*num_codebooks, seq_len) shape used for `input_ids`)."""
    with torch.no_grad():
        audio_codes = musicgen.audio_encoder.encode(input_values=waveform).audio_codes  # (frames,B,codebooks,T)
    frames, bsz, codebooks, seq_len = audio_codes.shape
    assert frames == 1, f"expected 1 EnCodec frame (disable chunking), got {frames}"
    return audio_codes[0].transpose(1, 2)  # (B, seq_len, codebooks)


def main():
    cfg = load_config()
    sc_cfg = cfg["stage_c"]
    a_cfg = cfg["alignment"]
    set_seed(cfg["seed"])
    device = get_device(sc_cfg["device_preference"])
    print(f"Device: {device}")

    print(f"Loading {sc_cfg['musicgen_model']} (frozen)...")
    musicgen = MusicgenForConditionalGeneration.from_pretrained(sc_cfg["musicgen_model"]).to(device).eval()
    if musicgen.config.decoder.decoder_start_token_id is None:
        # Known config-loading quirk: generation_config carries the correct value
        # (== pad_token_id, the reserved EnCodec token beyond the 2048-code vocab)
        # but config.decoder doesn't; shift_tokens_right needs it set on config.decoder.
        musicgen.config.decoder.decoder_start_token_id = musicgen.generation_config.decoder_start_token_id
    for p in musicgen.parameters():
        p.requires_grad = False
    cond_dim = musicgen.config.text_encoder.d_model
    print(f"MusicGen conditioning dim (T5 hidden size): {cond_dim}")

    alignment_model = AlignmentModel(
        shared_dim=a_cfg["shared_dim"], dropout=a_cfg["adapter_dropout"],
        logit_scale_init=a_cfg["logit_scale_init"], logit_scale_max=a_cfg["logit_scale_max"],
    ).to(device)
    from project.training.utils import load_checkpoint
    load_checkpoint(alignment_model, resolve_path(cfg, "paths.checkpoints_dir") / "alignment_adapters_best.pt",
                     map_location=device)
    alignment_model.eval()
    for p in alignment_model.parameters():
        p.requires_grad = False

    bridge = EmbeddingToConditioningBridge(
        shared_dim=a_cfg["shared_dim"], seq_len=sc_cfg["bridge_seq_len"], cond_dim=cond_dim,
        hidden_dim=sc_cfg["bridge_hidden_dim"], n_heads=sc_cfg["bridge_n_heads"],
        n_layers=sc_cfg["bridge_n_layers"], dropout=sc_cfg["bridge_dropout"],
    ).to(device)

    dual_conditioning = sc_cfg.get("dual_conditioning", False)
    tokenizer = None
    if dual_conditioning:
        print("Dual conditioning enabled: concatenating MusicGen's frozen T5(text) with bridge(z).")
        tokenizer = build_musicgen_tokenizer(sc_cfg["musicgen_model"])
    id2label = {v: k for k, v in cfg["emotion_to_id"].items()} if dual_conditioning else None
    text_template = sc_cfg.get("text_template")

    video_dir = resolve_path(cfg, "paths.video_embedding_512_dir")
    audio_dir = resolve_path(cfg, "paths.audio_targets_dir")
    train_ds = VideoEmbeddingAudioDataset(
        resolve_path(cfg, "paths.train_csv"), video_dir, audio_dir,
        sc_cfg["audio_sample_rate"], sc_cfg["audio_channels"], sc_cfg["clip_duration_s"],
        id2label=id2label, text_template=text_template,
        subset_size=sc_cfg["subset_size"], seed=cfg["seed"],
    )
    val_ds = VideoEmbeddingAudioDataset(
        resolve_path(cfg, "paths.val_csv"), video_dir, audio_dir,
        sc_cfg["audio_sample_rate"], sc_cfg["audio_channels"], sc_cfg["clip_duration_s"],
        id2label=id2label, text_template=text_template,
        subset_size=max(sc_cfg["subset_size"] // 5, 10), seed=cfg["seed"],
    )
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=sc_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=sc_cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=sc_cfg["lr"], weight_decay=sc_cfg["weight_decay"])
    early_stopping = EarlyStopping(patience=sc_cfg["early_stopping_patience"], mode="min")
    checkpoint_name = sc_cfg["bridge_checkpoint_name"] if dual_conditioning else "bridge_best.pt"
    checkpoint_path = resolve_path(cfg, "paths.checkpoints_dir") / checkpoint_name

    start_epoch = 1
    if sc_cfg.get("resume", False) and checkpoint_path.exists():
        ckpt = load_checkpoint(bridge, checkpoint_path, optimizer=optimizer, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        early_stopping.best_score = ckpt["metrics"]["val_loss"]
        print(f"Resumed from {checkpoint_path} (epoch {ckpt['epoch']}, "
              f"best val_loss={early_stopping.best_score:.4f}) -- continuing at epoch {start_epoch}")

    def run_epoch(loader, train: bool):
        bridge.train(train)
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            if dual_conditioning:
                video_emb, waveform, _labels, _video_ids, texts = batch
            else:
                video_emb, waveform, _labels, _video_ids = batch
            video_emb, waveform = video_emb.to(device), waveform.to(device)
            with torch.no_grad():
                shared_z = alignment_model.encode_video(video_emb)  # (B, shared_dim), frozen
            bridge_out = bridge(shared_z)  # (B, T_bridge, cond_dim) -- only this has gradients
            bridge_mask = bridge.attention_mask(bridge_out.shape[0], device)

            if dual_conditioning:
                t5_hidden, t5_mask = t5_encode(musicgen, tokenizer, texts, device, sc_cfg["text_max_length"])
                cond, attn_mask = concat_conditioning(t5_hidden, t5_mask, bridge_out, bridge_mask)
            else:
                cond, attn_mask = bridge_out, bridge_mask

            codec_labels = audio_to_labels(musicgen, waveform)

            out = musicgen(encoder_outputs=(cond,), attention_mask=attn_mask, labels=codec_labels)
            loss = out.loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    for epoch in range(start_epoch, sc_cfg["max_epochs"] + 1):
        train_loss = run_epoch(train_loader, train=True)
        with torch.no_grad():
            val_loss = run_epoch(val_loader, train=False)
        print(f"[bridge] epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if early_stopping.step(val_loss):
            save_checkpoint(bridge, optimizer, epoch, {"val_loss": val_loss}, checkpoint_path)
        if early_stopping.should_stop:
            print(f"[bridge] early stopping at epoch {epoch} (best val_loss={early_stopping.best_score:.4f})")
            break


if __name__ == "__main__":
    main()
