"""Dual conditioning helpers — combine MusicGen's own frozen T5 text encoder
with the Stage C bridge's mood-steering sequence, per
`project/docs/text_conditioning_bridge_plan.pdf`.

The current (Milestone 1) bridge replaces MusicGen's T5 conditioning outright
(`encoder_outputs=(bridge(z),)`), which is sufficient signal for *which* of 5
mood classes to generate toward but discards everything else in the input
text (instrumentation, tempo, genre, descriptive detail). This module adds
the pieces needed to instead *concatenate* T5's real semantic tokens with the
bridge's mood tokens along the time axis, verified against the installed
`transformers` MusicGen source (`modeling_musicgen.py`,
`MusicgenForConditionalGeneration.forward`): `encoder_outputs` supplied as a
tuple is used as-is regardless of sequence length or source, and
`attention_mask` both zeroes masked positions of the combined sequence and is
passed straight through to the decoder's cross-attention -- a concatenation
of two conditioning sequences with a concatenated mask is a first-class
supported input, not a workaround.
"""
from typing import Dict, List

import torch
from transformers import AutoTokenizer, MusicgenForConditionalGeneration


def build_musicgen_tokenizer(model_name: str) -> AutoTokenizer:
    """Loads the T5 tokenizer shipped alongside the given MusicGen checkpoint --
    the same tokenizer stock MusicGen uses internally when `encoder_outputs`
    is not supplied and it falls back to calling `self.text_encoder` itself."""
    return AutoTokenizer.from_pretrained(model_name)


@torch.no_grad()
def t5_encode(
    musicgen: MusicgenForConditionalGeneration,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 64,
):
    """Runs raw text through MusicGen's own frozen T5 text encoder, exactly as
    stock MusicGen does internally for a text prompt. Returns
    (hidden (B, T_text, 768), mask (B, T_text)), both on `device`."""
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True
    ).to(device)
    hidden = musicgen.text_encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    return hidden, enc["attention_mask"]


def concat_conditioning(
    t5_hidden: torch.Tensor,
    t5_mask: torch.Tensor,
    bridge_out: torch.Tensor,
    bridge_mask: torch.Tensor,
):
    """Concatenates the T5 semantic sequence and the bridge's mood sequence
    along the time axis (Section 2.2 of the plan): cond = [T5(text); bridge(z)],
    mask = [mask_T5; mask_bridge]. Both inputs must already share batch size
    and hidden/cond dim (768 for musicgen-small's T5)."""
    cond = torch.cat([t5_hidden, bridge_out], dim=1)
    mask = torch.cat([t5_mask, bridge_mask], dim=1)
    return cond, mask


def label_template_text(labels: List[int], id2label: Dict[int, str], template: str) -> List[str]:
    """Builds the video-anchored raw-text branch from integer class labels,
    since EmoMV videos have no per-clip captions (see plan's "Open Assumptions
    to Confirm"): e.g. template "{emotion} instrumental music" + label 4
    ("relax") -> "relax instrumental music"."""
    return [template.format(emotion=id2label[int(label)]) for label in labels]
