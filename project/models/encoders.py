"""Phase 4 — frozen VideoMAE + CLIP encoder wrappers.

Both wrap a pretrained HuggingFace model in eval() mode with
requires_grad=False; they take the raw [0,1] (B,T,C,H,W) clips produced by
`datasets.video_dataset.EmoMVVideoDataset` and apply their own
encoder-specific mean/std normalization internally, so the same decoded clip
feeds both encoders without re-decoding or re-normalizing upstream.
"""
from typing import List

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, CLIPModel, CLIPProcessor, \
    VideoMAEImageProcessor, VideoMAEModel


def get_device(preference=("mps", "cuda", "cpu")) -> torch.device:
    for p in preference:
        if p == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


class FrozenVideoMAEEncoder(nn.Module):
    """(B,T,C,H,W) in [0,1] -> (B,768) mean-pooled VideoMAE embedding."""

    EMBED_DIM = 768

    def __init__(self, model_name: str = "MCG-NJU/videomae-base", device: torch.device = None):
        super().__init__()
        self.device = device or get_device()
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        clips = clips.to(self.device)
        normed = (clips - self.mean.to(self.device)) / self.std.to(self.device)
        out = self.model(pixel_values=normed)
        return out.last_hidden_state.mean(dim=1).cpu()  # (B, 768)


class FrozenCLIPEncoder(nn.Module):
    """(B,T,C,H,W) in [0,1] -> (B,512) mean-pooled-over-frames CLIP image embedding."""

    EMBED_DIM = 512

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: torch.device = None):
        super().__init__()
        self.device = device or get_device()
        processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std
        self.register_buffer("mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(image_std).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clips.shape
        frames = clips.view(b * t, c, h, w).to(self.device)
        normed = (frames - self.mean.to(self.device)) / self.std.to(self.device)
        # return_dict=False -> (last_hidden_state, image_embeds); we want the pooled embed_dim-512 projection.
        _, feats = self.model.get_image_features(pixel_values=normed, return_dict=False)  # (B*T, 512)
        return feats.view(b, t, -1).mean(dim=1).cpu()  # (B, 512)


class FrozenTextMoodEncoder(nn.Module):
    """Wraps the fine-tuned `text_mood_clf` (distilroberta) 5-class mood classifier
    and exposes its 768-d *penultimate* representation (post dense+tanh, pre
    out_proj) -- the task-supervised text embedding Stage B projects from, symmetric
    to how FusionClassifier's penultimate 512-d layer is used on the video side.

    Verified by direct reconstruction against the model's own logits: manually
    replaying roberta -> classifier.dropout -> classifier.dense -> tanh ->
    classifier.dropout -> classifier.out_proj reproduces `model(**enc).logits`
    exactly (see project/docs -- this class hardcodes that exact sequence).
    """

    EMBED_DIM = 768

    def __init__(self, model_path: str, device: torch.device = None, max_length: int = 64):
        super().__init__()
        self.device = device or get_device()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True
        ).to(self.device)
        hidden = self.model.roberta(**enc).last_hidden_state[:, 0, :]  # <s> token
        head = self.model.classifier
        x = head.dropout(hidden)
        x = torch.tanh(head.dense(x))
        x = head.dropout(x)
        return x.cpu()  # (len(texts), 768)
