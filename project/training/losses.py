"""Stage B — symmetric label-anchored contrastive loss.

Standard CLIP/InfoNCE assumes instance-level (image, caption) pairs. Here the
text side is exactly 5 fixed class-name anchors (see project/docs/cross_modal_alignment_plan.pdf,
"the core feasibility question: what counts as a pair?") rather than diverse
captions, so the loss reduces to a clean, honest special case:

  video -> text direction: each video has exactly one correct text anchor, so
  this is a plain softmax cross-entropy over the 5 anchors (mathematically an
  InfoNCE loss with the 4 other anchors as in-batch negatives).

  text -> video direction: many videos in a batch share a given anchor's
  class, so this is the multi-positive supervised-contrastive generalization
  (Khosla et al., 2020) -- for each class present in the batch, average the
  log-probability of every video sharing that class being the nearest match
  to that anchor.
"""
import torch
import torch.nn.functional as F


def label_anchored_contrastive_loss(
    video_z: torch.Tensor, text_z: torch.Tensor, labels: torch.Tensor, logit_scale: torch.Tensor
):
    """video_z: (B,D) L2-normalized. text_z: (C,D) L2-normalized, row c = class c's anchor.
    labels: (B,) int64 in [0, C). Returns (total_loss, {component: float} for logging)."""
    logits = logit_scale * video_z @ text_z.t()  # (B, C)

    loss_v2t = F.cross_entropy(logits, labels)

    logits_t2v = logits.t()  # (C, B)
    per_class_losses = []
    for c in range(text_z.shape[0]):
        positive_mask = labels == c
        if not positive_mask.any():
            continue
        log_probs = F.log_softmax(logits_t2v[c], dim=0)
        per_class_losses.append(-log_probs[positive_mask].mean())
    loss_t2v = torch.stack(per_class_losses).mean() if per_class_losses else torch.zeros((), device=video_z.device)

    total = 0.5 * (loss_v2t + loss_t2v)
    return total, {"loss_v2t": loss_v2t.item(), "loss_t2v": loss_t2v.item(), "total": total.item()}
