"""Instrument segmenters: U-Net (ResNet34) and SegFormer-B0, both class-agnostic with one logit channel.

Both take RGB float images in [0, 1], shape (B, 3, H, W), and return logits (B, 1, H, W). ImageNet
normalisation lives inside the wrapper, so callers pass uint8 / 255 and nothing else. Weights come from the
local HF cache (HF_HUB_OFFLINE is set so a missing network never stalls a run).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import load_config  # noqa: E402

MODEL_NAMES = ("unet", "segformer")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Segmenter(nn.Module):
    """Normalise, run the backbone, return logits at the input resolution."""

    def __init__(self, net, kind):
        super().__init__()
        self.net, self.kind = net, kind
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        z = (x - self.mean) / self.std
        if self.kind == "segformer":
            logits = self.net(pixel_values=z).logits      # quarter resolution
            return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(z)


def build_model(name, pretrained=True):
    """pretrained=False builds the same architecture with random weights (tests, loading a trained state_dict)."""
    spec = load_config()["segmenters"]["models"]
    if name == "unet":
        import segmentation_models_pytorch as smp
        s = spec["unet"]
        net = smp.Unet(s["encoder"], encoder_weights=s["weights"] if pretrained else None, in_channels=3, classes=1)
    elif name == "segformer":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        from transformers.utils import logging as hf_logging
        s = spec["segformer"]
        hf_logging.set_verbosity_error()                  # the new decode head is meant to be missing from the checkpoint
        hf_logging.disable_progress_bar()
        if pretrained:
            net = SegformerForSemanticSegmentation.from_pretrained(s["checkpoint"], num_labels=s["num_labels"])
        else:
            net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(s["checkpoint"], num_labels=s["num_labels"]))
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    if name == "segformer":
        # On MPS the 1x1 fuse conv hands the decode-head BatchNorm a channels_last tensor and its backward then
        # fails with a view error. A contiguous input avoids it; the values are unchanged.
        net.decode_head.batch_norm.register_forward_pre_hook(lambda mod, args: (args[0].contiguous(),))
    return Segmenter(net, name)
