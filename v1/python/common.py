"""Shared helpers: config, paths, image and mask readers, seeding."""
import random
from pathlib import Path

import cv2
import nrrd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "v1" / "config" / "analysis_plan.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def abspath(rel):
    """Manifest paths are stored relative to ROOT."""
    return ROOT / rel


def read_image(path):
    """RGB uint8, H x W x 3. EXIF rotation is ignored so the array matches the header size and the masks."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path):
    """Boolean H x W instrument mask, the union of all segments (value > 0).

    Slicer stores each NRRD as (W, H, 1) with identity space directions: axis 0 = image x, axis 1 = image y.
    No flip is needed, only a transpose (the dataset's own data_format_convert.py does `np.squeeze(data).T`).
    Checked three ways: the transposed shape equals the paired image's (H, W) for all 10,662 frames; the header
    Segment extents enclose the foreground bounding box on axis 0/1 for all of them (manifest audit); and on 180
    frames (60 per domain) the identity orientation had 2-3x the image gradient along the mask boundary of
    flipud/fliplr/rot180 and won 156 of 180 head-to-heads, with contours also checked visually.
    Values are {0,1} or {0,255}; 56 frames carry two segments (values {0,1,2}), hence the union.
    """
    data, _ = nrrd.read(str(path))
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    if data.ndim != 2:
        raise ValueError(f"unexpected mask shape {data.shape}: {path}")
    return np.ascontiguousarray(data.T) > 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
