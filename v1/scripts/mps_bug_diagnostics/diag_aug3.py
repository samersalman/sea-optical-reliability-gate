import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1 / "python"))
import dataset


def main():
    t = dataset.frame_table("center1_labeled").sample(16, random_state=1).reset_index(drop=True)
    imgs, _ = dataset.build_crops(t, workers=8, log=lambda *a: None)
    u = torch.from_numpy(imgs)                                   # uint8 N,H,W,3
    ref = u.permute(0, 3, 1, 2).float().div_(255)                # cpu, channels_last strides
    ident = torch.tensor([[1., 0., 0.], [0., 1., 0.]]).unsqueeze(0).repeat(16, 1, 1)
    grid = F.affine_grid(ident, list(ref.shape), align_corners=False)
    cpu_out = F.grid_sample(ref, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    print("cpu identity max|out-in|", float((cpu_out - ref).abs().max()), "mean in", float(ref.mean()))
    variants = {
        "mps NCHW-contiguous float": ref.contiguous().to("mps"),
        "mps channels_last (permute)": u.to("mps").permute(0, 3, 1, 2).float().div_(255),
        "mps channels_last .contiguous()": u.to("mps").permute(0, 3, 1, 2).float().div_(255).contiguous(),
        "mps uint8 NCHW-contiguous -> float": u.permute(0, 3, 1, 2).contiguous().to("mps").float().div_(255),
    }
    g = grid.to("mps")
    for k, v in variants.items():
        o = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False).cpu()
        print(f"{k:38s} strides {tuple(v.stride())} in-mean {float(v.cpu().mean()):.4f} out-mean {float(o.mean()):.4f} max|out-ref| {float((o - ref).abs().max()):.4f}")


if __name__ == "__main__":
    main()
