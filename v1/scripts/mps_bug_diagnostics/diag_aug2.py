import sys, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1 / "python"))
import dataset
spec = importlib.util.spec_from_file_location("tr", V1 / "python/04_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)


def main():
    t = dataset.frame_table("center1_labeled").sample(32, random_state=1).reset_index(drop=True)
    imgs, msks = dataset.build_crops(t, workers=8, log=lambda *a: None)
    idx = np.arange(32)
    xc, yc = tr.to_batch(imgs, msks, idx, "cpu")
    xm, ym = tr.to_batch(imgs, msks, idx, "mps")
    print("to_batch strides (mps):", xm.stride(), "contiguous:", xm.is_contiguous())
    theta = tr.sample_theta(32, torch.Generator().manual_seed(0))
    ref, _ = tr.apply_affine(xc, yc, theta)
    for label, xin in (("as-is (non-contiguous)", xm), ("contiguous()", xm.contiguous())):
        grid = F.affine_grid(theta.to("mps"), list(xin.shape), align_corners=False)
        out = F.grid_sample(xin, grid.contiguous(), mode="bilinear", padding_mode="zeros", align_corners=False).cpu()
        print(f"{label:24s} mean {float(out.mean()):.4f} (cpu ref {float(ref.mean()):.4f})  max|diff| {float((out - ref).abs().max()):.5f}")
    grid = F.affine_grid(theta.to("mps"), list(xm.shape), align_corners=False)
    print("grid contiguous:", grid.is_contiguous())
    out = F.grid_sample(xm.contiguous(), grid, mode="bilinear", padding_mode="zeros", align_corners=False).cpu()
    print(f"contiguous x, raw grid   mean {float(out.mean()):.4f} max|diff| {float((out - ref).abs().max()):.5f}")


if __name__ == "__main__":
    main()
