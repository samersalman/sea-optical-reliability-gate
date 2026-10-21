"""Compare the training augmentation (affine_grid + grid_sample) on MPS vs CPU, and against identity."""
import sys, importlib.util
from pathlib import Path
import numpy as np, torch
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
    gen = torch.Generator().manual_seed(0)
    theta = tr.sample_theta(32, gen)
    ident = torch.tensor([[1., 0., 0.], [0., 1., 0.]]).unsqueeze(0).repeat(32, 1, 1)
    for label, th in (("identity", ident), ("random", theta)):
        xa, ya = tr.apply_affine(xc, yc, th)
        xb, yb = tr.apply_affine(xm, ym, th.to("mps"))
        xb, yb = xb.cpu(), yb.cpu()
        print(f"{label:8s} cpu-vs-mps image max|diff| {float((xa - xb).abs().max()):.4f} mean|diff| {float((xa - xb).abs().mean()):.5f} "
              f"mask mismatch frac {float((ya != yb).float().mean()):.5f}")
        print(f"{label:8s} image mean in {float(xc.mean()):.4f} out(cpu) {float(xa.mean()):.4f} out(mps) {float(xb.mean()):.4f}; "
              f"per-channel std in {xc.std((0,2,3)).numpy().round(4)} out(cpu) {xa.std((0,2,3)).numpy().round(4)} out(mps) {xb.std((0,2,3)).numpy().round(4)}")
        print(f"{label:8s} mask fg frac in {float(yc.mean()):.4f} out(cpu) {float(ya.mean()):.4f} out(mps) {float(yb.mean()):.4f}")
    # identity round trip should reproduce the input exactly
    xa, _ = tr.apply_affine(xc, yc, ident)
    print("identity cpu max|diff| vs input", float((xa - xc).abs().max()))
    print("theta sample:", theta[0].numpy().round(3).tolist())
    print("theta scale stats (|det|):", float(theta[:, :, :2].det().abs().min()), float(theta[:, :, :2].det().abs().max()))


if __name__ == "__main__":
    main()
