import sys, time, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1 / "python"))
import dataset, models
spec = importlib.util.spec_from_file_location("tr", V1 / "python/04_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)


def main():
    t = dataset.frame_table("center1_labeled").sample(32, random_state=1).reset_index(drop=True)
    imgs, msks = dataset.build_crops(t, workers=8, log=lambda *a: None)
    idx = np.arange(32)
    x = torch.from_numpy(imgs[idx]).permute(0, 3, 1, 2).float().div_(255).contiguous()
    y = torch.from_numpy(msks[idx]).unsqueeze(1).float().contiguous()
    gen = torch.Generator().manual_seed(0)
    theta = tr.sample_theta(32, gen)
    t0 = time.time()
    for _ in range(10):
        xa, ya = tr.apply_affine(x[:16], y[:16], theta[:16])
    print(f"CPU apply_affine batch 16: {(time.time() - t0) / 10 * 1000:.1f} ms; image mean in {float(x[:16].mean()):.4f} out {float(xa.mean()):.4f}")
    xa, ya = xa.contiguous(), ya.contiguous()
    # U-Net forward parity: CPU vs MPS on the same augmented batch, contiguous input
    m = models.build_model("unet", pretrained=False)
    ck = torch.load(V1 / "work/models/unet_fold0_pilot_ckpt.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        lc = m(xa)
        lm = m.to("mps")(xa.to("mps")).cpu()
    print(f"U-Net eval forward CPU vs MPS: max|logit diff| {float((lc - lm).abs().max()):.4f}  mean|diff| {float((lc - lm).abs().mean()):.5f}  logit std {float(lc.std()):.3f}")
    # training-mode forward + loss parity (fresh BN batch stats), contiguous input
    m = models.build_model("unet", pretrained=False); m.load_state_dict(ck["model"]); m.train()
    lc = m(xa); loss_c = tr.seg_loss(lc, ya)
    mm = models.build_model("unet", pretrained=False); mm.load_state_dict(ck["model"]); mm.train().to("mps")
    lmm = mm(xa.to("mps")); loss_m = tr.seg_loss(lmm, ya.to("mps"))
    print(f"train-mode loss CPU {float(loss_c):.4f} vs MPS {float(loss_m):.4f}")


if __name__ == "__main__":
    main()
