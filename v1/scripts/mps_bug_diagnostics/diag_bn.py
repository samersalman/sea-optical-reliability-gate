"""Does recomputing BN running statistics ("precise BN") fix the eval-mode collapse? Uses the pilot checkpoint."""
import sys, importlib.util, copy
from pathlib import Path
import numpy as np, torch
V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1 / "python"))
import dataset, models, fov
spec = importlib.util.spec_from_file_location("tr", V1 / "python/04_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
CACHE = {}


def iou_of(m, imgs, msks, dev, disk):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 32):
            x, y = tr.to_batch(imgs, msks, np.arange(i, min(i + 32, len(imgs))), dev)
            out.append(tr.batch_metrics(m(x), y, disk)[0].cpu())
    v = torch.cat(out)
    return f"mean IoU {v.mean():.3f} median {v.median():.3f} frac<0.5 {(v<0.5).float().mean():.2f}"


def recal(m, imgs, msks, dev, augment, n_batches=60, seed=0):
    m = copy.deepcopy(m).to(dev).train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats(); mod.momentum = None
    gen = torch.Generator().manual_seed(seed)
    order = np.random.default_rng(seed).permutation(len(imgs))
    with torch.no_grad():
        for s in range(min(n_batches, len(order) // 16)):
            x, y = tr.to_batch(imgs, msks, order[s * 16:(s + 1) * 16], dev)
            if augment:
                x, y = tr.apply_affine(x, y, tr.sample_theta(len(x), gen).to(dev))
            m(x)
    return m


def main():
    dev = "mps"
    parts = dataset.patients_for(0, pilot=True)
    t = dataset.frame_table("center1_labeled")
    rng = np.random.default_rng(0)
    def pick(pats, n):
        sub = t[t.patient_id.isin(pats)]
        return sub.iloc[rng.choice(len(sub), min(n, len(sub)), replace=False)].reset_index(drop=True)
    sets = {"train": pick(parts["train"], 960), "val": pick(parts["val"], 320)}
    data = {k: dataset.build_crops(v, workers=8, log=lambda *a: None) for k, v in sets.items()}
    disk = torch.from_numpy(fov.fov_disk_mask(384, fov.ERODE_FRAC)).to(dev)
    ck = torch.load(V1 / "work/models/unet_fold0_pilot_ckpt.pt", map_location="cpu", weights_only=False)
    print("checkpoint epoch", ck["epoch"])
    m = models.build_model("unet", pretrained=False); m.load_state_dict(ck["model"]); m.to(dev)
    for name, (imgs, msks) in data.items():
        print(f"stored stats          {name:5s}", iou_of(m, imgs, msks, dev, disk))
    tr_imgs, tr_msks = data["train"]
    for aug in (False, True):
        m2 = recal(m, tr_imgs, tr_msks, dev, augment=aug)
        for name, (imgs, msks) in data.items():
            print(f"recal aug={aug!s:5s}      {name:5s}", iou_of(m2, imgs, msks, dev, disk))
    # how different are the stored running stats from the recomputed ones?
    m3 = recal(m, tr_imgs, tr_msks, dev, augment=True)
    worst = []
    for (n1, b1), (n2, b2) in zip(m.named_buffers(), m3.named_buffers()):
        if n1.endswith("running_var"):
            ratio = (b1.cpu() / b2.cpu().clamp(min=1e-12))
            worst.append((float(ratio.median()), n1))
    worst.sort()
    print("running_var stored/recomputed median ratio, lowest 3 and highest 3:", worst[:3], worst[-3:])


if __name__ == "__main__":
    main()
