"""Diagnose the pilot's erratic validation IoU: eval-mode vs batch-statistics BN, train vs inner-val frames, MPS vs CPU."""
import sys, importlib.util
from pathlib import Path
import numpy as np, torch
V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1 / "python"))
import dataset, models, fov
spec = importlib.util.spec_from_file_location("tr", V1 / "python/04_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)

CACHE = {}


def main():
    parts = dataset.patients_for(0, pilot=True)
    t = dataset.frame_table("center1_labeled")
    rng = np.random.default_rng(0)
    def pick(pats, n):
        sub = t[t.patient_id.isin(pats)]
        return sub.iloc[rng.choice(len(sub), min(n, len(sub)), replace=False)].reset_index(drop=True)
    sets = {"train": pick(parts["train"], 320), "val": pick(parts["val"], 320)}
    ck = torch.load(V1 / "work/models/unet_fold0_pilot_ckpt.pt", map_location="cpu", weights_only=False)
    print("ckpt keys:", list(ck.keys()), "epoch", ck.get("epoch"))
    state = ck.get("model", ck.get("model_state"))
    for dev in ("mps", "cpu"):
        for bnmode in ("eval", "batchstats"):
            m = models.build_model("unet", pretrained=False); m.load_state_dict(state); m.to(dev)
            m.eval()
            if bnmode == "batchstats":
                for mod in m.modules():
                    if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                        mod.train(); mod.momentum = 0.0   # use batch stats, do not update running stats
            for name, tab in sets.items():
                if dev == "cpu" and len(tab) > 96: tab = tab.iloc[:96]
                imgs, msks = dataset.build_crops(tab, workers=6, log=lambda *a: None) if (name, dev) not in CACHE else CACHE[(name, dev)]
                CACHE[(name, dev)] = (imgs, msks)
                disk = torch.from_numpy(fov.fov_disk_mask(384, fov.ERODE_FRAC)).to(dev)
                ious = []
                with torch.no_grad():
                    for i in range(0, len(tab), 32):
                        x, y = tr.to_batch(imgs, msks, np.arange(i, min(i + 32, len(tab))), dev)
                        iou, _ = tr.batch_metrics(m(x), y, disk); ious.append(iou.cpu())
                v = torch.cat(ious)
                print(f"{dev:4s} bn={bnmode:10s} {name:5s} mean IoU {v.mean():.3f} median {v.median():.3f} frac<0.5 {(v<0.5).float().mean():.2f}")


if __name__ == "__main__":
    main()
