"""Gate G2 evidence: fold-0 pilot models scored on the 12 held-out fold-0 patients (Center 1 only).

Scratch script (not part of the deliverable pipeline). IoU is on the 384 grid inside the eroded FOV disk, an
approximation of the native-resolution endpoint that T11 computes later. Center 2 is never touched here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1 / "python"))
import dataset
import fov
import models
from common import load_config



def main():
    cfg = load_config()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    parts = dataset.patients_for(0, pilot=True)
    t = dataset.frame_table("center1_labeled")
    t = t[t.patient_id.isin(parts["heldout"])].reset_index(drop=True)
    print("held-out fold 0:", t.patient_id.nunique(), "patients", len(t), "frames")
    images, masks = dataset.build_crops(t)
    disk = torch.from_numpy(fov.fov_disk_mask(dataset.SIZE, cfg["fov"]["erode_frac"])).to(dev)

    preds = {}
    for name in ("unet", "segformer"):
        w = V1 / "work/models" / f"{name}_fold0_pilot.pt"
        if not w.exists():
            print("missing", w)
            continue
        m = models.build_model(name, pretrained=False)
        m.load_state_dict(torch.load(w, map_location="cpu"))
        m.eval().to(dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(t), 32):
                x = torch.from_numpy(images[i:i + 32]).to(dev).permute(0, 3, 1, 2).float() / 255
                out.append((torch.sigmoid(m(x))[:, 0] > 0.5).cpu().numpy())
        preds[name] = np.concatenate(out)

    d = disk.cpu().numpy().astype(bool)
    res = t[["frame_id", "patient_id", "domain", "scenario", "instrument"]].copy()
    for name, p in preds.items():
        g = masks.astype(bool) & d
        pp = p & d
        inter = (g & pp).sum((1, 2))
        union = (g | pp).sum((1, 2))
        res[f"iou_{name}"] = inter / np.maximum(union, 1)
    if len(preds) == 2:
        a, b = preds["unet"] & d, preds["segformer"] & d
        dice = 2 * (a & b).sum((1, 2)) / np.maximum(a.sum((1, 2)) + b.sum((1, 2)), 1)
        res["ens_dis"] = np.where(a.sum((1, 2)) + b.sum((1, 2)) == 0, 0.0, 1 - dice)

    summary = {}
    for name in preds:
        v = res[f"iou_{name}"]
        row = {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()),
               "q": {str(q): float(v.quantile(q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)}}
        for thr in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
            ev = res[v < thr]
            row[f"fail_lt_{thr}"] = {"events": int(len(ev)), "share": float(len(ev) / len(v)),
                                     "patients_with_events": int(ev.patient_id.nunique())}
        row["by_domain_median"] = res.groupby("domain")[f"iou_{name}"].median().round(3).to_dict()
        row["by_scenario_median"] = res.groupby("scenario")[f"iou_{name}"].median().round(3).to_dict()
        row["fail075_by_domain"] = res.assign(f=v < 0.75).groupby("domain").f.mean().round(3).to_dict()
        summary[name] = row
    if "ens_dis" in res:
        summary["ens_dis_median"] = float(res.ens_dis.median())
    print(json.dumps(summary, indent=1))
    out = V1 / "work/logs/g2_fold0_pilot_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    res.to_csv(V1 / "work/logs/g2_fold0_pilot_frames.csv", index=False)

    for name in preds:
        log = V1 / "work/models" / f"train_log_{name}_fold0_pilot.csv"
        if log.exists():
            print(name, "inner-val curve (epoch, val_iou, val_fail_frac):")
            print(pd.read_csv(log)[["epoch", "train_loss", "val_iou", "val_iou_median", "val_fail_frac"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
