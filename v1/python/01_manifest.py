"""Build derived/manifest.parquet and work/logs/manifest_audit.json from the extracted SEA dataset.

Labeled layout: classified/{Big channel|Small channel}/{cervical|lumbar}/{patient}/
{Normal scenario|Difficult scenario}/{instrument}/{stem}.{png|jpg}, mask at {instrument}/seg/{stem}.nrrd.
Stream layout: unclassified/{patient}/{stem}.jpg, unlabeled.

frame_id is the upper-cased file stem (a few Center 1 stems are lower-case, e.g. p23-1-0140). A repeated
stem would get suffix __2, __3 in path order; the real data have none (audited, reported below).
Exits non-zero, after printing the reconciliation, if any expected count differs.
"""
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nrrd
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

STEM_RE = re.compile(r"^([PT]\d+)-(?:(\d+)-)?(\d+)$", re.IGNORECASE)
CHANNELS = {"Big channel": "big", "Small channel": "small"}
SCENARIOS = {"Normal scenario": "Normal", "Difficult scenario": "Difficult"}
INSTRUMENTS = {"grasping forceps": "forceps", "bipolar": "bipolar", "drill": "drill",
               "punch": "punch", "scissor": "scissor", "dissector": "dissector"}
REGIONS = ("cervical", "lumbar")
IMAGE_EXT = {".png": "png", ".jpg": "jpg"}
EXIF_ORIENTATION = 274
POOL = 16


def parse_stem(stem):
    """'P7-2-0204' -> ('P7', 1, 2, 204); 'T001-0618' (no video index) -> ('T001', 2, None, 618)."""
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(f"unparseable frame name: {stem}")
    patient = m.group(1).upper()
    video = int(m.group(2)) if m.group(2) is not None else None
    return patient, 1 if patient[0] == "P" else 2, video, int(m.group(3))


def _images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT and p.is_file())


def _dirs(folder):
    return sorted(p for p in folder.iterdir() if p.is_dir())


def _rel(p, root):
    return p.relative_to(root).as_posix()


def scan_labeled(labeled_dir, root):
    rows = []
    for ch_dir in _dirs(labeled_dir):
        for reg_dir in _dirs(ch_dir):
            for pat_dir in _dirs(reg_dir):
                for sc_dir in _dirs(pat_dir):
                    for ins_dir in _dirs(sc_dir):
                        for img in _images(ins_dir):
                            patient, center, video, frame = parse_stem(img.stem)
                            if patient != pat_dir.name.upper():
                                raise ValueError(f"frame {img} does not belong to patient folder {pat_dir.name}")
                            rows.append(dict(
                                stem=img.stem, partition=f"center{center}_labeled", patient_id=patient,
                                patient_folder=pat_dir.name, center=center,
                                channel=CHANNELS[ch_dir.name], region=reg_dir.name,
                                scenario=SCENARIOS[sc_dir.name], instrument=INSTRUMENTS[ins_dir.name],
                                file_format=IMAGE_EXT[img.suffix.lower()], video_idx=video, frame_idx=frame,
                                image_path=_rel(img, root), mask_path=_rel(ins_dir / "seg" / (img.stem + ".nrrd"), root),
                                labeled=True))
    return rows


def scan_stream(stream_dir, root):
    rows = []
    for pat_dir in _dirs(stream_dir):
        for img in _images(pat_dir):
            patient, center, video, frame = parse_stem(img.stem)
            rows.append(dict(
                stem=img.stem, partition="stream", patient_id=patient, patient_folder=pat_dir.name,
                center=center, channel=None, region=None, scenario=None, instrument=None,
                file_format=IMAGE_EXT[img.suffix.lower()], video_idx=video, frame_idx=frame,
                image_path=_rel(img, root), mask_path=None, labeled=False))
    return rows


def assign_frame_ids(rows):
    """Upper-cased stem; repeats get __2, __3 ... in path order. Returns the number of collisions."""
    seen, collisions = Counter(), 0
    for r in sorted(rows, key=lambda r: r["image_path"]):
        base = r["stem"].upper()
        seen[base] += 1
        r["frame_id"] = base if seen[base] == 1 else f"{base}__{seen[base]}"
        collisions += seen[base] > 1
    return collisions


def _header(args):
    root, rel = args
    with Image.open(root / rel) as im:
        orient = im.getexif().get(EXIF_ORIENTATION) if im.format == "JPEG" else None
        return im.size[0], im.size[1], orient


def add_sizes(rows, root):
    with ThreadPoolExecutor(POOL) as ex:
        for r, (w, h, orient) in zip(rows, ex.map(_header, [(root, r["image_path"]) for r in rows])):
            r["width"], r["height"], r["exif_orientation"] = w, h, orient


def assign_domains(rows, domains):
    by_size = {tuple(d["size"]): (k, d["center"]) for k, d in domains.items()}
    unknown = Counter()
    for r in rows:
        hit = by_size.get((r["width"], r["height"]))
        if hit is None:
            unknown[(r["partition"], r["width"], r["height"])] += 1
            continue
        r["domain"] = hit[0]
        if r["labeled"] and hit[1] != r["center"]:
            unknown[(r["partition"], "center/size mismatch", r["width"], r["height"])] += 1
    if unknown:
        raise AssertionError(f"frames outside the three known domains: {dict(unknown)}")


def pair_masks(rows, labeled_dir, root):
    """Every labeled image needs its NRRD and every NRRD its image."""
    want = {r["mask_path"] for r in rows if r["labeled"]}
    have = {_rel(p, root) for p in labeled_dir.rglob("*.nrrd")}
    return sorted(want - have), sorted(have - want)


def scan_mask(args):
    root, rel, width, height = args
    data, hdr = nrrd.read(str(root / rel))
    shape, fg = data.shape, data > 0
    seg_ids = [k for k in hdr if re.fullmatch(r"Segment\d+_ID", k)]
    out = dict(shape_ok=data.ndim == 3 and shape == (width, height, 1), values=tuple(int(v) for v in np.unique(data)),
               n_segments=len(seg_ids), empty=not fg.any(), fg_frac=float(fg.mean()))
    xs, ys = np.nonzero(fg.any(axis=tuple(range(2, fg.ndim))))
    got = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) if len(xs) else None
    ext = [list(map(int, hdr[f"Segment{i}_Extent"].split()))[:4] for i in range(len(seg_ids)) if f"Segment{i}_Extent" in hdr]
    want = (min(e[0] for e in ext), max(e[1] for e in ext), min(e[2] for e in ext), max(e[3] for e in ext)) if ext else None
    # The Slicer header extent is the tight foreground box, the whole frame, or a looser box that encloses the
    # foreground (checks axis 0 = x, axis 1 = y); anything else would signal an axis problem.
    if want == got:
        out["extent"] = "tight"
    elif want == (0, width - 1, 0, height - 1):
        out["extent"] = "full_frame"
    elif want and got and want[0] <= got[0] and want[1] >= got[1] and want[2] <= got[2] and want[3] >= got[3]:
        out["extent"] = "encloses_foreground"
    else:
        out["extent"] = "other"
    return out


def scan_masks(rows, root):
    labeled = [r for r in rows if r["labeled"]]
    with ThreadPoolExecutor(POOL) as ex:
        return list(ex.map(scan_mask, [(root, r["mask_path"], r["width"], r["height"]) for r in labeled]))


def build(root, labeled_dir, stream_dir, domains):
    rows = scan_labeled(labeled_dir, root) + scan_stream(stream_dir, root)
    collisions = assign_frame_ids(rows)
    add_sizes(rows, root)
    assign_domains(rows, domains)
    orphan_masks, orphan_images = pair_masks(rows, labeled_dir, root)
    return rows, dict(frame_id_collisions=collisions, images_without_mask=orphan_masks, masks_without_image=orphan_images)


COLUMNS = ["frame_id", "partition", "patient_id", "center", "domain", "width", "height", "channel", "region",
           "scenario", "instrument", "file_format", "video_idx", "frame_idx", "image_path", "mask_path", "labeled"]


def to_frame(rows):
    df = pd.DataFrame(rows).sort_values("image_path", kind="stable").reset_index(drop=True)
    for c in ("video_idx", "frame_idx"):
        df[c] = df[c].astype("Int64")
    return df


def reconcile(df, pairs, expected):
    lab = df[df.labeled]
    observed = dict(
        images_total=len(df), labeled_frames=len(lab), labeled_patients=lab.patient_id.nunique(),
        center1_patients=lab[lab.center == 1].patient_id.nunique(), center2_patients=lab[lab.center == 2].patient_id.nunique(),
        center1_frames=int((lab.center == 1).sum()), center2_frames=int((lab.center == 2).sum()),
        stream_frames=int((df.partition == "stream").sum()), stream_patients=df[df.partition == "stream"].patient_id.nunique())
    table = {k: dict(expected=expected[k], observed=observed[k], ok=expected[k] == observed[k]) for k in expected}
    extra = dict(unique_frame_ids=df.frame_id.is_unique, paired_images=len(lab) - len(pairs["images_without_mask"]),
                 orphan_masks=len(pairs["masks_without_image"]), orphan_images=len(pairs["images_without_mask"]))
    ok = all(v["ok"] for v in table.values()) and extra["unique_frame_ids"] and extra["paired_images"] == expected["labeled_frames"] \
        and not extra["orphan_masks"] and not extra["orphan_images"]
    return table, extra, ok


def per_patient(lab):
    out = {}
    for dom, g in lab.groupby("domain"):
        n = g.groupby("patient_id").size()
        out[dom] = dict(patients=int(len(n)), frames=int(n.sum()), median=float(n.median()), min=int(n.min()), max=int(n.max()))
    return out


def audit(df, masks, rows, pairs, table, extra):
    lab = df[df.labeled]
    lab_rows = [r for r in rows if r["labeled"]]
    enc = Counter("{" + ",".join(map(str, m["values"])) + "}" for m in masks)
    return dict(
        reconciliation=table, pairing=extra, frame_id_collisions=pairs["frame_id_collisions"],
        counts=dict(
            by_domain={k: dict(patients=int(g.patient_id.nunique()), frames=len(g)) for k, g in lab.groupby("domain")},
            stream_by_domain=df[~df.labeled].domain.value_counts().to_dict(),
            by_channel=lab.channel.value_counts().to_dict(), by_region=lab.region.value_counts().to_dict(),
            by_scenario=lab.scenario.value_counts().to_dict(), by_instrument=lab.instrument.value_counts().to_dict(),
            by_file_format=lab.file_format.value_counts().to_dict(),
            scenario_by_domain=lab.groupby(["domain", "scenario"]).size().unstack(fill_value=0).to_dict("index")),
        frames_per_patient=per_patient(lab),
        stream_frames_per_patient=dict(median=float(df[~df.labeled].groupby("patient_id").size().median()),
                                       min=int(df[~df.labeled].groupby("patient_id").size().min()),
                                       max=int(df[~df.labeled].groupby("patient_id").size().max())),
        anomalies=dict(
            lowercase_patient_folders=sorted({r["patient_folder"] for r in rows if r["patient_folder"] != r["patient_id"]}),
            lowercase_stems=sum(r["stem"] != r["stem"].upper() for r in rows),
            frames_without_video_index=df[df.video_idx.isna()].groupby("partition").size().to_dict(),
            jpeg_exif_orientation_not_1=sum(1 for r in rows if r["exif_orientation"] not in (None, 1)),
            mask_value_encodings=dict(enc), masks_with_n_segments=dict(Counter(m["n_segments"] for m in masks)),
            two_segment_frames_by_encoding=dict(Counter("{" + ",".join(map(str, m["values"])) + "}" for m in masks if m["n_segments"] == 2)),
            mask_shape_not_W_H_1=sum(not m["shape_ok"] for m in masks),
            mask_header_extent=dict(Counter(m["extent"] for m in masks)),
            empty_masks=sum(m["empty"] for m in masks),
            patients_in_more_than_one_channel_region=int((lab.groupby("patient_id")[["channel", "region"]].nunique() > 1).any(axis=1).sum()),
            patients_with_mixed_domain=int((lab.groupby("patient_id").domain.nunique() > 1).sum())),
        mask_foreground_fraction=dict(median=float(np.median([m["fg_frac"] for m in masks])),
                                      min=float(min(m["fg_frac"] for m in masks)), max=float(max(m["fg_frac"] for m in masks))),
        n_labeled_rows_scanned=len(lab_rows))


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    cfg = load_config()
    root = ROOT
    p = cfg["paths"]
    rows, pairs = build(root, root / p["labeled_dir"], root / p["stream_dir"], cfg["domains"])
    df = to_frame(rows)
    table, extra, ok = reconcile(df, pairs, cfg["dataset_counts_expected"])
    print("reconciliation (expected vs observed)")
    for k, v in table.items():
        print(f"  {k:18s} {v['expected']:>7} {v['observed']:>7}  {'ok' if v['ok'] else 'MISMATCH'}")
    print(f"  unique frame_id={extra['unique_frame_ids']}  paired={extra['paired_images']}  "
          f"orphan masks={extra['orphan_masks']}  images without mask={extra['orphan_images']}  "
          f"frame_id collisions={pairs['frame_id_collisions']}")
    if not ok:
        print("FAIL: counts differ from analysis_plan.yaml:dataset_counts_expected", file=sys.stderr)
        sys.exit(1)
    masks = scan_masks(rows, root)
    report = audit(df, masks, rows, pairs, table, extra)
    out_parquet = root / p["derived"] / "manifest.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(out_parquet, index=False)
    log = root / p["logs"] / "manifest_audit.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(report, indent=2, default=_plain))
    print(f"wrote {out_parquet} ({len(df)} rows) and {log}")


if __name__ == "__main__":
    main()
