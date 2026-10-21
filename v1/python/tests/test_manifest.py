import nrrd
import numpy as np
import pytest
from PIL import Image

from conftest import load_script

man = load_script("01_manifest.py")
from common import read_image, read_mask  # noqa: E402

DOMAINS = {"D1": {"center": 1, "size": [64, 36]}, "D2": {"center": 1, "size": [48, 27]}, "D3": {"center": 2, "size": [40, 32]}}


def put_frame(base, rel, size, mask_values, segments=1, ext="peek", fmt=None):
    """Write an image and its NRRD next to it (mask at <instrument dir>/seg/<stem>.nrrd)."""
    img = base / rel
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 90, np.uint8)).save(img)
    if mask_values is None:
        return
    data = np.zeros((size[0], size[1], 1), np.uint8)  # NRRD layout: (W, H, 1)
    data[3:10, 2:8, 0] = mask_values[0]
    if len(mask_values) > 1:
        data[12:15, 12:15, 0] = mask_values[1]
    hdr = {f"Segment{i}_ID": f"Segment_{i + 1}" for i in range(segments)}
    hdr["Segment0_Extent"] = "3 9 2 7 0 0" if segments == 1 else "3 14 2 14 0 0"
    seg = img.parent / "seg"
    seg.mkdir(exist_ok=True)
    nrrd.write(str(seg / (img.stem + ".nrrd")), data, hdr)


@pytest.fixture
def tree(tmp_path):
    lab = "classified"
    put_frame(tmp_path, f"{lab}/Small channel/cervical/p46/Normal scenario/grasping forceps/P46-1-0003.png", (48, 27), [255])
    put_frame(tmp_path, f"{lab}/Big channel/lumbar/P7/Difficult scenario/drill/P7-2-0204.jpg", (64, 36), [1])
    put_frame(tmp_path, f"{lab}/Big channel/lumbar/P7/Difficult scenario/drill/p7-2-0205.jpg", (64, 36), [1, 2], segments=2)
    put_frame(tmp_path, f"{lab}/Small channel/lumbar/T001/Normal scenario/bipolar/T001-0618.png", (40, 32), [1])
    put_frame(tmp_path, "unclassified/P69/P69-1-0001.jpg", (64, 36), None)
    return tmp_path


def run(tree):
    return man.build(tree, tree / "classified", tree / "unclassified", DOMAINS)


def test_parse_stem_variants():
    assert man.parse_stem("P7-2-0204") == ("P7", 1, 2, 204)
    assert man.parse_stem("p46-1-0003") == ("P46", 1, 1, 3)
    assert man.parse_stem("T001-0618") == ("T001", 2, None, 618)
    assert man.parse_stem("T122-1-1517") == ("T122", 2, 1, 1517)
    with pytest.raises(ValueError):
        man.parse_stem("frame_12")


def test_layout_and_field_mapping(tree):
    rows, pairs = run(tree)
    df = man.to_frame(rows).set_index("frame_id")
    assert len(df) == 5 and df.index.is_unique
    p46 = df.loc["P46-1-0003"]
    assert (p46.patient_id, p46.center, p46.domain) == ("P46", 1, "D2")  # folder p46 upper-cased
    assert (p46.channel, p46.region, p46.scenario, p46.instrument) == ("small", "cervical", "Normal", "forceps")
    assert p46.file_format == "png" and p46.labeled
    assert p46.image_path == "classified/Small channel/cervical/p46/Normal scenario/grasping forceps/P46-1-0003.png"
    assert p46.mask_path == "classified/Small channel/cervical/p46/Normal scenario/grasping forceps/seg/P46-1-0003.nrrd"
    assert (p46.width, p46.height) == (48, 27)
    assert "P7-2-0205" in df.index  # lower-case stem p7-2-0205 upper-cased in frame_id
    assert df.loc["P7-2-0204"].instrument == "drill" and df.loc["P7-2-0204"].domain == "D1"


def test_missing_video_index_is_null(tree):
    rows, _ = run(tree)
    df = man.to_frame(rows).set_index("frame_id")
    t = df.loc["T001-0618"]
    assert t.center == 2 and t.domain == "D3" and t.partition == "center2_labeled"
    assert df.video_idx.isna().sum() == 1 and df.loc["T001-0618"].frame_idx == 618


def test_stream_rows_unlabeled_with_domain_by_resolution(tree):
    rows, _ = run(tree)
    s = man.to_frame(rows).set_index("frame_id").loc["P69-1-0001"]
    assert s.partition == "stream" and not s.labeled and s.domain == "D1"
    assert s.mask_path is None and s.channel is None and s.instrument is None


def test_pairing_and_orphans(tree):
    rows, pairs = run(tree)
    assert pairs["images_without_mask"] == [] and pairs["masks_without_image"] == []
    (tree / "classified/Big channel/lumbar/P7/Difficult scenario/drill/seg/P7-2-0204.nrrd").unlink()
    nrrd.write(str(tree / "classified/Big channel/lumbar/P7/Difficult scenario/drill/seg/P7-9-9999.nrrd"), np.zeros((64, 36, 1), np.uint8))
    _, pairs = run(tree)
    assert pairs["images_without_mask"] == ["classified/Big channel/lumbar/P7/Difficult scenario/drill/seg/P7-2-0204.nrrd"]
    assert len(pairs["masks_without_image"]) == 1


def test_unknown_resolution_fails(tree):
    put_frame(tree, "classified/Small channel/lumbar/P8/Normal scenario/punch/P8-1-0001.png", (50, 50), [1])
    with pytest.raises(AssertionError, match="outside the three known domains"):
        run(tree)


def test_frame_id_collision_gets_suffix(tree):
    put_frame(tree, "classified/Small channel/lumbar/P7/Normal scenario/drill/P7-2-0204.png", (64, 36), [1])
    rows, _ = run(tree)
    ids = sorted(r["frame_id"] for r in rows if r["stem"].upper() == "P7-2-0204")
    assert ids == ["P7-2-0204", "P7-2-0204__2"]
    assert man.assign_frame_ids(rows) == 1


def test_reconcile_flags_mismatch(tree):
    rows, pairs = run(tree)
    df = man.to_frame(rows)
    expected = dict(images_total=5, labeled_frames=4, labeled_patients=3, center1_patients=2, center2_patients=1,
                    center1_frames=3, center2_frames=1, stream_frames=1, stream_patients=1)
    table, extra, ok = man.reconcile(df, pairs, expected)
    assert ok and extra["paired_images"] == 4
    _, _, ok = man.reconcile(df, pairs, {**expected, "center1_frames": 4})
    assert not ok


def test_mask_union_and_orientation(tree):
    b = "classified/Big channel/lumbar/P7/Difficult scenario/drill/seg/"
    m = read_mask(tree / (b + "P7-2-0204.nrrd"))
    assert m.dtype == bool and m.shape == (36, 64)  # H x W
    assert m[2:8, 3:10].all() and m.sum() == 6 * 7  # array [x=3..9, y=2..7] -> image rows 2..7, cols 3..9
    two = read_mask(tree / (b + "P7-2-0205.nrrd"))
    assert two.sum() == 6 * 7 + 3 * 3 and two[12:15, 12:15].all()  # second segment (value 2) is unioned
    p46 = read_mask(tree / "classified/Small channel/cervical/p46/Normal scenario/grasping forceps/seg/P46-1-0003.nrrd")
    assert p46.sum() == 6 * 7  # {0,255} encoding


def test_scan_mask_reports_encoding_segments_extent(tree):
    b = "classified/Big channel/lumbar/P7/Difficult scenario/drill/seg/"
    one = man.scan_mask((tree, b + "P7-2-0204.nrrd", 64, 36))
    assert one["values"] == (0, 1) and one["n_segments"] == 1 and one["shape_ok"] and one["extent"] == "tight"
    two = man.scan_mask((tree, b + "P7-2-0205.nrrd", 64, 36))
    assert two["values"] == (0, 1, 2) and two["n_segments"] == 2 and two["extent"] == "tight"
    assert man.scan_mask((tree, b + "P7-2-0204.nrrd", 36, 64))["shape_ok"] is False  # axis order guard


def test_read_image_rgb_order(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.full((4, 6, 3), (200, 10, 30), np.uint8)).save(p)
    img = read_image(p)
    assert img.shape == (4, 6, 3) and img.dtype == np.uint8 and tuple(img[0, 0]) == (200, 10, 30)
