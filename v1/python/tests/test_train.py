"""Tests for dataset.py, models.py and 04_train.py: CPU only, tiny synthetic crops, no dependence on the real images."""
import numpy as np
import pandas as pd
import pytest
import torch

import dataset
import models
from conftest import ROOT, load_script

train = load_script("04_train.py")

S = 64   # synthetic crop size (fully convolutional models, divisible by 32)
ATLAS = ROOT / "Spine endoscopic atlas"


def synthetic_crops(n=32, size=S, seed=0):
    """Noisy grey frames with one bright rectangle; the mask is the rectangle. An easy, learnable task."""
    rng = np.random.default_rng(seed)
    images = rng.integers(20, 90, size=(n, size, size, 3), dtype=np.uint8)
    masks = np.zeros((n, size, size), dtype=np.uint8)
    for i in range(n):
        y0, x0 = rng.integers(4, size // 2, 2)
        h, w = rng.integers(size // 5, size // 3, 2)
        images[i, y0:y0 + h, x0:x0 + w] = rng.integers(180, 250, size=3)
        masks[i, y0:y0 + h, x0:x0 + w] = 1
    return images, masks


@pytest.fixture
def light_models(monkeypatch):
    """Random-init weights: no dependence on the weight cache and faster to build."""
    monkeypatch.setattr(train, "build_model", lambda name: models.build_model(name, pretrained=False))


# ---------------------------------------------------------------- models

@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_model_forward_backward_shape(name):
    m = models.build_model(name, pretrained=False)
    x = torch.rand(2, 3, 384, 384)                 # uint8 / 255 style input
    out = m(x)
    assert out.shape == (2, 1, 384, 384) and out.dtype == torch.float32
    out.mean().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_pretrained_weights_load_from_local_cache(name):
    m = models.build_model(name)                   # fails if the cached ImageNet / mit-b0 weights are not there
    assert m(torch.rand(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_normalisation_lives_inside_the_wrapper():
    m = models.build_model("unet", pretrained=False).eval()
    x = torch.rand(1, 3, 64, 64)
    z = (x - m.mean) / m.std
    assert torch.allclose(m(x), m.net(z), atol=1e-6)


def test_unknown_model_name_rejected():
    with pytest.raises(ValueError):
        models.build_model("nnunet")


# ---------------------------------------------------------------- augmentation

def rectangle(size=S, y=(10, 26), x=(18, 46)):
    m = torch.zeros(1, 1, size, size)
    m[..., y[0]:y[1], x[0]:x[1]] = 1
    return m


def test_translation_moves_image_and_mask_together():
    m = rectangle()
    img = m.repeat(1, 3, 1, 1)
    t = torch.tensor([[[1.0, 0.0, 0.25], [0.0, 1.0, -0.125]]])     # sample 0.25 right / 0.125 up: content moves 8 px left, 4 down
    xw, mw = train.apply_affine(img, m, t)
    assert torch.equal((xw[:, :1] > 0.5).float(), mw)               # image and mask stay aligned
    ys, xs = torch.nonzero(m[0, 0], as_tuple=True)
    yw, xw_ = torch.nonzero(mw[0, 0], as_tuple=True)
    assert xw_.float().mean() - xs.float().mean() == pytest.approx(-8.0, abs=0.01)
    assert yw.float().mean() - ys.float().mean() == pytest.approx(4.0, abs=0.01)
    assert mw.sum() == m.sum()


def test_quarter_turn_lands_where_the_affine_puts_it():
    m = rectangle()
    img = m.repeat(1, 3, 1, 1)                                       # white rectangle on black
    t = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]])        # out[i, j] = in[j, S - 1 - i]
    xw, mw = train.apply_affine(img, m, t)
    expected = m.transpose(-1, -2).flip(-2)
    assert torch.equal(mw, expected)
    assert torch.allclose(xw[:, 0:1], expected, atol=1e-5)


def test_random_augmentation_keeps_image_and_mask_aligned():
    m = rectangle().repeat(16, 1, 1, 1)
    img = m.repeat(1, 3, 1, 1)
    theta = train.sample_theta(16, torch.Generator().manual_seed(3))
    xw, mw = train.apply_affine(img, m, theta)
    a, b = xw[:, 0:1] > 0.5, mw > 0.5
    iou = ((a & b).sum((1, 2, 3)).float() / (a | b).sum((1, 2, 3)).float().clamp(min=1))
    assert iou.min() > 0.9
    assert (mw.sum((1, 2, 3)) > 0).all()


def test_flips_are_pixel_permutations():
    img = torch.rand(2, 3, S, S)
    m = torch.rand(2, 1, S, S).round()
    for t in ([[-1.0, 0, 0], [0, 1.0, 0]], [[1.0, 0, 0], [0, -1.0, 0]], [[-1.0, 0, 0], [0, -1.0, 0]]):
        xw, mw = train.apply_affine(img, m, torch.tensor([t, t]))
        assert torch.allclose(xw.flatten(2).sort(-1).values, img.flatten(2).sort(-1).values, atol=1e-5)   # grid float error only
        assert torch.equal(mw.flatten(2).sort(-1).values, m.flatten(2).sort(-1).values)


def test_augmentation_does_not_change_colour():
    colour = torch.tensor([0.2, 0.5, 0.8]).view(1, 3, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    disk = (((yy - S / 2 + 0.5) ** 2 + (xx - S / 2 + 0.5) ** 2) < (0.45 * S / 2) ** 2).float().view(1, 1, S, S)
    img = (disk * colour).repeat(32, 1, 1, 1)
    theta = train.sample_theta(32, torch.Generator().manual_seed(5))
    xw, _ = train.apply_affine(img, disk.repeat(32, 1, 1, 1), theta)
    # interpolation can only blend the colour with the black surround, so chromaticity is fixed and the
    # brightest pixels keep the original colour exactly
    assert torch.allclose(xw[:, 1], 2.5 * xw[:, 0], atol=1e-6) and torch.allclose(xw[:, 2], 4.0 * xw[:, 0], atol=1e-6)
    assert xw[:, 0].max() == pytest.approx(0.2, abs=1e-6)
    assert xw.max() <= 0.8 + 1e-6 and xw.min() >= 0


def test_sampled_transforms_stay_in_the_configured_ranges():
    theta = train.sample_theta(4000, torch.Generator().manual_seed(1))
    scale = 1 / torch.linalg.norm(theta[:, :, :2], dim=2)           # rows of R/s diag(f) have norm 1/s
    assert scale.min() >= train.AUG["scale"][0] - 1e-5 and scale.max() <= train.AUG["scale"][1] + 1e-5
    assert theta[:, :, 2].abs().max() <= train.AUG["shift"] + 1e-6
    assert ((theta[:, 0, 0] < 0).float().mean() - 0.5).abs() < 0.05   # about half the samples are flipped (rotation +-30 deg keeps the sign)


# ---------------------------------------------------------------- batch building on the host (MPS grid_sample regression)
# On MPS, F.grid_sample returned dimmed, low-contrast images for real crops, so augmented_batch warps on the CPU and
# only moves the finished batch to the device. These tests pin that behaviour.

IDENTITY = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def identity_draw(monkeypatch):
    monkeypatch.setattr(train, "sample_theta", lambda n, gen, aug=train.AUG: IDENTITY.repeat(n, 1, 1))


def test_identity_draw_keeps_image_statistics_and_mask_alignment(monkeypatch):
    identity_draw(monkeypatch)
    images, masks = synthetic_crops(8)
    idx = np.arange(8)
    x, y = train.augmented_batch(images, masks, idx, torch.Generator().manual_seed(0), "cpu")
    ref = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255)
    assert x.device.type == "cpu" and y.device.type == "cpu" and x.shape == ref.shape and y.shape == (8, 1, S, S)
    assert x.mean().item() == pytest.approx(ref.mean().item(), rel=0.02)
    assert x.std().item() == pytest.approx(ref.std().item(), rel=0.02)
    for i in range(8):                                               # per frame too: dimming would shrink every frame
        assert x[i].mean().item() == pytest.approx(ref[i].mean().item(), rel=0.02)
        assert x[i].std().item() == pytest.approx(ref[i].std().item(), rel=0.02)
    assert torch.equal(y[:, 0], torch.from_numpy(masks).float())     # the rectangle masks are untouched ...
    assert torch.equal((x.amax(1) > 0.6).float(), y[:, 0])           # ... and still sit exactly on the bright rectangles


def circle_crops(n=32, size=384, seed=0):
    """1080p-like content: full-range noisy frames inside a circle, pure black outside, a rectangle mask inside."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:size, :size]
    disk = (yy - size / 2 + 0.5) ** 2 + (xx - size / 2 + 0.5) ** 2 < (0.46 * size) ** 2
    images = rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8) * disk[None, :, :, None].astype(np.uint8)
    masks = np.zeros((n, size, size), dtype=np.uint8)
    for i in range(n):
        y0, x0 = rng.integers(100, 180, 2)
        masks[i, y0:y0 + 80, x0:x0 + 60] = 1
    return images, masks


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available on this machine; the device parity check needs an Apple GPU")
def test_mps_batches_equal_the_cpu_batches():
    images, masks = circle_crops()
    idx = np.arange(32)
    xc, yc = train.augmented_batch(images, masks, idx, torch.Generator().manual_seed(11), "cpu")
    xm, ym = train.augmented_batch(images, masks, idx, torch.Generator().manual_seed(11), "mps")
    assert xm.device.type == "mps" and ym.device.type == "mps"
    assert (xm.cpu() - xc).abs().max().item() <= 1e-6
    assert (ym.cpu() - yc).abs().max().item() <= 1e-6
    # the failure seen on MPS was a loss of brightness and contrast; the device batch must keep the CPU statistics
    assert xm.cpu().std().item() == pytest.approx(xc.std().item(), rel=1e-6)
    assert xm.cpu().mean().item() == pytest.approx(xc.mean().item(), rel=1e-6)


def tiny_epoch_devices(monkeypatch, device):
    """Run one epoch with a stub optimiser step; return the devices apply_affine saw and the device the step received."""
    seen, stepped = [], []
    real = train.apply_affine

    def spy(x, m, theta):
        seen.append((x.device.type, m.device.type, theta.device.type))
        return real(x, m, theta)

    def stub_step(model, opt, sched, x, y, precision="fp32", scaler=None):
        stepped.append((x.device.type, y.device.type))
        return torch.tensor(0.5, device=x.device)

    monkeypatch.setattr(train, "apply_affine", spy)
    monkeypatch.setattr(train, "train_step", stub_step)
    images, masks = synthetic_crops(32)
    run = train.Run("unet", "t", 1, 3, device, "fp32", None, None)
    train.train_epoch(torch.nn.Identity(), None, None, images, masks, np.arange(32), 1, run)
    return seen, stepped


def test_train_epoch_never_warps_off_the_cpu(monkeypatch):
    seen, stepped = tiny_epoch_devices(monkeypatch, "cpu")
    assert len(seen) == 32 // train.BATCH and stepped
    assert set(seen) == {("cpu", "cpu", "cpu")}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available on this machine; the device guard needs an Apple GPU")
def test_train_epoch_on_mps_warps_on_the_cpu_and_steps_on_the_device(monkeypatch):
    seen, stepped = tiny_epoch_devices(monkeypatch, "mps")
    assert set(seen) == {("cpu", "cpu", "cpu")}                      # apply_affine never sees an MPS tensor
    assert set(stepped) == {("mps", "mps")}                          # only the finished batch goes to the device


# ---------------------------------------------------------------- loss and metrics

def test_loss_is_small_for_a_confident_correct_prediction_and_large_for_the_wrong_one():
    y = rectangle().repeat(4, 1, 1, 1)
    good = (y * 2 - 1) * 12
    assert train.seg_loss(good, y) < 1e-3
    assert train.seg_loss(-good, y) > 5


def test_iou_and_dice_on_known_masks():
    disk = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    gt = torch.zeros(1, 1, 4, 4)
    gt[..., :2, :] = 1                                              # 8 px
    pred = torch.full((1, 1, 4, 4), -1.0)
    pred[..., :1, :] = 1                                            # 4 px, inside the GT
    iou, dice = train.batch_metrics(pred, gt, disk)
    assert iou.item() == pytest.approx(0.5) and dice.item() == pytest.approx(2 * 4 / 12)
    iou, dice = train.batch_metrics(torch.full((1, 1, 4, 4), -1.0), gt, disk)
    assert iou.item() == 0 and dice.item() == 0                     # empty prediction scores 0
    outside = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    outside[..., 2:, :] = True
    iou, _ = train.batch_metrics(torch.full((1, 1, 4, 4), 1.0), gt, outside)
    assert iou.item() == 0                                          # only pixels inside the FOV disk count


# ---------------------------------------------------------------- training loop

@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_loss_falls_on_an_easy_task(name):
    images, masks = synthetic_crops(16)
    x, y = train.to_batch(images, masks, np.arange(16), "cpu")
    torch.manual_seed(0)
    m = models.build_model(name, pretrained=False).train()
    opt = torch.optim.AdamW(m.parameters(), lr=train.LR, weight_decay=train.WEIGHT_DECAY)
    losses = [float(train.train_step(m, opt, None, x, y)) for _ in range(25)]
    assert np.isfinite(losses).all()
    assert np.mean(losses[-3:]) < 0.8 * np.mean(losses[:3])


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_checkpoint_resume_reproduces_the_run(name, tmp_path, light_models):
    images, masks = synthetic_crops(40)
    idx, val = np.arange(32), np.arange(32, 40)
    epochs = 3

    def make(dirname):
        d = tmp_path / dirname
        d.mkdir()
        run = train.Run(name, "t", epochs, 7, "cpu", "fp32", d / "ck.pt", d / "log.csv")
        return run, *train.make_training(name, epochs, len(idx) // train.BATCH, "cpu", 7)

    run, ref_model, opt, sched = make("ref")
    ref = train.fit(run, ref_model, opt, sched, images, masks, idx, val, log=lambda m: None)

    run, model, opt, sched = make("killed")
    train.fit(run, model, opt, sched, images, masks, idx, val, log=lambda m: None, stop_after=1)
    assert run.ckpt_path.exists() and len(pd.read_csv(run.log_path)) == 1
    # a new process: fresh objects with a different random init, restored purely from the checkpoint
    run2 = train.Run(name, "t", epochs, 7, "cpu", "fp32", run.ckpt_path, run.log_path)
    model2, opt2, sched2 = train.make_training(name, epochs, len(idx) // train.BATCH, "cpu", 12345)
    resumed = train.fit(run2, model2, opt2, sched2, images, masks, idx, val, log=lambda m: None)

    assert [r["epoch"] for r in resumed] == [1, 2, 3]
    assert [r["train_loss"] for r in resumed] == pytest.approx([r["train_loss"] for r in ref], rel=1e-5)
    assert [r["val_iou"] for r in resumed] == pytest.approx([r["val_iou"] for r in ref], rel=1e-5, abs=1e-6)
    for (k, a), (_, b) in zip(ref_model.state_dict().items(), model2.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), k


def test_resume_refuses_a_checkpoint_from_a_different_schedule(tmp_path, light_models):
    images, masks = synthetic_crops(32)
    idx = np.arange(32)
    run = train.Run("unet", "t", 2, 1, "cpu", "fp32", tmp_path / "ck.pt", tmp_path / "log.csv")
    model, opt, sched = train.make_training("unet", 2, 2, "cpu", 1)
    train.fit(run, model, opt, sched, images, masks, idx, [], log=lambda m: None, stop_after=1)
    other = train.Run("unet", "t", 5, 1, "cpu", "fp32", tmp_path / "ck.pt", tmp_path / "log.csv")
    with pytest.raises(ValueError, match="epochs"):
        train.fit(other, model, opt, sched, images, masks, idx, [], log=lambda m: None)


def test_pilot_outputs_never_collide_with_the_real_fold_files():
    real = train.run_paths("unet", "0", False)
    pilot = train.run_paths("unet", "0", True)
    assert real[0] == "fold0" and pilot[0] == "fold0_pilot"
    assert not {p.name for p in real[1:]} & {p.name for p in pilot[1:]}
    assert train.run_paths("segformer", "final", False)[1].name == "segformer_final.pt"


# ---------------------------------------------------------------- patient sets

def synthetic_folds(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(45):
        dom = "D1" if i < 30 else "D2"
        rows.append(dict(patient_id=f"P{i + 1}", center=1, domain=dom, n_frames=int(rng.integers(20, 400)), role="cv", fold=i % 5))
    rows += [dict(patient_id=f"T{i + 1}", center=2, domain="D3", n_frames=20, role="external", fold=-1) for i in range(4)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("pilot", [False, True])
def test_patient_sets_are_disjoint_synthetic(fold, pilot):
    f = synthetic_folds()
    p = dataset.patients_for(fold, pilot, folds=f)
    tr, va, ho = set(p["train"]), set(p["val"]), set(p["heldout"])
    assert not (tr & va) and not (tr & ho) and not (va & ho)
    assert tr | va | ho == set(f[f.role == "cv"].patient_id)
    assert ho == set(f[(f.role == "cv") & (f.fold == fold)].patient_id)
    assert len(va) == (dataset.N_INNER if pilot else 0)
    if pilot:
        dom = f.set_index("patient_id").loc[sorted(va)].domain
        assert set(dom) == {"D1", "D2"} and (dom == "D2").sum() >= dataset.MIN_INNER_PER_DOMAIN
        assert p == dataset.patients_for(fold, True, folds=f)       # deterministic


def test_final_uses_all_cv_patients_and_rejects_pilot():
    f = synthetic_folds()
    p = dataset.patients_for("final", folds=f)
    assert set(p["train"]) == set(f[f.role == "cv"].patient_id) and not p["val"] and not p["heldout"]
    with pytest.raises(ValueError):
        dataset.patients_for("final", pilot=True, folds=f)


def test_patient_sets_are_disjoint_on_the_real_folds():
    path = dataset.DERIVED / "folds.csv"
    if not path.exists():
        pytest.skip(f"{path} is missing; run python/03_folds.py first")
    f = pd.read_csv(path)
    cv = set(f[f.role == "cv"].patient_id)
    held_all = []
    for fold in range(5):
        for pilot in (False, True):
            p = dataset.patients_for(fold, pilot)
            tr, va, ho = set(p["train"]), set(p["val"]), set(p["heldout"])
            assert not (tr & va) and not (tr & ho) and not (va & ho)
            assert tr | va | ho == cv
            assert len(va) == (10 if pilot else 0)
        held_all += p["heldout"]
        if pilot:
            assert set(f.set_index("patient_id").loc[p["val"]].domain) == {"D1", "D2"}
    assert sorted(held_all) == sorted(cv)                            # every cv patient is held out exactly once
    assert not set(f[f.role != "cv"].patient_id) & cv


@pytest.mark.skipif(not ATLAS.exists(), reason="dataset not present")
def test_frame_table_and_crops_match_geometry_on_real_data():
    if not (dataset.DERIVED / "fov.parquet").exists() or not (dataset.DERIVED / "manifest.parquet").exists():
        pytest.skip("derived/fov.parquet or manifest.parquet missing")
    t = dataset.frame_table("center1_labeled").iloc[:6]
    images, masks = dataset.build_crops(t, workers=2, log=lambda m: None)
    assert images.shape == (6, dataset.SIZE, dataset.SIZE, 3) and images.dtype == np.uint8 and images.flags.c_contiguous
    assert masks.shape == (6, dataset.SIZE, dataset.SIZE) and set(np.unique(masks)) <= {0, 1}
    assert (masks.reshape(6, -1).mean(1) > 0.005).all()               # no empty masks


# ---------------------------------------------------------------- epochs, queue order, exclusions (stubs, no GPU, no crops)

class Args:
    def __init__(self, epochs=None, workers=1, precision="fp32"):
        self.epochs, self.workers, self.precision = epochs, workers, precision


def write_lock(tmp_path, epochs):
    path = tmp_path / "analysis_lock.yaml"
    import yaml
    path.write_text(yaml.safe_dump({"epochs": epochs}))
    return path


def test_queue_epochs_from_argument_or_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "LOCK_PATH", tmp_path / "missing.yaml")
    assert train.queue_epochs(12) == {"unet": 12, "segformer": 12}
    with pytest.raises(SystemExit, match="--all needs"):
        train.queue_epochs(None)
    monkeypatch.setattr(train, "LOCK_PATH", write_lock(tmp_path, {"unet": 20, "segformer": 25}))
    assert train.queue_epochs(None) == {"unet": 20, "segformer": 25}
    assert train.queue_epochs(0) == {"unet": 20, "segformer": 25}           # 0 / absent means "use the lock"
    monkeypatch.setattr(train, "LOCK_PATH", write_lock(tmp_path, {"unet": 20}))
    with pytest.raises(SystemExit, match="--all needs"):                      # a half-filled lock is not enough
        train.queue_epochs(None)


def test_epochs_flag_may_not_override_the_frozen_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "LOCK_PATH", write_lock(tmp_path, {"unet": 20, "segformer": 25}))
    with pytest.raises(SystemExit, match="contradicts"):
        train.queue_epochs(10)
    with pytest.raises(SystemExit, match="contradicts"):
        train.check_epochs_against_lock(20, ["unet", "segformer"])           # right for unet, wrong for segformer
    train.check_epochs_against_lock(20, ["unet"])
    train.check_epochs_against_lock(25, ["segformer"])
    monkeypatch.setattr(train, "LOCK_PATH", write_lock(tmp_path, {"unet": 20}))
    train.check_epochs_against_lock(7, ["segformer"])                         # no lock entry for that model: no check
    with pytest.raises(SystemExit, match="contradicts"):
        train.check_epochs_against_lock(7, ["unet"])
    monkeypatch.setattr(train, "LOCK_PATH", tmp_path / "missing.yaml")
    train.check_epochs_against_lock(7, ["unet", "segformer"])                 # no lock at all: no check


def run_main(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["04_train.py"] + argv)
    calls = []
    monkeypatch.setattr(dataset, "load_center1", lambda workers, log: calls.append("load") or (None, None, None))
    monkeypatch.setattr(train, "train_run", lambda *a, **k: calls.append(("train", a[:4])))
    monkeypatch.setattr(train, "pick_device", lambda: "cpu")
    train.main()
    return calls


def test_single_run_epochs_check_and_the_pilot_exemption(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "LOCK_PATH", write_lock(tmp_path, {"unet": 20, "segformer": 25}))
    with pytest.raises(SystemExit, match="contradicts"):
        run_main(monkeypatch, ["--model", "segformer", "--fold", "0", "--epochs", "20"])
    assert run_main(monkeypatch, ["--model", "segformer", "--fold", "0", "--epochs", "25"])[1] == ("train", ("segformer", "0", 25, False))
    pilot = run_main(monkeypatch, ["--model", "segformer", "--fold", "0", "--epochs", "20", "--pilot"])
    assert pilot[1] == ("train", ("segformer", "0", 20, True))                # the pilot runs before any lock exists and is exempt


def run_queue(monkeypatch, tmp_path, existing, epochs=7, fail=None):
    models_dir, logs_dir = tmp_path / "models", tmp_path / "logs"             # logs_dir deliberately does not exist
    models_dir.mkdir(exist_ok=True)
    for name in existing:
        (models_dir / name).write_bytes(b"x")
    monkeypatch.setattr(train, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(train, "LOCK_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(train, "pick_device", lambda: "cpu")
    calls = []

    def loader(workers, log):
        calls.append("load")
        return None, None, None

    def stub(name, fold, ep, pilot, table, images, masks, device, precision, log):
        calls.append((name, fold, ep, pilot))
        if fail == (name, fold):
            raise RuntimeError("boom")

    monkeypatch.setattr(dataset, "load_center1", loader)
    monkeypatch.setattr(train, "train_run", stub)
    train.run_all(Args(epochs=epochs))
    return calls, logs_dir


def test_all_queue_order_and_skip_when_weights_exist(tmp_path, monkeypatch):
    calls, logs_dir = run_queue(monkeypatch, tmp_path, ["unet_fold0.pt", "segformer_fold3.pt", "unet_final.pt", "unet_fold1_pilot.pt"])
    assert calls[0] == "load"
    assert calls[1:] == [("unet", "1", 7, False), ("unet", "2", 7, False), ("unet", "3", 7, False), ("unet", "4", 7, False),
                         ("segformer", "0", 7, False), ("segformer", "1", 7, False), ("segformer", "2", 7, False),
                         ("segformer", "4", 7, False), ("segformer", "final", 7, False)]   # a pilot file does not count as done
    assert "queue: 9/12 runs to do" in (logs_dir / "train_all.log").read_text()               # logs dir was created on demand


def test_all_queue_with_everything_done_loads_nothing(tmp_path, monkeypatch):
    done = [f"{m}_{t}.pt" for m in models.MODEL_NAMES for t in ("fold0", "fold1", "fold2", "fold3", "fold4", "final")]
    calls, _ = run_queue(monkeypatch, tmp_path, done)
    assert calls == []


def test_all_queue_continues_after_a_failed_run_and_exits_nonzero(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as e:
        run_queue(monkeypatch, tmp_path, [], fail=("unet", "2"))
    assert e.value.code == 1


def test_make_logger_creates_its_directory(tmp_path):
    path = tmp_path / "a" / "b" / "run.log"
    train.make_logger(path)("hello")
    assert "hello" in path.read_text()


def fake_derived(tmp_path, found):
    ids = [f"F{i}" for i in range(len(found))]
    pd.DataFrame(dict(frame_id=ids, partition="center1_labeled", patient_id=[f"P{i % 3}" for i in range(len(found))])) \
        .to_parquet(tmp_path / "manifest.parquet")
    pd.DataFrame(dict(frame_id=ids, fov_found=found, fov_cx=1.0, fov_cy=1.0, fov_r=1.0)).to_parquet(tmp_path / "fov.parquet")
    pd.DataFrame(dict(patient_id=["P0", "P1", "P2"], fold=[0, 1, 2])).to_csv(tmp_path / "folds.csv", index=False)


def test_frame_table_drops_and_counts_frames_without_fov(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "DERIVED", tmp_path)
    fake_derived(tmp_path, [True, False, True, True, False, True])
    t = dataset.frame_table("center1_labeled")
    assert t.attrs["dropped_no_fov"] == 2 and len(t) == 4 and t.fov_found.all()
    lines = []
    dataset.log_exclusions(t, lines.append)
    assert lines == ["Center 1 labeled frames: 4 used, 2 excluded (no detected FOV)"]


def test_frame_table_refuses_to_keep_frames_without_fov(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "DERIVED", tmp_path)
    fake_derived(tmp_path, [True, False, True])
    monkeypatch.setitem(dataset.CFG["fov"], "exclude_not_found", False)
    with pytest.raises(ValueError, match="not a valid crop"):
        dataset.frame_table("center1_labeled")


def test_real_frame_table_drops_exactly_the_three_unfound_center1_frames():
    man, fovp = dataset.DERIVED / "manifest.parquet", dataset.DERIVED / "fov.parquet"
    if not (man.exists() and fovp.exists()):
        pytest.skip("derived/manifest.parquet or fov.parquet missing")
    t = dataset.frame_table("center1_labeled")           # reads the two tables only, builds no crops
    n_manifest = int((pd.read_parquet(man).partition == "center1_labeled").sum())
    assert t.attrs["dropped_no_fov"] == 3
    assert len(t) == n_manifest - 3 and t.fov_found.all()
