import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1 / "python"))
import dataset


def main():
    t = dataset.frame_table("center1_labeled").sample(16, random_state=1).reset_index(drop=True)
    imgs, _ = dataset.build_crops(t, workers=8, log=lambda *a: None)
    ref = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255)
    ident = torch.tensor([[1., 0., 0.], [0., 1., 0.]]).unsqueeze(0).repeat(16, 1, 1)
    size = list(ref.shape)
    g_cpu = F.affine_grid(ident, size, align_corners=False)
    g_mps = F.affine_grid(ident.to("mps"), size, align_corners=False)
    print("grid cpu vs mps max|diff|:", float((g_cpu - g_mps.cpu()).abs().max()), " shapes", tuple(g_cpu.shape), tuple(g_mps.shape))
    x = ref.to("mps")
    cpu_out = F.grid_sample(ref, g_cpu, mode="bilinear", padding_mode="zeros", align_corners=False)
    for label, g in (("cpu grid -> mps", g_cpu.to("mps")), ("mps grid -> mps", g_mps)):
        o = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False).cpu()
        print(f"{label:16s} out-mean {float(o.mean()):.4f} (cpu {float(cpu_out.mean()):.4f}) max|diff| {float((o - cpu_out).abs().max()):.4f}")
    print("grid mps contiguous:", g_mps.is_contiguous(), "strides", g_mps.stride())
    # does a contiguous copy of the MPS grid fix it?
    o = F.grid_sample(x, g_mps.contiguous().clone(), mode="bilinear", padding_mode="zeros", align_corners=False).cpu()
    print(f"mps grid .clone()  out-mean {float(o.mean()):.4f} max|diff| {float((o - cpu_out).abs().max()):.4f}")


if __name__ == "__main__":
    main()
