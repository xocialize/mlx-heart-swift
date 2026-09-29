#!/usr/bin/env python
"""x2_route.py — PORTING-SPEC H5, the ×2 route: native `heart_2x` (.clean2x, 512² → 1024²) against `.fidelity` ×4
(512² → 2048²) downsampled to 1024², both scored against the 2048² reference downsampled to 1024² (Lanczos, Pillow),
via vosrgate `pair` (SSIMULACRA2 · PSNR · octave SSIM on same-name pairs).

    python x2_route.py --bench <benchDir> --x4 <bench outDir with <cell>__HEART-fp16-64-whole.png> --x2 <x2 outDir> --out <dir>
"""
import argparse, glob, os, subprocess
from PIL import Image
p = argparse.ArgumentParser(); p.add_argument("--bench", required=True); p.add_argument("--x4", required=True)
p.add_argument("--x2", required=True); p.add_argument("--out", required=True); a = p.parse_args()
VG = "/Volumes/Satechi/Development/mlxengine-forge/Tools/vosrgate/.build/release/vosrgate"
ref_dir = os.path.join(a.out, "ref1024"); x4_dir = os.path.join(a.out, "fidelity_x4_down"); x2_dir = os.path.join(a.out, "clean2x_native")
for d in (ref_dir, x4_dir, x2_dir): os.makedirs(d, exist_ok=True)
cells = sorted(os.path.basename(f)[:-len("__low.png")] for f in glob.glob(os.path.join(a.bench, "*__low.png")))
for cell in cells:
    name = cell.split("__")[0]
    ref = Image.open(os.path.join(a.bench, f"{name}__reference.png")).convert("RGB")
    ref.resize((1024, 1024), Image.LANCZOS).save(os.path.join(ref_dir, f"{cell}.png"))
    x4 = Image.open(os.path.join(a.x4, f"{cell}__HEART-fp16-64-whole.png")).convert("RGB")
    x4.resize((1024, 1024), Image.LANCZOS).save(os.path.join(x4_dir, f"{cell}.png"))
    x2 = Image.open(os.path.join(a.x2, f"{cell}__HEART-fp16-64-whole.png")).convert("RGB")
    assert x2.size == (1024, 1024), x2.size
    x2.save(os.path.join(x2_dir, f"{cell}.png"))
for arm, d in (("fidelity_x4_down", x4_dir), ("clean2x_native", x2_dir)):
    csv = os.path.join(a.out, f"pair_{arm}.csv")
    subprocess.run([VG, "pair", ref_dir, d, csv], check=True)
    rows = [l.split(",") for l in open(csv).read().strip().split("\n")]
    hdr = rows[0]; fr_i = hdr.index("fr") if "fr" in hdr else 1
    vals = [float(r[fr_i]) for r in rows[1:] if r[fr_i] not in ("nan", "")]
    vals.sort()
    print(f"{arm:18} n={len(vals)} SSIMULACRA2 vs ref@1024: mean {sum(vals)/len(vals):.2f} median {vals[len(vals)//2]:.2f} min {vals[0]:.2f}  ({csv})")
