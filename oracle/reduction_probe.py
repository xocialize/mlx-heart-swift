#!/usr/bin/env python
"""reduction_probe.py — how accurate are MLX's mean/var over an i-LN-sized tensor (H·W·180 elements) on the CPU and GPU
streams, against float64 truth and torch CPU (double-accumulated Welford)? And does a two-stage (chunked) sum fix it?
Run: PYTHONPATH=mlxdeps python reduction_probe.py"""
import numpy as np, mlx.core as mx, torch
rng = np.random.default_rng(0)
print(f"mlx {mx.__version__}, torch {torch.__version__}")
for n_hw, label in [(32*32, "32² (20×20/12×12 padded)"), (64*64, "64²"), (128*160, "128×160 (100×140 padded)"), (512*512, "512²")]:
    N = n_hw * 180
    x = (rng.random(N, dtype=np.float32) * 2 - 1).astype(np.float32) + np.float32(0.3)
    x64 = x.astype(np.float64); m64 = x64.mean(); v64 = ((x64 - m64) ** 2).mean(); s64 = np.sqrt(v64 + 1e-4)
    xt = torch.from_numpy(x); mt = xt.mean().item(); st = np.sqrt(xt.var(unbiased=False).item() + 1e-4)
    print(f"N = {N:>10,} ({label}): torch-CPU mean rel {abs(mt-m64)/abs(m64):.1e} std rel {abs(st-s64)/s64:.1e}")
    for dev, name in [(mx.cpu, "cpu"), (mx.gpu, "gpu")]:
        xm = mx.array(x)
        with mx.stream(dev):
            mm = xm.mean().item(); sm = np.sqrt(mx.var(xm).item() + 1e-4)
            c = xm.reshape(-1, 4096) if N % 4096 == 0 else xm.reshape(-1, 1024)
            mc = c.sum(-1).sum(-1).item() / N; sc = np.sqrt(((c - mc) ** 2).sum(-1).sum(-1).item() / N + 1e-4)
        print(f"    mlx {name}: plain mean rel {abs(mm-m64)/abs(m64):.1e} std rel {abs(sm-s64)/s64:.1e} | two-stage ({c.shape[-1]}) mean rel {abs(mc-m64)/abs(m64):.1e} std rel {abs(sc-s64)/s64:.1e}")
