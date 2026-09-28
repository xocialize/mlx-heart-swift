#!/usr/bin/env python
"""ort_gap_probe.py — why does the author's ONNX (ORT CPU) drift from torch CPU with image size (90 dB at 128²,
54 dB at 512²)? Hypothesis: ORT's ReduceMean in i-LN accumulates naively in fp32 over H·W·C elements. Test: torch CPU
vs ORT at 128², 256², 384² on the fidelity checkpoint, plus torch CPU vs torch MPS (tree reductions) as a control."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traiNNer.archs.heart_arch import heart
from safetensors.torch import load_file
import onnxruntime as ort
torch.set_grad_enabled(False)
m = heart(scale=4).eval(); m.load_state_dict(load_file("upstream/models/heart_4x_otf_v2.safetensors"), strict=True)
sess = ort.InferenceSession("upstream/onnx/heart_4x_otf_v2_fp32_op17.onnx", providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
rng = np.random.default_rng(20260928)
def psnr(a, b): mse = float(np.mean((a - b) ** 2)); return 10 * np.log10(1 / mse)
for side in (128, 256, 384):
    x = rng.random((1, 3, side, side), dtype=np.float32)
    y = m(torch.from_numpy(x)).numpy(); yo = sess.run(None, {iname: x})[0]
    print(f"{side}²: torch-CPU vs ORT-CPU PSNR {psnr(y, yo):.1f} dB (max|d| {np.abs(y-yo).max():.2e}), N_iln = {side*side*180:,}", flush=True)
