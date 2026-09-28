#!/usr/bin/env python
"""convert_weights.py — MLX-layout weights for mlx-heart-swift from the pinned Phips/HEART checkpoints.

    python convert_weights.py --src upstream/models --out weights

Writes, per checkpoint, `<name>_fp32.safetensors` (the parity lane) and `<name>_fp16.safetensors` (the shipping lane),
plus one `config.json` describing the architecture, the variants and the source pins.

Key contract (S0): the keys are the UPSTREAM state-dict keys, unchanged — 748 tensors per ×4 checkpoint, 746 for ×2
(the ×2 `Upsample` has one conv). Layout: every 4-D tensor is a Conv2d weight and is transposed PyTorch (O, I, kH, kW)
→ MLX (O, kH, kW, I); everything else is stored as-is (`Linear` (out, in); RIB `to_hidden` (42, 32), `hidden_b`
(1, 32), `to_q`/`to_k` (heads, 32, 8)). No derived tensors: `rib_coords` and the i-LN `std_ema` buffer are
non-persistent upstream and are rebuilt / never needed by the port.

fp16 lane dtype rule: the conv / linear weights and biases are fp16; the i-LN and `AffineTransform` weight/bias and
the four RIB implicit-net parameters stay fp32 (upstream computes the RIB features in fp32 explicitly, and the port
runs i-LN's statistics and affine in fp32) — 220 fp32 + 528 fp16 tensors per ×4 file.
Every array is materialised through numpy (C-contiguous) before saving — no lazy tensors.
"""
import argparse, hashlib, json, os, re
import numpy as np
import torch
from safetensors.torch import load_file
from safetensors.numpy import save_file

REVISION = "868878ce4c253a8061300f923b620fdc6edf090a"
CKPTS = {
    "heart_4x_otf_v2":   {"scale": 4, "variant": "fidelity", "role": "4x OTF fidelity (default)"},
    "heart_4x_otf_gan":  {"scale": 4, "variant": "sharp",    "role": "4x OTF GAN"},
    "heart_4x_pretrain": {"scale": 4, "variant": "clean",    "role": "4x official pretrain (clean)"},
    "heart_2x":          {"scale": 2, "variant": "clean2x",  "role": "2x official pretrain (clean)"},
}
KEEP_FP32 = re.compile(r"(^|\.)(norm1|norm2|norm)\.(weight|bias)$|(^|\.)(to_hidden|hidden_b|to_q|to_k)$")

p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

config = {
    "architecture": "HEART", "embed_dim": 180, "depths": [6] * 6, "num_heads": [6] * 6, "window_size": 32,
    "compress_ratio": 3, "squeeze_factor": 30, "conv_scale": 0.01, "mlp_ratio": 2.0, "num_feat": 64,
    "rank": 8, "rib_hidden_dim": 32, "rib_n_freqs": 10, "attention_freq": 2, "iln_eps": 1e-4,
    "layout": {"conv_weight": "OHWI", "activations": "NHWC"},
    "fp16_keeps_fp32": ["*.norm1.{weight,bias}", "*.norm2.{weight,bias}", "norm.{weight,bias}",
                        "patch_embed.norm.{weight,bias}", "*.to_hidden", "*.hidden_b", "*.to_q", "*.to_k"],
    "source": {"repo": "Phips/HEART", "revision": REVISION, "license": "apache-2.0",
               "training_data": "Phips/lucid-cc0-v2-hc-512 (CC0)"},
    "variants": {},
}
for name, meta in CKPTS.items():
    src = os.path.join(a.src, f"{name}.safetensors")
    sd = load_file(src)
    out32, out16, n_conv = {}, {}, 0
    for k, v in sd.items():
        if v.ndim == 4:
            v = v.permute(0, 2, 3, 1); n_conv += 1
        arr = np.ascontiguousarray(v.detach().cpu().float().numpy())
        out32[k] = arr
        out16[k] = arr if KEEP_FP32.search(k) else np.ascontiguousarray(arr.astype(np.float16))
    n_params = int(sum(v.size for v in out32.values()))
    expect_conv = 192 if meta["scale"] == 4 else 191   # 6 top-level + 6 groups × (24 CAB + 6 attn + 1)
    assert n_conv == expect_conv, (name, n_conv)
    assert len(out32) == (748 if meta["scale"] == 4 else 746), (name, len(out32))
    n_kept = sum(1 for k in out16 if out16[k].dtype == np.float32)
    f32 = os.path.join(a.out, f"{name}_fp32.safetensors"); f16 = os.path.join(a.out, f"{name}_fp16.safetensors")
    save_file(out32, f32); save_file(out16, f16)
    config["variants"][name] = {
        "scale": meta["scale"], "variant": meta["variant"], "role": meta["role"], "tensors": len(out32),
        "params": n_params, "conv_tensors": n_conv, "fp16_file_fp32_tensors": n_kept,
        "source_file": f"models/{name}.safetensors", "source_sha256": sha256(src),
        "fp32_sha256": sha256(f32), "fp16_sha256": sha256(f16),
        "fp32_bytes": os.path.getsize(f32), "fp16_bytes": os.path.getsize(f16),
    }
    print(f"{name}: {len(out32)} tensors, {n_params:,} params, {n_conv} convs transposed, fp16 file keeps {n_kept} fp32 "
          f"| fp32 {os.path.getsize(f32)/1e6:.1f} MB, fp16 {os.path.getsize(f16)/1e6:.1f} MB")
json.dump(config, open(os.path.join(a.out, "config.json"), "w"), indent=1)
print("wrote", os.path.join(a.out, "config.json"))
