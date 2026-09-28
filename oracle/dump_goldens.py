#!/usr/bin/env python
"""dump_goldens.py — per-sub-op goldens for the HEART port from THE oracle: upstream `heart_arch.py`
(Phips/HEART 868878ce4c25) + the verbatim traiNNer-redux helpers (686293190aca) rebuilt by make_shim.sh, run on
torch CPU fp32 with a seeded uniform [0,1] input. The author's ONNX export under ONNX Runtime is the second oracle.

    PYTHONPATH=shim:pydeps python dump_goldens.py --weights upstream/models --onnx upstream/onnx --out goldens

Files (safetensors, float32, C-contiguous; 4-D activations stored NHWC = the port's layout, tokens (B, L, C) reshaped
to (B, H, W, C) — the same memory):
  taps_heart_4x_otf_v2_<HxW>.safetensors   sub-op taps at 64x64, 96x96, 100x140 (check_image_size reflect path →
                                           128x160), 20x20 (reflect, pad 12 < 20) and 12x12 (the REPLICATE fallback:
                                           pad 20 ≥ 12 — the plan's "20x20" case is a reflect case, see the note)
  e2e_<ckpt>_<HxW>.safetensors             `in`, `out` (torch CPU) and `onnx` (ORT CPU) at 128x128 and 512x512
                                           for every released checkpoint
Tables gated at tolerance 0 come first in the tap files: `rib_coords`, every attention block's `q_pos` / `k_pos`
(the (heads, N, rank) position features, fp32 as upstream computes them) and `g0b2.shiftpad` (the reflect-padded
tensor the shifted block attends over).
"""
import argparse, os, sys, time, json
import numpy as np, torch, torch.nn.functional as F
from safetensors.numpy import save_file
from safetensors.torch import load_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traiNNer.archs.heart_arch import heart

p = argparse.ArgumentParser(); p.add_argument("--weights", required=True); p.add_argument("--onnx", required=True)
p.add_argument("--out", required=True); p.add_argument("--skip-e2e", action="store_true"); p.add_argument("--only-e2e", action="store_true")
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.set_grad_enabled(False); torch.manual_seed(0)
CKPTS = {"heart_4x_otf_v2": 4, "heart_4x_otf_gan": 4, "heart_4x_pretrain": 4, "heart_2x": 2}

def load(name):
    m = heart(scale=CKPTS[name]).eval()
    m.load_state_dict(load_file(os.path.join(a.weights, f"{name}.safetensors")), strict=True); return m

def nhwc(t):  # (B,C,H,W) -> (B,H,W,C) float32 contiguous
    return np.ascontiguousarray(t.detach().permute(0, 2, 3, 1).float().numpy())
def tok(t, hw):  # (B,L,C) -> (B,H,W,C)
    b, l, c = t.shape; return np.ascontiguousarray(t.detach().reshape(b, hw[0], hw[1], c).float().numpy())
def npy(t): return np.ascontiguousarray(t.detach().float().numpy())

rng = np.random.default_rng(20260928)

def dump_taps(m, h, w):
    x = torch.from_numpy(rng.random((1, 3, h, w), dtype=np.float32))
    g = {"in": nhwc(x)}
    xp = m.check_image_size(x); g["padded"] = nhwc(xp); hw = (xp.shape[2], xp.shape[3])
    taps = {}
    def hook(name, fn):
        def _h(mod, inp, out): taps[name] = fn(inp, out)
        return _h
    hs = []
    L0 = m.layers[0].residual_group.blocks
    hs.append(m.conv_first.register_forward_hook(hook("conv_first", lambda i, o: nhwc(o))))
    hs.append(m.patch_embed.register_forward_hook(hook("patch_embed", lambda i, o: tok(o, hw))))
    hs.append(L0[0].norm1.register_forward_hook(hook("g0b0.norm1", lambda i, o: (tok(o[0], hw), npy(o[1].reshape(-1))))))
    hs.append(L0[0].conv_block.register_forward_hook(hook("g0b0.cab", lambda i, o: nhwc(o))))
    hs.append(L0[0].conv_block.cab[2].register_forward_hook(hook("g0b0.cab_pre_ca", lambda i, o: nhwc(o))))
    hs.append(L0[0].attn.register_forward_hook(hook("g0b0.attn", lambda i, o: nhwc(o))))
    hs.append(L0[0].norm2.register_forward_hook(hook("g0b0.norm2", lambda i, o: (tok(o[0], hw), npy(o[1].reshape(-1))))))
    hs.append(L0[0].mlp.register_forward_hook(hook("g0b0.mlp", lambda i, o: tok(o, hw))))
    hs.append(L0[0].register_forward_hook(hook("g0b0.out", lambda i, o: tok(o, hw))))
    hs.append(L0[1].register_forward_hook(hook("g0b1.out", lambda i, o: tok(o, hw))))
    hs.append(L0[2].attn.register_forward_hook(hook("g0b2.attn", lambda i, o: (nhwc(i[0]), nhwc(o)))))
    hs.append(L0[2].register_forward_hook(hook("g0b2.out", lambda i, o: tok(o, hw))))
    hs.append(m.layers[0].register_forward_hook(hook("g0.out", lambda i, o: tok(o, hw))))
    hs.append(m.layers[5].residual_group.blocks[4].attn.register_forward_hook(hook("g5b4.attn", lambda i, o: nhwc(o))))
    hs.append(m.norm.register_forward_hook(hook("features", lambda i, o: tok(o, hw))))
    hs.append(m.conv_after_body.register_forward_hook(hook("conv_after_body", lambda i, o: nhwc(o))))
    hs.append(m.conv_before_upsample.register_forward_hook(hook("before_upsample", lambda i, o: (nhwc(i[0]), nhwc(o)))))
    hs.append(m.upsample.register_forward_hook(hook("upsample", lambda i, o: nhwc(o))))
    hs.append(m.conv_last.register_forward_hook(hook("conv_last", lambda i, o: nhwc(o))))
    t0 = time.perf_counter(); y = m(x); dt = time.perf_counter() - t0
    for hh in hs: hh.remove()
    g["out"] = nhwc(y)
    g["conv_first"] = taps["conv_first"]; g["patch_embed"] = taps["patch_embed"]
    g["g0b0.norm1.x"], g["g0b0.norm1.std"] = taps["g0b0.norm1"]
    g["g0b0.cab"] = taps["g0b0.cab"]; g["g0b0.cab_pre_ca"] = taps["g0b0.cab_pre_ca"]; g["g0b0.attn"] = taps["g0b0.attn"]
    g["g0b0.norm2.x"], g["g0b0.norm2.std"] = taps["g0b0.norm2"]
    g["g0b0.mlp"] = taps["g0b0.mlp"]; g["g0b0.out"] = taps["g0b0.out"]; g["g0b1.out"] = taps["g0b1.out"]
    attn_in, g["g0b2.attn"] = taps["g0b2.attn"]
    # the reflect-padded shift tensor the shifted block attends over (F.pad reflect 16 each side)
    g["g0b2.shiftpad"] = nhwc(F.pad(torch.from_numpy(attn_in).permute(0, 3, 1, 2), (16, 16, 16, 16), mode="reflect"))
    g["g0b2.out"] = taps["g0b2.out"]; g["g0.out"] = taps["g0.out"]; g["g5b4.attn"] = taps["g5b4.attn"]
    g["features"] = taps["features"]; g["conv_after_body"] = taps["conv_after_body"]
    g["after_body_residual"], g["before_upsample"] = taps["before_upsample"]
    g["upsample"] = taps["upsample"]; g["conv_last"] = taps["conv_last"]
    # tables: rib_coords (shared) and every attention block's position features, exactly as forward computes them
    for gi, layer in enumerate(m.layers):
        for bi, blk in enumerate(layer.residual_group.blocks):
            if not blk.use_attention: continue
            at = blk.attn
            if gi == 0 and bi == 0: g["rib_coords"] = npy(at.rib_coords)
            inter = F.relu(at.rib_coords.float() @ at.to_hidden.float() + at.hidden_b.float())
            g[f"g{gi}b{bi}.q_pos"] = npy(torch.einsum("nd,hdr->hnr", inter, at.to_q.float()))
            g[f"g{gi}b{bi}.k_pos"] = npy(torch.einsum("nd,hdr->hnr", inter, at.to_k.float()))
    path = os.path.join(a.out, f"taps_heart_4x_otf_v2_{h}x{w}.safetensors")
    save_file(g, path)
    print(f"taps {h}x{w} (padded {hw[0]}x{hw[1]}, {'replicate' if (32-h%32)%32 >= h or (32-w%32)%32 >= w else 'reflect'}): "
          f"{len(g)} arrays, out {tuple(y.shape)}, {dt:.1f} s -> {path}", flush=True)

if not a.only_e2e:
    m = load("heart_4x_otf_v2")
    for (h, w) in [(64, 64), (96, 96), (100, 140), (20, 20), (12, 12)]:
        dump_taps(m, h, w)

if not a.skip_e2e:
    import onnxruntime as ort
    for name, scale in CKPTS.items():
        m = load(name)
        sess = ort.InferenceSession(os.path.join(a.onnx, f"{name}_fp32_op17.onnx"), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        for side in (128, 512):
            x = torch.from_numpy(rng.random((1, 3, side, side), dtype=np.float32))
            t0 = time.perf_counter(); y = m(x); dt = time.perf_counter() - t0
            t1 = time.perf_counter(); yo = sess.run(None, {iname: x.numpy()})[0]; do = time.perf_counter() - t1
            d = np.abs(y.numpy() - yo); mse = float((d ** 2).mean())
            path = os.path.join(a.out, f"e2e_{name}_{side}x{side}.safetensors")
            save_file({"in": nhwc(x), "out": nhwc(y), "onnx": nhwc(torch.from_numpy(yo))}, path)
            print(f"e2e {name} {side}² -> {tuple(y.shape[2:])}: torch {dt:.1f} s, ort {do:.1f} s, torch-vs-onnx max|d| {d.max():.2e} "
                  f"PSNR {10*np.log10(1/mse):.1f} dB -> {path}", flush=True)
