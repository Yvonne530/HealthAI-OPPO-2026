#!/usr/bin/env python3
"""
export_stgcn_fp32.py
====================
Re-exports STGCN in FP32 precision to fix the 0.079 mean error seen in FP16.

Mixed-precision strategy (engineering rationale):
  STGCN  → FP32  (input source; FP16 error propagates and amplifies)
  FNO    → FP16  (output averaged over 10 future frames; tolerant)
  Risk   → FP16  (classification head; inherently robust to small errors)

Usage:
  python export_stgcn_fp32.py \
    --ckpt checkpoints/stgcn_bestgrf_PhaseA.pth \
    --onnx_dir export/onnx_phasea_android_opt \
    --mnn_dir  export/mnn_phasea_android_opt \
    --mnnconvert MNN/build_mnnconvert/MNNConvert

This replaces stgcn_phaseA.mnn with a FP32 version.
All other models are left untouched.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import onnx
from onnxsim import simplify


# ── Minimal STGCN wrapper ─────────────────────────────────────────────────────
# Adjust the import to match your project structure

def load_stgcn(ckpt_path: str, device: str = "cpu"):
    """
    Load STGCN checkpoint.  Adjust the import below to match your project.
    """
    try:
        from models.stgcn import STGCN  # adjust path as needed
        from omegaconf import OmegaConf
        from configs.config import get_cfg

        cfg = get_cfg()  # load default config
        model = STGCN(
            num_nodes   = cfg.stgcn.num_nodes,    # 33
            num_frames  = cfg.stgcn.num_frames,   # 5
            output_dim  = cfg.stgcn.output_dim,   # 23
        )
    except ImportError:
        # Fallback: load raw state dict into checkpoint model
        print("[WARN] Could not import STGCN class. Loading raw checkpoint.")
        state = torch.load(ckpt_path, map_location=device)
        return state, None

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ERROR] Missing keys in STGCN checkpoint: {missing}")
        print("        FP32 export aborted — checkpoint / model mismatch.")
        sys.exit(1)
    if unexpected:
        print(f"[WARN] Unexpected keys (ignored): {unexpected}")

    model.to(device).eval()
    return model, state


class STGCNExportWrapper(nn.Module):
    """
    Wraps STGCN with fixed input/output tensor names matching Android contract.
      Input:  visual_seq [B, 5, 33, 3]
      Output: joint_angles [B, 23]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, visual_seq):          # [B, 5, 33, 3]
        return self.model(visual_seq)       # [B, 23]


# ── Export ────────────────────────────────────────────────────────────────────

def export_onnx(model_wrapped, onnx_path: Path, opset: int = 11):
    dummy = torch.zeros(1, 5, 33, 3)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model_wrapped, dummy, str(onnx_path),
        opset_version   = opset,
        input_names     = ["visual_seq"],
        output_names    = ["joint_angles"],
        dynamic_axes    = {"visual_seq": {0: "batch"}, "joint_angles": {0: "batch"}},
        do_constant_folding = True,
    )
    print(f"[ONNX] Exported to {onnx_path}")

    # Simplify (required by MNN for clean graph)
    model_onnx = onnx.load(str(onnx_path))
    model_sim, ok = simplify(model_onnx)
    if not ok:
        print("[ERROR] onnxsim failed — aborting FP32 export.")
        sys.exit(1)
    onnx.save(model_sim, str(onnx_path))
    onnx.checker.check_model(model_sim)
    print(f"[ONNX] Simplified & verified ✓")


def convert_mnn(onnx_path: Path, mnn_path: Path, mnnconvert: str):
    """
    Convert ONNX → MNN **without** --fp16 flag.
    This is intentional: FP32 for STGCN to preserve accuracy (error < 1e-5).
    """
    mnn_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path = str(mnn_path) + ".weight"

    cmd = [
        mnnconvert,
        "-f", "ONNX",
        "--modelFile",       str(onnx_path),
        "--MNNModel",        str(mnn_path),
        "--saveExternalData=1",
        # NOTE: NO --fp16 here — intentional for accuracy
    ]
    print(f"[MNN] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] MNNConvert failed:\n{result.stderr}")
        sys.exit(1)

    print(f"[MNN] Converted to {mnn_path}  (FP32, no quantization)")
    print(f"[MNN] Weight file: {weight_path}")
    return mnn_path, weight_path


def verify_precision(model_wrapped, mnn_path: Path):
    """
    Quick numerical parity check: compare PyTorch vs MNN output.
    Requires the MNN Python binding: pip install MNN
    """
    try:
        import MNN
        import numpy as np
    except ImportError:
        print("[SKIP] MNN Python binding not found — skipping precision check.")
        return

    dummy_np = np.random.randn(1, 5, 33, 3).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_np)

    with torch.no_grad():
        pt_out = model_wrapped(dummy_pt).numpy()

    # MNN inference
    net   = MNN.Interpreter(str(mnn_path))
    sess  = net.createSession()
    inp   = net.getSessionInput(sess, "visual_seq")
    inp.copyFromHostTensor(MNN.Tensor(
        [1, 5, 33, 3], MNN.Halide_Type_Float, dummy_np, MNN.Tensor_DimensionType_Caffe
    ))
    net.runSession(sess)
    out_t = net.getSessionOutput(sess, "joint_angles")
    mnn_out = np.array(out_t.getData()).reshape(1, 23)

    max_err  = float(np.abs(pt_out - mnn_out).max())
    mean_err = float(np.abs(pt_out - mnn_out).mean())
    rmse     = float(np.sqrt(((pt_out - mnn_out) ** 2).mean()))

    status = "✅" if max_err < 1e-4 else ("⚠️ " if max_err < 1e-2 else "❌")
    print(f"\n[PRECISION] STGCN FP32 vs PyTorch:")
    print(f"  Max error : {max_err:.2e}  {status}")
    print(f"  Mean error: {mean_err:.2e}")
    print(f"  RMSE      : {rmse:.2e}")
    if max_err > 1e-2:
        print("  [WARN] Error still high — check model architecture match.")
    net.releaseSession(sess); net.releaseModel()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True,  help="Path to stgcn_bestgrf_PhaseA.pth")
    ap.add_argument("--onnx_dir",    required=True,  help="ONNX output directory")
    ap.add_argument("--mnn_dir",     required=True,  help="MNN output directory")
    ap.add_argument("--mnnconvert",  required=True,  help="Path to MNNConvert binary")
    ap.add_argument("--opset",       default=11,     type=int)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    onnx_path = Path(args.onnx_dir) / "stgcn_phaseA_fp32.onnx"
    mnn_path  = Path(args.mnn_dir)  / "stgcn_phaseA.mnn"    # replaces existing file

    print("=" * 60)
    print("  STGCN FP32 Export  (混合精度策略: STGCN=FP32)")
    print("=" * 60)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  ONNX output: {onnx_path}")
    print(f"  MNN output : {mnn_path}  [REPLACES existing FP16 version]")
    print()

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}"); sys.exit(1)

    model, _ = load_stgcn(str(ckpt_path))
    if model is None:
        print("[ERROR] Could not load STGCN model."); sys.exit(1)

    wrapped = STGCNExportWrapper(model)

    # Step 1: ONNX
    export_onnx(wrapped, onnx_path, opset=args.opset)

    # Step 2: MNN (no --fp16)
    convert_mnn(onnx_path, mnn_path, args.mnnconvert)

    # Step 3: Precision check
    verify_precision(wrapped, mnn_path)

    print()
    print("=" * 60)
    print("  Done! Copy to Android assets/models/:")
    print(f"    {mnn_path}")
    print(f"    {mnn_path}.weight")
    print()
    print("  Expected precision: Max error < 1e-5 (FP32 parity)")
    print("  FNO + Risk remain FP16 — no change needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
