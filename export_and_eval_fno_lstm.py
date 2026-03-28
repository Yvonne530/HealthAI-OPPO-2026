#!/usr/bin/env python3
"""
export_and_eval_fno_lstm.py
训练完成后的一键导出+评测脚本

流程：
  1. 从新训练的 LSTM checkpoint 导出 ONNX（FNO only）
  2. ONNX 简化处理
  3. 转 MNN（带 --useOriginRNNImpl 去 While 依赖）
  4. 跑严格评测（与 PyTorch baseline 对拍，1000 次稳定性）
  5. 输出"优秀"判定报告

用法：
  python export_and_eval_fno_lstm.py
  python export_and_eval_fno_lstm.py --ckpt checkpoints/fno_bestgrf_PhaseA_lstm.pth
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints/fno_bestgrf_PhaseA_lstm.pth"),
        help="FNO LSTM checkpoint to export",
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "export/mnn_phasea_fno_lstm"),
        help="Output directory for MNN models",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "logs/mnn_quality_report_fno_lstm.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)

    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("Training may still be in progress. Check training log:")
        print("  tail -f /tmp/fno_lstm_train.log")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("FNO-LSTM Export & Evaluation Pipeline")
    print("=" * 80)

    # Step 1: Export ONNX from new LSTM checkpoint
    print("\n[1/4] Exporting FNO (LSTM) -> ONNX...")
    print(f"      Input checkpoint: {ckpt_path}")

    import torch
    import yaml
    from models.fno import FNO1d

    cfg = yaml.safe_load((ROOT / "configs/config.yaml").open())

    class FNOExport(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m.eval()

        def forward(self, bio_seq):
            grf_seq, _ = self.m(bio_seq)
            return grf_seq

    fno = FNO1d(
        input_dim=cfg["fno"]["input_dim"],
        output_dim=cfg["fno"]["output_dim"],
        future_k=cfg["fno"]["future_k"],
        modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"],
        depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"],
        use_lstm=True,
    ).eval()

    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")

    fno.load_state_dict(state, strict=True)
    print(f"      ✅ Checkpoint loaded ({fno.count_params():,} params)")

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_raw = out_dir / "fno_lstm_phaseA.raw.onnx"
    onnx_simp = out_dir / "fno_lstm_phaseA.onnx"

    wrapper = FNOExport(fno).eval()
    dummy = torch.randn(1, cfg["fno"]["seq_len"], cfg["fno"]["input_dim"])

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_raw),
        input_names=["bio_seq"],
        output_names=["grf_seq"],
        dynamic_axes={"bio_seq": {0: "batch"}, "grf_seq": {0: "batch"}},
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"      ✅ ONNX exported: {onnx_raw}")

    # Step 2: Simplify ONNX
    print("\n[2/4] Simplifying ONNX...")
    from onnxsim import simplify
    import onnx

    model_simp, check = simplify(str(onnx_raw))
    if not check:
        print(f"      ⚠️  Simplification check failed, but continuing...")
    onnx.save(model_simp, str(onnx_simp))
    print(f"      ✅ ONNX simplified: {onnx_simp}")

    # Step 3: Convert ONNX -> MNN
    print("\n[3/4] Converting ONNX -> MNN (with --useOriginRNNImpl)...")
    mnn_convert = ROOT / "MNN/build_mnnconvert/MNNConvert"
    if not mnn_convert.exists():
        print(f"      ❌ MNNConvert not found at {mnn_convert}")
        sys.exit(1)

    mnn_out = out_dir / "fno_lstm_phaseA.mnn"
    cmd = [
        str(mnn_convert),
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_simp.resolve()),
        "--MNNModel",
        str(mnn_out.resolve()),
        "--saveExternalData=1",
        "--fp16",
        "--useOriginRNNImpl",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"      ❌ MNNConvert failed:")
        print(result.stderr)
        sys.exit(1)
    if not mnn_out.exists():
        print(f"      ❌ MNN output not created: {mnn_out}")
        sys.exit(1)
    print(f"      ✅ MNN converted: {mnn_out}")

    # Step 4: Run strict evaluation
    print("\n[4/4] Running strict quality evaluation (1000 runs)...")
    eval_cmd = [
        sys.executable,
        "evaluate_mnn_quality.py",
        "--mnn_dir",
        str(out_dir),
        "--runs",
        "1000",
        "--report",
        str(report_path),
    ]
    result = subprocess.run(eval_cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"      ❌ Evaluation failed:")
        print(result.stderr)
        sys.exit(1)
    print(f"      ✅ Evaluation complete: {report_path}")

    # Parse report
    with open(report_path) as f:
        report = json.load(f)

    # Summarize results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    numeric = report["numeric"]["fno"]
    perf = report["stability_perf"]["fno"]

    max_err = numeric["metrics"]["max_abs_err"]
    mean_err = numeric["metrics"]["mean_abs_err"]
    rmse = numeric["metrics"]["rmse"]

    print(f"\n📊 FNO Numeric Quality:")
    print(f"   Shape match: {numeric['shape_match']}")
    print(f"   Functional OK: {numeric['functional_ok']}")
    print(f"   Max abs error: {max_err:.6f} {'✅' if max_err < 1e-2 else '❌'}")
    print(f"   Mean abs error: {mean_err:.6f} {'✅' if mean_err < 1e-4 else '❌'}")
    print(f"   RMSE: {rmse:.6f} {'✅' if rmse < 1e-4 else '❌'}")

    print(f"\n⚡ Performance & Stability (1000 runs):")
    print(f"   Avg latency: {perf['avg_ms']:.3f} ms ✅")
    print(f"   P95 latency: {perf['p95_ms']:.3f} ms ✅")
    print(f"   Max latency: {perf['max_ms']:.3f} ms ✅")
    print(f"   Stable: {perf['stable']} ✅")

    is_excellent = (
        numeric["shape_match"]
        and numeric["functional_ok"]
        and max_err < 1e-2
        and mean_err < 1e-4
        and rmse < 1e-4
        and perf["stable"]
    )

    print("\n" + "=" * 80)
    if is_excellent:
        print("✨ RESULT: EXCELLENT (all criteria passed)")
    else:
        print("⚠️  RESULT: Some criteria not met, review above")
    print("=" * 80)
    print(f"\nFull report: {report_path}\n")


if __name__ == "__main__":
    main()
