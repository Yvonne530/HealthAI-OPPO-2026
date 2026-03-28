#!/usr/bin/env python3
"""
Export PhaseA checkpoints (.pth) to ONNX and then convert to MNN.

Usage:
  python export_phasea_to_mnn.py
  python export_phasea_to_mnn.py --onnx_dir export/onnx_phasea --mnn_dir export/mnn_phasea
"""

import argparse
import os
import subprocess
from pathlib import Path

import torch
import yaml
from onnxsim import simplify


ROOT = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_load_state_dict(model: torch.nn.Module, ckpt_path: Path, strict: bool = True):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    return model.load_state_dict(state, strict=strict)


def _ckpt_has_lstm_keys(ckpt_path: Path) -> bool:
    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    return any(k.startswith("lstm.") or k.startswith("lstm_head.") for k in state.keys())


def export_stgcn(cfg: dict, ckpt: Path, out_path: Path) -> None:
    from models.stgcn import STGCN

    class STGCNExport(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m.eval()

        def forward(self, visual_seq: torch.Tensor) -> torch.Tensor:
            ja, _ = self.m(visual_seq)
            return ja

    model = STGCN(
        num_nodes=cfg["stgcn"]["num_nodes"],
        in_channels=cfg["stgcn"]["in_channels"],
        hidden_channels=cfg["stgcn"]["hidden_channels"],
        num_frames=cfg["stgcn"]["num_frames"],
        output_dim=cfg["stgcn"]["output_dim"],
        dropout=cfg["stgcn"]["dropout"],
    ).eval()
    _safe_load_state_dict(model, ckpt, strict=True)

    wrapper = STGCNExport(model).eval()
    dummy = torch.randn(1, cfg["stgcn"]["num_frames"], cfg["stgcn"]["num_nodes"], 3)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["visual_seq"],
        output_names=["joint_angles"],
        dynamic_axes={"visual_seq": {0: "batch"}, "joint_angles": {0: "batch"}},
        opset_version=11,
        do_constant_folding=True,
    )


def export_fno(cfg: dict, ckpt: Path, out_path: Path, use_lstm: bool) -> None:
    from models.fno import FNO1d

    class FNOExport(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m.eval()

        def forward(self, bio_seq: torch.Tensor) -> torch.Tensor:
            grf_seq, _ = self.m(bio_seq)
            return grf_seq

    model = FNO1d(
        input_dim=cfg["fno"]["input_dim"],
        output_dim=cfg["fno"]["output_dim"],
        future_k=cfg["fno"].get("future_k", cfg["data"].get("future_k", 10)),
        modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"],
        depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"],
        use_lstm=use_lstm,
    ).eval()

    if use_lstm:
        load_res = _safe_load_state_dict(model, ckpt, strict=False)
        if load_res.missing_keys:
            raise ValueError(
                "FNO checkpoint does not contain LSTM branch weights. "
                f"Missing keys sample: {load_res.missing_keys[:6]}. "
                "Use --fno_use_lstm false, or provide an LSTM-trained FNO checkpoint."
            )
    else:
        _safe_load_state_dict(model, ckpt, strict=True)

    wrapper = FNOExport(model).eval()
    dummy = torch.randn(1, cfg["fno"]["seq_len"], cfg["fno"]["input_dim"])

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["bio_seq"],
        output_names=["grf_seq"],
        dynamic_axes={"bio_seq": {0: "batch"}, "grf_seq": {0: "batch"}},
        opset_version=11,
        do_constant_folding=True,
    )


def export_risk(cfg: dict, ckpt: Path, out_path: Path, default_hr: float = 70.0, default_sleep: float = 80.0) -> None:
    from models.risk_model import RiskMLP

    class RiskExport(torch.nn.Module):
        def __init__(self, m: torch.nn.Module, hr: float, sleep: float):
            super().__init__()
            self.m = m.eval()
            self.hr = float(hr)
            self.sleep = float(sleep)

        def forward(self, risk_seq: torch.Tensor):
            b = risk_seq.shape[0]
            hr_t = torch.full((b, 1), self.hr, dtype=risk_seq.dtype, device=risk_seq.device)
            sl_t = torch.full((b, 1), self.sleep, dtype=risk_seq.dtype, device=risk_seq.device)
            logits, conf = self.m(risk_seq, hr_t, sl_t)
            return logits, conf

    model = RiskMLP(
        input_dim=cfg["risk"]["input_dim"],
        hidden_dim=cfg["risk"]["hidden_dim"],
        num_classes=cfg["risk"]["num_classes"],
        seq_len=cfg["risk"].get("seq_len", 20),
        use_physio=cfg["risk"].get("use_physio", True),
    ).eval()
    _safe_load_state_dict(model, ckpt, strict=True)

    wrapper = RiskExport(model, default_hr, default_sleep).eval()
    dummy = torch.randn(1, cfg["risk"].get("seq_len", 20), cfg["risk"]["input_dim"])

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["risk_seq"],
        output_names=["risk_logits", "risk_confidence"],
        dynamic_axes={
            "risk_seq": {0: "batch"},
            "risk_logits": {0: "batch"},
            "risk_confidence": {0: "batch"},
        },
        opset_version=11,
        do_constant_folding=True,
    )


def simplify_onnx_model(src: Path, dst: Path) -> None:
    model_simp, check = simplify(str(src))
    if not check:
        raise RuntimeError(f"onnx-simplifier failed consistency check: {src}")
    import onnx
    onnx.save(model_simp, str(dst))


def convert_onnx_to_mnn(
    mnnconvert: Path,
    onnx_path: Path,
    mnn_path: Path,
    quant_bits: int = 0,
    extra_args: list[str] | None = None,
    fp16: bool = False,
) -> None:
    onnx_path = onnx_path.resolve()
    mnn_path = mnn_path.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found for conversion: {onnx_path}")

    cmd = [
        str(mnnconvert),
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_path),
        "--MNNModel",
        str(mnn_path),
        "--saveExternalData=1",
    ]
    if quant_bits > 0:
        cmd.append(f"--weightQuantBits={quant_bits}")

    if fp16:
        cmd.append("--fp16")

    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, check=True)
    if not mnn_path.exists():
        raise RuntimeError(f"MNNConvert did not produce output: {mnn_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PhaseA .pth to ONNX and MNN")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--onnx_dir", default=str(ROOT / "export" / "onnx_phasea"))
    parser.add_argument("--mnn_dir", default=str(ROOT / "export" / "mnn_phasea"))
    parser.add_argument("--mnnconvert", default=str(ROOT / "MNN" / "build_mnnconvert" / "MNNConvert"))
    parser.add_argument("--quant_bits", type=int, default=0)
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier before MNN conversion")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 optimization in MNNConvert")
    parser.add_argument(
        "--fno_use_lstm",
        choices=["auto", "true", "false"],
        default="auto",
        help="Select FNO export branch. auto: infer from checkpoint keys.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    ckpt_dir = Path(args.ckpt_dir).resolve()
    onnx_dir = Path(args.onnx_dir).resolve()
    mnn_dir = Path(args.mnn_dir).resolve()
    mnnconvert = Path(args.mnnconvert).resolve()

    onnx_dir.mkdir(parents=True, exist_ok=True)
    mnn_dir.mkdir(parents=True, exist_ok=True)

    if not mnnconvert.exists():
        raise FileNotFoundError(f"MNNConvert not found: {mnnconvert}")

    stgcn_ckpt = ckpt_dir / "stgcn_bestgrf_PhaseA.pth"
    fno_ckpt = ckpt_dir / "fno_bestgrf_PhaseA.pth"
    risk_ckpt = ckpt_dir / "risk_bestgrf_PhaseA.pth"

    if args.fno_use_lstm == "auto":
        fno_use_lstm = _ckpt_has_lstm_keys(fno_ckpt)
    else:
        fno_use_lstm = args.fno_use_lstm == "true"

    stgcn_onnx_raw = onnx_dir / "stgcn_phaseA.raw.onnx"
    fno_onnx_raw = onnx_dir / "fno_lstm_phaseA.raw.onnx"
    risk_onnx_raw = onnx_dir / "risk_phaseA.raw.onnx"

    stgcn_onnx = onnx_dir / "stgcn_phaseA.onnx"
    fno_onnx = onnx_dir / "fno_lstm_phaseA.onnx"
    risk_onnx = onnx_dir / "risk_phaseA.onnx"

    stgcn_mnn = mnn_dir / "stgcn_phaseA.mnn"
    fno_mnn = mnn_dir / "fno_lstm_phaseA.mnn"
    risk_mnn = mnn_dir / "risk_phaseA.mnn"

    print("[1/6] Export STGCN -> ONNX")
    export_stgcn(cfg, stgcn_ckpt, stgcn_onnx_raw)
    print(f"  OK: {stgcn_onnx_raw}")

    print(f"[2/6] Export FNO(use_lstm={fno_use_lstm}) -> ONNX")
    export_fno(cfg, fno_ckpt, fno_onnx_raw, use_lstm=fno_use_lstm)
    print(f"  OK: {fno_onnx_raw}")

    print("[3/6] Export Risk -> ONNX")
    export_risk(cfg, risk_ckpt, risk_onnx_raw)
    print(f"  OK: {risk_onnx_raw}")

    if args.simplify:
        print("[3.1/6] Simplify ONNX models")
        simplify_onnx_model(stgcn_onnx_raw, stgcn_onnx)
        simplify_onnx_model(fno_onnx_raw, fno_onnx)
        simplify_onnx_model(risk_onnx_raw, risk_onnx)
    else:
        stgcn_onnx_raw.replace(stgcn_onnx)
        fno_onnx_raw.replace(fno_onnx)
        risk_onnx_raw.replace(risk_onnx)
    print(f"  OK: {stgcn_onnx}")
    print(f"  OK: {fno_onnx}")
    print(f"  OK: {risk_onnx}")

    print("[4/6] Convert STGCN ONNX -> MNN")
    convert_onnx_to_mnn(mnnconvert, stgcn_onnx, stgcn_mnn, quant_bits=args.quant_bits, fp16=args.fp16)
    print(f"  OK: {stgcn_mnn}")

    print("[5/6] Convert FNO ONNX -> MNN")
    # Avoid While-subgraph dependency for LSTM export path.
    fno_extra = ["--useOriginRNNImpl"] if fno_use_lstm else None
    convert_onnx_to_mnn(
        mnnconvert,
        fno_onnx,
        fno_mnn,
        quant_bits=args.quant_bits,
        extra_args=fno_extra,
        fp16=args.fp16,
    )
    print(f"  OK: {fno_mnn}")

    print("[6/6] Convert Risk ONNX -> MNN")
    convert_onnx_to_mnn(mnnconvert, risk_onnx, risk_mnn, quant_bits=args.quant_bits, fp16=args.fp16)
    print(f"  OK: {risk_mnn}")

    print("\nDone.")
    print("ONNX outputs:")
    print(f"  - {stgcn_onnx}")
    print(f"  - {fno_onnx}")
    print(f"  - {risk_onnx}")
    print("MNN outputs:")
    print(f"  - {stgcn_mnn}")
    print(f"  - {fno_mnn}")
    print(f"  - {risk_mnn}")


if __name__ == "__main__":
    main()
