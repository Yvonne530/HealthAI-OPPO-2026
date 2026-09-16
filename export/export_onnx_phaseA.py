#!/usr/bin/env python3
"""
export/export_onnx_phaseA.py
导出 PhaseA 训练结果为 ONNX 文件（跳过 PhaseB 训练）

使用方式：
  python export/export_onnx_phaseA.py --phase A --output /path/to/onnx
"""
import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)


def export_stgcn(model, cfg: dict, out_dir: str) -> str:
    """导出 ST-GCN 模型"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stgcn.onnx")
    T, N = cfg["stgcn"]["num_frames"], cfg["stgcn"]["num_nodes"]
    dummy = torch.randn(1, T, N, 3)

    torch.onnx.export(
        model, dummy, path,
        input_names=["visual_seq"],
        output_names=["joint_angles"],
        dynamic_axes={"visual_seq": {0: "batch"},
                      "joint_angles": {0: "batch"}},
        opset_version=12,  # 支持 einsum 操作
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"✅ ST-GCN ONNX: {path} ({size_mb:.1f} MB)")
    return path


def export_fno(model, cfg: dict, out_dir: str) -> str:
    """导出 FNO 模型 (LSTM 降级版，避免 FFT 算子)"""
    from models.fno import FNO1d
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fno_lstm.onnx")
    T, D = cfg["fno"]["seq_len"], cfg["fno"]["input_dim"]
    dummy = torch.randn(1, T, D)

    # 创建 LSTM 版本的 FNO（NPU 兼容性更好）
    fno_lstm = FNO1d(
        input_dim=cfg["fno"]["input_dim"],
        output_dim=cfg["fno"]["output_dim"],
        modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"],
        depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"],
        use_lstm=True,  # 强制 LSTM，避免 FFT 算子不兼容
    ).eval()

    # 复制权重
    if hasattr(model, "lstm"):
        fno_lstm.load_state_dict(model.state_dict(), strict=False)

    torch.onnx.export(
        fno_lstm, dummy, path,
        input_names=["bio_seq"],
        output_names=["grf_seq"],
        dynamic_axes={"bio_seq": {0: "batch"},
                      "grf_seq": {0: "batch"}},
        opset_version=12,  # 支持 einsum 操作
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"✅ FNO(LSTM) ONNX: {path} ({size_mb:.1f} MB)")
    return path


def export_risk(model, cfg: dict, out_dir: str) -> str:
    """导出 Risk 模型（带生理特征融合）"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "risk.onnx")
    D = cfg["risk"]["input_dim"]
    T = cfg["risk"].get("seq_len", 20)
    
    # Risk 模型的实际输入包含生理特征（hr + sleep_score）
    # 创建包装类，使得 ONNX 输入维度符合期望
    class RiskWrapper(nn.Module):
        def __init__(self, risk_model):
            super().__init__()
            self.risk_model = risk_model
        
        def forward(self, risk_feat):
            # risk_feat: (B, T, 35)，手动添加默认生理特征
            B, T, _ = risk_feat.shape
            hr_default = torch.full((B, 1), 70.0, device=risk_feat.device)
            sl_default = torch.full((B, 1), 80.0, device=risk_feat.device)
            logits, conf = self.risk_model(risk_feat, hr_default, sl_default)
            return logits
    
    wrapper = RiskWrapper(model).eval()
    dummy = torch.randn(1, T, D)
    
    torch.onnx.export(
        wrapper, dummy, path,
        input_names=["risk_feat"],
        output_names=["logits"],
        dynamic_axes={"risk_feat": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=12,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"✅ Risk ONNX: {path} ({size_mb:.1f} MB)")
    return path


def verify_onnx(path: str, dummy: torch.Tensor) -> bool:
    """验证 ONNX 模型可执行性"""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: dummy.numpy()})
        logger.info(f"  验证通过: 输入={dummy.shape} → 输出={out[0].shape}")
        return True
    except ImportError:
        logger.warning("  ⚠️  onnxruntime 未安装，跳过验证 (pip install onnxruntime)")
        return False
    except Exception as e:
        logger.error(f"  ❌ 验证失败: {e}")
        return False


def export_phaseA(cfg: dict, phase: str = "A", output_dir: str = None) -> None:
    """
    导出指定 Phase 的模型为 ONNX
    
    Args:
        cfg: 配置字典
        phase: "A" 或 "B"
        output_dir: 输出目录（默认为 export/onnx_phaseX）
    """
    from models.stgcn import STGCN
    from models.fno import FNO1d
    from models.risk_model import RiskMLP

    ckpt_dir = cfg["train"]["checkpoint_dir"]
    phase_suffix = f"grf_Phase{phase}"
    
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            f"onnx_phase{phase.lower()}"
        )

    device = torch.device("cpu")

    logger.info(f"\n{'='*60}")
    logger.info(f"导出 Phase{phase} 模型为 ONNX")
    logger.info(f"{'='*60}\n")

    # ======================== ST-GCN ========================
    logger.info(f"[1/3] 加载 ST-GCN...")
    stgcn = STGCN(
        num_nodes=cfg["stgcn"]["num_nodes"],
        hidden_channels=cfg["stgcn"]["hidden_channels"],
        output_dim=cfg["stgcn"]["output_dim"],
    ).eval()

    stgcn_ck = os.path.join(ckpt_dir, f"stgcn_best{phase_suffix}.pth")
    if os.path.exists(stgcn_ck):
        stgcn.load_state_dict(torch.load(stgcn_ck, map_location=device))
        logger.info(f"  已加载: {stgcn_ck}")
    else:
        logger.warning(f"  ⚠️  {stgcn_ck} 不存在，使用随机初始化")

    stgcn.to(device)
    p1 = export_stgcn(stgcn, cfg, output_dir)
    verify_onnx(p1, torch.randn(1, cfg["stgcn"]["num_frames"],
                                   cfg["stgcn"]["num_nodes"], 3))

    # ======================== FNO ========================
    logger.info(f"\n[2/3] 加载 FNO...")
    fno = FNO1d(
        input_dim=cfg["fno"]["input_dim"],
        output_dim=cfg["fno"]["output_dim"],
        modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"],
        depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"],
        use_lstm=False,  # 原始训练版本
    ).eval()

    fno_ck = os.path.join(ckpt_dir, f"fno_best{phase_suffix}.pth")
    if os.path.exists(fno_ck):
        fno.load_state_dict(torch.load(fno_ck, map_location=device))
        logger.info(f"  已加载: {fno_ck}")
    else:
        logger.warning(f"  ⚠️  {fno_ck} 不存在，使用随机初始化")

    fno.to(device)
    p2 = export_fno(fno, cfg, output_dir)
    verify_onnx(p2, torch.randn(1, cfg["fno"]["seq_len"],
                                   cfg["fno"]["input_dim"]))

    # ======================== Risk ========================
    logger.info(f"\n[3/3] 加载 Risk Model...")
    risk = RiskMLP(
        input_dim=cfg["risk"]["input_dim"],
        hidden_dim=cfg["risk"]["hidden_dim"],
    ).eval()

    risk_ck = os.path.join(ckpt_dir, f"risk_best{phase_suffix}.pth")
    if os.path.exists(risk_ck):
        risk.load_state_dict(torch.load(risk_ck, map_location=device))
        logger.info(f"  已加载: {risk_ck}")
    else:
        logger.warning(f"  ⚠️  {risk_ck} 不存在，使用随机初始化")

    risk.to(device)
    p3 = export_risk(risk, cfg, output_dir)
    T = cfg["risk"].get("seq_len", 20)
    verify_onnx(p3, torch.randn(1, T, cfg["risk"]["input_dim"]))

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Phase{phase} 所有模型已导出")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="导出 PhaseA/B 训练结果为 ONNX 文件"
    )
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B"],
                        help="要导出的阶段 (A 或 B，默认 A)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认 export/onnx_phaseX)")
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs/config.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    export_phaseA(cfg, phase=args.phase, output_dir=args.output)
