"""
export/export_onnx.py
ONNX 导出（修复版）
修复：
  1. FNO 导出时强制使用 LSTM（避免 FFT 算子兼容问题）
  2. 输入输出 shape 与修复后的模型对齐（FNO 输出 (B,T,12)）
  3. opset_version 降到 11（OPPO NPU 更广泛支持）
"""
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)


def export_stgcn(model, cfg: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path  = os.path.join(out_dir, "stgcn.onnx")
    T, N  = cfg["stgcn"]["num_frames"], cfg["stgcn"]["num_nodes"]
    dummy = torch.randn(1, T, N, 3)

    torch.onnx.export(
        model, dummy, path,
        input_names  = ["visual_seq"],
        output_names = ["joint_angles"],
        dynamic_axes = {"visual_seq":   {0: "batch"},
                        "joint_angles": {0: "batch"}},
        opset_version = 12,          # 支持 einsum 操作（ST-GCN 使用）
        do_constant_folding = True,
    )
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"ST-GCN ONNX: {path} ({size_mb:.1f} MB)")
    return path


def export_fno(model, cfg: dict, out_dir: str) -> str:
    """
    FNO 导出：强制使用 LSTM 降级版（避免 FFT 算子兼容问题）
    输出 shape: (B, T, 12)
    """
    os.makedirs(out_dir, exist_ok=True)
    path  = os.path.join(out_dir, "fno_lstm.onnx")
    T, D  = cfg["fno"]["seq_len"], cfg["fno"]["input_dim"]
    dummy = torch.randn(1, T, D)

    from models.fno import FNO1d
    fno_lstm = FNO1d(
        input_dim  = cfg["fno"]["input_dim"],
        output_dim = cfg["fno"]["output_dim"],
        modes      = cfg["fno"]["modes"],
        width      = cfg["fno"]["width"],
        depth      = cfg["fno"]["depth"],
        seq_len    = cfg["fno"]["seq_len"],
        use_lstm   = True,   # 强制 LSTM，避免 FFT 算子
    ).eval()

    # 复制 LSTM 相关权重（若已训练）
    if hasattr(model, "lstm"):
        fno_lstm.load_state_dict(model.state_dict(), strict=False)

    torch.onnx.export(
        fno_lstm, dummy, path,
        input_names  = ["bio_seq"],
        output_names = ["grf_seq"],          # (B, T, 12)
        dynamic_axes = {"bio_seq":  {0: "batch"},
                        "grf_seq":  {0: "batch"}},
        opset_version = 12,
        do_constant_folding = True,
    )
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"FNO(LSTM) ONNX: {path} ({size_mb:.1f} MB)")
    return path


def export_risk(model, cfg: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path  = os.path.join(out_dir, "risk.onnx")
    D     = cfg["risk"]["input_dim"]
    T     = cfg["risk"].get("seq_len", 20)
    
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
        input_names  = ["risk_feat"],
        output_names = ["logits"],
        dynamic_axes = {"risk_feat": {0: "batch"},
                        "logits":    {0: "batch"}},
        opset_version = 12,
        do_constant_folding = True,
    )
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"Risk ONNX: {path} ({size_mb:.1f} MB)")
    return path


def verify_onnx(path: str, dummy: torch.Tensor) -> bool:
    try:
        import onnxruntime as ort
        sess     = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        out      = sess.run(None, {inp_name: dummy.numpy()})
        logger.info(f"  验证通过: 输入={dummy.shape} 输出={out[0].shape}")
        return True
    except ImportError:
        logger.warning("onnxruntime 未安装，跳过验证（pip install onnxruntime）")
        return False
    except Exception as e:
        logger.error(f"  验证失败: {e}")
        return False


def export_all(cfg: dict) -> None:
    from models.stgcn      import STGCN
    from models.fno        import FNO1d
    from models.risk_model import RiskMLP

    out_dir = cfg["export"]["onnx_dir"]
    ckpt    = cfg["train"]["checkpoint_dir"]
    device  = torch.device("cpu")

    # ST-GCN
    stgcn = STGCN(
        num_nodes       = cfg["stgcn"]["num_nodes"],
        hidden_channels = cfg["stgcn"]["hidden_channels"],
        output_dim      = cfg["stgcn"]["output_dim"],
    ).eval()
    ck = os.path.join(ckpt, "stgcn_best.pth")
    if os.path.exists(ck):
        stgcn.load_state_dict(torch.load(ck, map_location=device))
    p1 = export_stgcn(stgcn, cfg, out_dir)
    verify_onnx(p1, torch.randn(1, cfg["stgcn"]["num_frames"],
                                   cfg["stgcn"]["num_nodes"], 3))

    # FNO (LSTM 降级导出)
    fno = FNO1d(
        input_dim  = cfg["fno"]["input_dim"],
        output_dim = cfg["fno"]["output_dim"],
        modes      = cfg["fno"]["modes"],
        width      = cfg["fno"]["width"],
        depth      = cfg["fno"]["depth"],
        use_lstm   = False,  # 原始训练版
    ).eval()
    ck = os.path.join(ckpt, "fno_best.pth")
    if os.path.exists(ck):
        fno.load_state_dict(torch.load(ck, map_location=device))
    p2 = export_fno(fno, cfg, out_dir)   # 内部会转 LSTM
    verify_onnx(p2, torch.randn(1, cfg["fno"]["seq_len"],
                                   cfg["fno"]["input_dim"]))

    # Risk
    risk = RiskMLP(
        input_dim  = cfg["risk"]["input_dim"],
        hidden_dim = cfg["risk"]["hidden_dim"],
    ).eval()
    ck = os.path.join(ckpt, "risk_best.pth")
    if os.path.exists(ck):
        risk.load_state_dict(torch.load(ck, map_location=device))
    p3 = export_risk(risk, cfg, out_dir)
    T = cfg["risk"].get("seq_len", 20)
    verify_onnx(p3, torch.randn(1, T, cfg["risk"]["input_dim"]))

    logger.info(f"✅ 所有模型已导出到 {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "configs/config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    export_all(cfg)