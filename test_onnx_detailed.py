"""
test_onnx_detailed.py
详细的ONNX vs PyTorch调试脚本
检查单个样本的输出
"""
import logging
import os
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from models.stgcn import STGCN
from models.fno import FNO1d
from models.risk_model import RiskMLP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_pytorch_model(model_class, checkpoint_path: str, **kwargs):
    """加载PyTorch模型"""
    model = model_class(**kwargs).eval()
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state)
        logger.info(f"✓ 加载PyTorch模型: {checkpoint_path}")
    else:
        logger.warning(f"⚠ 检查点不存在: {checkpoint_path}，使用随机初始化")
    return model


def _find_onnx_path(cfg: dict, model_name: str) -> str:
    """查找ONNX文件，支持多个位置"""
    # 可能的ONNX目录
    onnx_dirs = [
        cfg["export"]["onnx_dir"],
        "./export/onnx_phasea/",
        "export/onnx_phasea/",
        os.path.join("export", "onnx_phasea"),
    ]
    
    # 尝试找到fno_lstm.onnx（FNO的LSTM版本）
    if model_name == "fno":
        for base_dir in onnx_dirs:
            path = os.path.join(base_dir, "fno_lstm.onnx")
            if os.path.exists(path):
                return path
    
    # 常规查找
    for base_dir in onnx_dirs:
        path = os.path.join(base_dir, f"{model_name}.onnx")
        if os.path.exists(path):
            return path
    
    return None


def test_stgcn_detailed(cfg: dict):
    """详细测试ST-GCN"""
    logger.info("\n" + "="*60)
    logger.info("详细测试 ST-GCN 模型")
    logger.info("="*60)
    
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("❌ 请安装 onnxruntime")
        return
    
    T, N = cfg["stgcn"]["num_frames"], cfg["stgcn"]["num_nodes"]
    onnx_path = _find_onnx_path(cfg, "stgcn")
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "stgcn_best.pth")
    
    # 加载模型
    stgcn_pt = load_pytorch_model(
        STGCN,
        ckpt_path,
        num_nodes=cfg["stgcn"]["num_nodes"],
        hidden_channels=cfg["stgcn"]["hidden_channels"],
        output_dim=cfg["stgcn"]["output_dim"],
    )
    
    # 检查eval模式
    logger.info(f"PyTorch模型训练模式: {stgcn_pt.training}")
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logger.info(f"✓ 加载ONNX模型: {onnx_path}")
    
    # 创建单个样本
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_input_np = np.random.randn(1, T, N, 3).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np)
    
    # PyTorch推理（多次运行检查一致性）
    with torch.no_grad():
        output1 = stgcn_pt(test_input_pt)[0].numpy()
        output2 = stgcn_pt(test_input_pt)[0].numpy()
    
    logger.info(f"PyTorch输出一致性检查: {np.allclose(output1, output2)}")
    logger.info(f"  第一次: {output1}")
    logger.info(f"  第二次: {output2}")
    
    # ONNX推理（多次运行检查一致性）
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    output_onnx1 = sess.run(output_names, {input_name: test_input_np})[0]
    output_onnx2 = sess.run(output_names, {input_name: test_input_np})[0]
    
    logger.info(f"ONNX输出一致性检查: {np.allclose(output_onnx1, output_onnx2)}")
    logger.info(f"  第一次: {output_onnx1}")
    logger.info(f"  第二次: {output_onnx2}")
    
    # 对比
    diff = np.abs(output1 - output_onnx1)
    logger.info(f"\n差异详情:")
    logger.info(f"  平均差异: {np.mean(diff):.6e}")
    logger.info(f"  最大差异: {np.max(diff):.6e}")
    logger.info(f"  最小差异: {np.min(diff):.6e}")
    logger.info(f"\nPyTorch输出统计:")
    logger.info(f"  均值: {output1.mean():.6f}")
    logger.info(f"  标准差: {output1.std():.6f}")
    logger.info(f"  最大值: {output1.max():.6f}")
    logger.info(f"  最小值: {output1.min():.6f}")
    logger.info(f"\nONNX输出统计:")
    logger.info(f"  均值: {output_onnx1.mean():.6f}")
    logger.info(f"  标准差: {output_onnx1.std():.6f}")
    logger.info(f"  最大值: {output_onnx1.max():.6f}")
    logger.info(f"  最小值: {output_onnx1.min():.6f}")


def test_fno_detailed(cfg: dict):
    """详细测试FNO"""
    logger.info("\n" + "="*60)
    logger.info("详细测试 FNO 模型")
    logger.info("="*60)
    
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("❌ 请安装 onnxruntime")
        return
    
    T, D = cfg["fno"]["seq_len"], cfg["fno"]["input_dim"]
    onnx_path = _find_onnx_path(cfg, "fno")
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "fno_best.pth")
    
    # 加载模型
    fno_pt = load_pytorch_model(
        FNO1d,
        ckpt_path,
        input_dim=cfg["fno"]["input_dim"],
        output_dim=cfg["fno"]["output_dim"],
        modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"],
        depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"],
        use_lstm=False,
    )
    
    logger.info(f"PyTorch模型训练模式: {fno_pt.training}")
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logger.info(f"✓ 加载ONNX模型: {onnx_path}")
    
    # 创建单个样本
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_input_np = np.random.randn(1, T, D).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np)
    
    # PyTorch推理
    with torch.no_grad():
        output1 = fno_pt(test_input_pt)[0].numpy()
        output2 = fno_pt(test_input_pt)[0].numpy()
    
    logger.info(f"PyTorch输出一致性检查: {np.allclose(output1, output2)}")
    logger.info(f"  第一次shape: {output1.shape}")
    logger.info(f"  第二次shape: {output2.shape}")
    logger.info(f"  第一个值: {output1[0, 0, :]}")
    logger.info(f"  第二个值: {output2[0, 0, :]}")
    
    # ONNX推理
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    output_onnx1 = sess.run(output_names, {input_name: test_input_np})[0]
    output_onnx2 = sess.run(output_names, {input_name: test_input_np})[0]
    
    logger.info(f"ONNX输出一致性检查: {np.allclose(output_onnx1, output_onnx2)}")
    logger.info(f"  第一次shape: {output_onnx1.shape}")
    logger.info(f"  第二次shape: {output_onnx2.shape}")
    logger.info(f"  第一个值: {output_onnx1[0, 0, :]}")
    logger.info(f"  第二个值: {output_onnx2[0, 0, :]}")
    
    # 对比
    diff = np.abs(output1 - output_onnx1)
    logger.info(f"\n差异详情:")
    logger.info(f"  平均差异: {np.mean(diff):.6e}")
    logger.info(f"  最大差异: {np.max(diff):.6e}")
    logger.info(f"  最小差异: {np.min(diff):.6e}")
    logger.info(f"\nPyTorch输出统计:")
    logger.info(f"  均值: {output1.mean():.6f}")
    logger.info(f"  标准差: {output1.std():.6f}")
    logger.info(f"  最大值: {output1.max():.6f}")
    logger.info(f"  最小值: {output1.min():.6f}")
    logger.info(f"\nONNX输出统计:")
    logger.info(f"  均值: {output_onnx1.mean():.6f}")
    logger.info(f"  标准差: {output_onnx1.std():.6f}")
    logger.info(f"  最大值: {output_onnx1.max():.6f}")
    logger.info(f"  最小值: {output_onnx1.min():.6f}")


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "configs/config.yaml")
    cfg = load_config(cfg_path)
    
    test_stgcn_detailed(cfg)
    test_fno_detailed(cfg)


if __name__ == "__main__":
    main()
