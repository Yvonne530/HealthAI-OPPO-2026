"""
main.py  (v3 - 修复 Bug 3/4/5 + 双数据源支持)
修复的 Bug：
  3. result 变量作用域（循环外访问保护）
  4. FNO1d 缺少 future_k 参数
  5. RiskMLP 缺少 seq_len / use_physio 参数
  新增：支持 raw OpenSim (.sto) 数据源
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logger = logging.getLogger("main")


def setup_logging(cfg: dict) -> None:
    log_dir = cfg["train"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8"),
        ],
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =====================================================================
# 数据预处理（支持 .b3d 和原始 OpenSim 双源）
# =====================================================================

def run_preprocess(cfg: dict) -> list:
    """加载数据（自动检测 .b3d 或 原始 OpenSim 格式），运行教师打标"""
    from utils.teacher_labeler import TeacherLabeler

    b3d_root = cfg["data"]["raw_root"]
    raw_root = cfg["data"].get("raw_opensim_root", "")

    all_samples = []

    # 1. 加载 .b3d 数据
    if os.path.isdir(b3d_root):
        from utils.b3d_loader import B3DLoader
        loader = B3DLoader(b3d_root)
        logger.info(f"[.b3d] 加载 {len(loader)} 个文件...")
        all_samples.extend(loader.get_all_samples())
    else:
        logger.warning(f".b3d 目录不存在: {b3d_root}")

    # 2. 加载原始 OpenSim 数据（若配置了 raw_opensim_root）
    if raw_root and os.path.isdir(raw_root):
        from utils.raw_data_loader import RawDataLoader
        raw_loader = RawDataLoader(raw_root)
        logger.info(f"[OpenSim] 加载 {len(raw_loader)} 个受试者...")
        all_samples.extend(raw_loader.get_all_samples())
    elif raw_root:
        logger.warning(f"OpenSim 原始数据目录不存在: {raw_root}")

    if not all_samples:
        logger.warning("未找到真实数据，使用合成数据演示...")
        all_samples = _make_synthetic_samples(n=600)

    logger.info(f"总样本数: {len(all_samples)}")

    # 教师打标（写回 risk_label）
    labeler = TeacherLabeler()
    labels, scores = labeler.label_sequence(all_samples)
    dist = np.bincount(labels, minlength=3)
    logger.info(f"风险标签分布: 低={dist[0]} 中={dist[1]} 高={dist[2]}")

    return all_samples


def _make_synthetic_samples(n: int = 600) -> list:
    from utils.rg_sample import RGSample
    rng = np.random.default_rng(42)
    samples = []
    seq_ids = [f"seq_{i//100}" for i in range(n)]
    for i in range(n):
        s = RGSample()
        s.t = float(i % 100) / 60; s.sequence_id = seq_ids[i]; s.valid = True
        s.joint_angles = rng.normal(0, 0.3, 23).astype(np.float32)
        s.joint_vel    = rng.normal(0, 1.0, 23).astype(np.float32)
        s.joint_acc    = rng.normal(0, 5.0, 23).astype(np.float32)
        s.markers      = rng.normal(0, 0.5, (28, 3)).astype(np.float32)
        s.grf_left     = np.abs(rng.normal(300, 100, 6)).astype(np.float32)
        s.grf_right    = np.abs(rng.normal(300, 100, 6)).astype(np.float32)
        s.com          = rng.normal(0, 1.0, 3).astype(np.float32)
        s.grf_mask     = 1.0; s.physics_weight = 1.0
        s.contact_left = True; s.contact_right = False
        s.risk_label   = int(rng.integers(0, 3))
        samples.append(s)
    logger.info(f"合成数据: {n} 帧")
    return samples


# =====================================================================
# 训练
# =====================================================================

def run_train(cfg: dict, samples: list) -> None:
    import torch
    from models.stgcn          import STGCN
    from models.fno            import FNO1d
    from models.risk_model     import RiskMLP
    from trainers.train_pipeline import RehabGuardianTrainer
    from data.datasets import _split_by_sequence

    train_samples = _split_by_sequence(samples, split="train",
                                       train_ratio=cfg["data"]["train_ratio"],
                                       val_ratio=cfg["data"]["val_ratio"])
    val_samples   = _split_by_sequence(samples, split="val",
                                       train_ratio=cfg["data"]["train_ratio"],
                                       val_ratio=cfg["data"]["val_ratio"])
    logger.info(f"训练集: {len(train_samples)} | 验证集: {len(val_samples)}")

    # Bug Fix #4：FNO1d 补全 future_k 参数
    future_k = cfg["fno"].get("future_k", 10)
    stgcn = STGCN(
        num_nodes       = cfg["stgcn"]["num_nodes"],
        in_channels     = cfg["stgcn"]["in_channels"],
        hidden_channels = cfg["stgcn"]["hidden_channels"],
        num_frames      = cfg["stgcn"]["num_frames"],
        output_dim      = cfg["stgcn"]["output_dim"],
        dropout         = cfg["stgcn"]["dropout"],
    )
    fno = FNO1d(
        input_dim  = cfg["fno"]["input_dim"],
        output_dim = cfg["fno"]["output_dim"],
        future_k   = future_k,               # ✅ 修复：补全 future_k
        modes      = cfg["fno"]["modes"],
        width      = cfg["fno"]["width"],
        depth      = cfg["fno"]["depth"],
        seq_len    = cfg["fno"]["seq_len"],
    )
    # Bug Fix #5：RiskMLP 补全 seq_len 和 use_physio
    risk = RiskMLP(
        input_dim   = cfg["risk"]["input_dim"],
        hidden_dim  = cfg["risk"]["hidden_dim"],
        num_classes = cfg["risk"]["num_classes"],
        seq_len     = cfg["risk"].get("seq_len", 20),      # ✅ 修复
        use_physio  = cfg["risk"].get("use_physio", True), # ✅ 修复
    )

    logger.info(f"ST-GCN: {stgcn.count_params()/1e6:.2f}M 参数")
    logger.info(f"FNO:    {fno.count_params()/1e6:.3f}M 参数")
    logger.info(f"Risk:   {risk.count_params()} 参数")
    logger.info(f"FNO 初始时延: {fno.current_lag_ms:.1f}ms")

    trainer = RehabGuardianTrainer(cfg)
    history = trainer.train_all(stgcn, fno, risk, train_samples, val_samples)

    final_val = history["val_loss"][-1] if history["val_loss"] else float("nan")
    final_lag = history["learned_lag_ms"][-1] if history["learned_lag_ms"] else 0
    logger.info(f"训练完成 | 最终验证 loss={final_val:.4f} | "
                f"学习时延={final_lag:.1f}ms")


# =====================================================================
# 推理
# =====================================================================

def run_infer(cfg: dict) -> None:
    from inference.inference import RehabGuardianInference
    from inference.heytap_health_adapter import HeytapHealthAdapter

    engine  = RehabGuardianInference(cfg, device="cpu")
    adapter = HeytapHealthAdapter(mock=True)

    hr    = adapter.get_heart_rate()
    sleep = adapter.get_sleep_score()
    engine.update_health_data(hr, sleep)

    # Bug Fix #3：result 变量初始化，避免循环0次时访问失败
    result = None
    latencies = []
    for i in range(30):
        skeleton = np.random.randn(33, 3).astype(np.float32)
        result   = engine.run(skeleton, mode="pose")
        latencies.append(result["latency_ms"])
        if i % 10 == 0:
            logger.info(
                f"帧{i:02d}: {result['risk_text']} "
                f"{result['latency_ms']:.1f}ms "
                f"conf={result.get('confidence',0):.2f}"
            )

    if latencies:
        logger.info(f"平均延迟: {np.mean(latencies):.2f}ms (目标 <15ms)")

    # Bug Fix #3：result 可能为 None 的保护
    if result is not None:
        adapter.write_risk_analysis(
            start_time_ms = int(time.time() * 1000),
            duration_ms   = int(30 / 60 * 1000),
            risk_score    = result["risk_score"],
            risk_level    = result["risk_label"],
        )


# =====================================================================
# 导出
# =====================================================================

def run_export(cfg: dict) -> None:
    from export.export_onnx import export_all
    export_all(cfg)


# =====================================================================
# 入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="RehabGuardian 10.0 v3")
    parser.add_argument("--mode",   default="train",
                        choices=["train", "infer", "export", "preprocess"])
    parser.add_argument("--config", default=os.path.join(ROOT, "configs/config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    np.random.seed(cfg["project"]["seed"])
    try:
        import torch
        torch.manual_seed(cfg["project"]["seed"])
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    logger.info("=" * 60)
    logger.info(f"RehabGuardian {cfg['project']['version']}")
    logger.info(f"模式: {args.mode}")
    logger.info("=" * 60)

    if args.mode == "preprocess":
        samples = run_preprocess(cfg)
        logger.info(f"预处理完成，{len(samples)} 个样本")
    elif args.mode == "train":
        samples = run_preprocess(cfg)
        run_train(cfg, samples)
    elif args.mode == "infer":
        run_infer(cfg)
    elif args.mode == "export":
        run_export(cfg)

    logger.info("任务完成 ✅")


if __name__ == "__main__":
    main()