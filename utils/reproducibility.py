"""
utils/reproducibility.py
工程严谨性工具集：
  1. 数据版本锁（data fingerprint）
     - hash(文件路径 + mtime + 大小) → 唯一标识本次训练用的数据
  2. 结构化训练日志（JSON）
     - loss_curve, val_metrics, pass_selection, model_config
  3. 轻量自动消融实验（Ablation Runner）
     - 自动开关各模块（对称损失/PINN/时延模块/分阶段训练）
     - 输出消融对比表格
"""
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# 1. 数据版本锁
# =====================================================================

def compute_dataset_fingerprint(file_paths: List[str]) -> str:
    """
    计算数据集指纹：SHA256(sorted(path + mtime + size))
    用于确保不同实验使用完全相同的数据。

    用法：
        fingerprint = compute_dataset_fingerprint(loader.file_list)
        # 写入 config / 训练日志，便于复现
    """
    h = hashlib.sha256()
    for fp in sorted(file_paths):
        h.update(fp.encode())
        if os.path.exists(fp):
            stat = os.stat(fp)
            h.update(str(stat.st_mtime_ns).encode())
            h.update(str(stat.st_size).encode())
    return h.hexdigest()[:16]   # 取前16位，够用且简洁


def lock_dataset_version(
    b3d_files:  List[str],
    raw_files:  List[str],
    config:     dict,
    save_path:  str,
) -> dict:
    """
    生成数据版本锁文件，写入 config 并保存。
    Returns version_info dict。
    """
    version_info = {
        "b3d_fingerprint": compute_dataset_fingerprint(b3d_files),
        "raw_fingerprint": compute_dataset_fingerprint(raw_files),
        "b3d_file_count":  len(b3d_files),
        "raw_file_count":  len(raw_files),
        "created":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_seed":     config.get("project", {}).get("seed", 42),
    }
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(version_info, f, indent=2)
    logger.info(
        f"[版本锁] b3d={version_info['b3d_fingerprint']} "
        f"raw={version_info['raw_fingerprint']} "
        f"保存到 {save_path}"
    )
    return version_info


def verify_dataset_version(lock_path: str, current_files: List[str]) -> bool:
    """
    验证当前数据集与版本锁是否一致。
    不一致时警告（不报错，让用户决定）。
    """
    if not os.path.exists(lock_path):
        logger.warning(f"[版本锁] 锁文件不存在: {lock_path}（首次运行？）")
        return False
    with open(lock_path) as f:
        locked = json.load(f)
    current_fp = compute_dataset_fingerprint(current_files)
    if current_fp != locked.get("b3d_fingerprint"):
        logger.warning(
            f"[版本锁] ⚠️  数据集已变更！"
            f"锁定={locked['b3d_fingerprint']} 当前={current_fp}\n"
            "  如果是有意更改数据集，请删除版本锁文件重新生成。"
        )
        return False
    logger.info(f"[版本锁] 数据集一致 ✅  ({current_fp})")
    return True


# =====================================================================
# 2. 结构化训练日志
# =====================================================================

@dataclass
class TrainingRun:
    """单次训练完整记录"""
    run_id:        str
    config:        dict
    data_version:  dict
    model_params:  dict   # {stgcn: N, fno: N, risk: N}

    # 训练过程
    loss_curve:    List[float] = field(default_factory=list)
    val_loss:      List[float] = field(default_factory=list)
    val_grf_mae:   List[float] = field(default_factory=list)
    val_risk_acc:  List[float] = field(default_factory=list)
    learned_lag_ms:List[float] = field(default_factory=list)

    # 数据选择统计
    pass_selection: Dict[str, int] = field(default_factory=dict)  # {pass0: N, pass1: N, pass2: N}
    physics_weight_stats: dict     = field(default_factory=dict)

    # 元数据
    start_time:    str  = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time:      str  = ""
    best_val_loss: float = float("inf")
    total_epochs:  int   = 0
    early_stopped: bool  = False


class StructuredLogger:
    """
    结构化 JSON 训练日志记录器

    用法：
        slog = StructuredLogger("./logs/run_20240101.json")
        slog.start(config, data_version, model_params)
        for epoch ...:
            slog.log_epoch(tr_loss, val_loss, ...)
        slog.finish(early_stopped=False)
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._run: Optional[TrainingRun] = None
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def start(
        self,
        config:       dict,
        data_version: dict,
        model_params: dict,
    ) -> str:
        """开始新的训练 run，返回 run_id"""
        run_id     = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self._run  = TrainingRun(
            run_id=run_id,
            config=config,
            data_version=data_version,
            model_params=model_params,
        )
        logger.info(f"[StructuredLogger] 开始记录 run_id={run_id}")
        return run_id

    def log_epoch(
        self,
        train_loss:   float,
        val_loss:     float,
        val_grf_mae:  float  = 0.0,
        val_risk_acc: float  = 0.0,
        learned_lag:  float  = 0.0,
    ) -> None:
        if self._run is None:
            return
        self._run.loss_curve.append(round(train_loss, 6))
        self._run.val_loss.append(round(val_loss,     6))
        self._run.val_grf_mae.append(round(val_grf_mae, 6))
        self._run.val_risk_acc.append(round(val_risk_acc, 6))
        self._run.learned_lag_ms.append(round(learned_lag, 2))

        if val_loss < self._run.best_val_loss:
            self._run.best_val_loss = val_loss
        self._run.total_epochs += 1

    def log_pass_selection(self, pass_counts: Dict[str, int]) -> None:
        if self._run is None:
            return
        self._run.pass_selection = pass_counts

    def log_physics_stats(self, mean_pw: float, std_pw: float) -> None:
        if self._run is None:
            return
        self._run.physics_weight_stats = {
            "mean": round(mean_pw, 4),
            "std":  round(std_pw,  4),
            "description": "physics_weight = 1/(residual_error+eps), normalized to mean=1"
        }

    def finish(self, early_stopped: bool = False) -> None:
        if self._run is None:
            return
        self._run.end_time     = time.strftime("%Y-%m-%d %H:%M:%S")
        self._run.early_stopped= early_stopped

        # 转为 dict（过滤不可序列化的字段）
        run_dict = {
            "run_id":       self._run.run_id,
            "start_time":   self._run.start_time,
            "end_time":     self._run.end_time,
            "total_epochs": self._run.total_epochs,
            "early_stopped":self._run.early_stopped,
            "best_val_loss":round(self._run.best_val_loss, 6),
            "model_params": self._run.model_params,
            "data_version": self._run.data_version,
            "pass_selection":self._run.pass_selection,
            "physics_weight_stats": self._run.physics_weight_stats,
            "loss_curve":   self._run.loss_curve,
            "val_loss":     self._run.val_loss,
            "val_grf_mae":  self._run.val_grf_mae,
            "val_risk_acc": self._run.val_risk_acc,
            "learned_lag_ms": self._run.learned_lag_ms,
        }

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(run_dict, f, indent=2, ensure_ascii=False)
        logger.info(
            f"[StructuredLogger] 日志保存到 {self.log_path} "
            f"(best_val={self._run.best_val_loss:.4f})"
        )


# =====================================================================
# 3. 自动消融实验
# =====================================================================

@dataclass
class AblationResult:
    name:          str
    val_loss:      float
    val_grf_mae:   float
    val_risk_acc:  float
    learned_lag_ms:float
    config_diff:   dict   # 与 baseline 的差异


class AblationRunner:
    """
    轻量自动消融实验

    预设消融组合：
      baseline:    所有组件全开
      no_sym:      去掉对称约束损失
      no_pinn:     去掉 PINN 约束
      no_lag:      固定时延=0（禁用可学习时延）
      no_stages:   不分阶段，直接 E2E
      no_physw:    所有 physics_weight=1（忽略残差加权）

    用法（在训练脚本中调用）：
        ablation = AblationRunner(base_cfg)
        results = ablation.run(train_fn)  # train_fn(cfg) → metrics
        ablation.print_table(results)
    """

    ABLATION_CONFIGS = {
        "baseline": {},
        "no_sym_loss": {
            "_ablation_no_sym": True,
        },
        "no_pinn": {
            "_ablation_no_pinn": True,
        },
        "no_learnable_lag": {
            "_ablation_no_lag": True,
        },
        "no_stage_training": {
            "_ablation_no_stages": True,
        },
        "no_physics_weight": {
            "_ablation_no_physw": True,
        },
    }

    def __init__(self, base_cfg: dict, quick_epochs: int = 10):
        self.base_cfg     = base_cfg
        self.quick_epochs = quick_epochs

    def run(self, train_fn) -> List[AblationResult]:
        """
        运行所有消融配置，返回结果列表。
        train_fn(cfg) 应返回 dict with keys:
          val_loss, val_grf_mae, val_risk_acc, learned_lag_ms
        """
        results = []
        for name, diff in self.ABLATION_CONFIGS.items():
            logger.info(f"[Ablation] 运行配置: {name}")
            cfg = self._merge_config(self.base_cfg, diff)
            cfg["train"]["num_epochs"] = self.quick_epochs

            try:
                metrics = train_fn(cfg)
                results.append(AblationResult(
                    name          = name,
                    val_loss      = metrics.get("val_loss",     float("nan")),
                    val_grf_mae   = metrics.get("val_grf_mae",  float("nan")),
                    val_risk_acc  = metrics.get("val_risk_acc", float("nan")),
                    learned_lag_ms= metrics.get("learned_lag_ms", 0.0),
                    config_diff   = diff,
                ))
            except Exception as e:
                logger.warning(f"  消融 {name} 失败: {e}")
                results.append(AblationResult(
                    name=name, val_loss=float("nan"),
                    val_grf_mae=float("nan"), val_risk_acc=float("nan"),
                    learned_lag_ms=0.0, config_diff=diff,
                ))

        return results

    def print_table(self, results: List[AblationResult]) -> None:
        """打印消融结果对比表"""
        print("\n" + "=" * 75)
        print("消融实验结果对比")
        print("=" * 75)
        print(f"{'配置':<22} {'val_loss':>10} {'grf_mae':>10} {'risk_acc':>10} {'lag(ms)':>10}")
        print("-" * 75)

        baseline = next((r for r in results if r.name == "baseline"), None)
        for r in results:
            delta = ""
            if baseline and r.name != "baseline":
                d = r.val_loss - baseline.val_loss
                delta = f" ({'+' if d>0 else ''}{d:.4f})"
            print(f"{r.name:<22} {r.val_loss:>10.4f} "
                  f"{r.val_grf_mae:>10.4f} {r.val_risk_acc:>10.3f} "
                  f"{r.learned_lag_ms:>10.1f}{delta}")

        print("=" * 75)
        print("(+δ 表示比 baseline 更差，即该组件有正面贡献)")

    def save_results(self, results: List[AblationResult], path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = [
            {
                "name":           r.name,
                "val_loss":       round(r.val_loss,      4),
                "val_grf_mae":    round(r.val_grf_mae,   4),
                "val_risk_acc":   round(r.val_risk_acc,  4),
                "learned_lag_ms": round(r.learned_lag_ms,1),
                "config_diff":    r.config_diff,
            }
            for r in results
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[Ablation] 结果保存到 {path}")

    @staticmethod
    def _merge_config(base: dict, diff: dict) -> dict:
        import copy
        cfg = copy.deepcopy(base)
        cfg.update(diff)
        return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ---- 数据版本锁测试 ----
    fake_files = ["/tmp/test1.b3d", "/tmp/test2.b3d"]
    for f in fake_files:
        with open(f, "w") as fp:
            fp.write(f)
    fp1 = compute_dataset_fingerprint(fake_files)
    fp2 = compute_dataset_fingerprint(fake_files)
    assert fp1 == fp2, "相同文件应有相同指纹"
    fp3 = compute_dataset_fingerprint(fake_files[:1])
    assert fp1 != fp3, "不同文件集应有不同指纹"
    print(f"数据版本锁: {fp1} ✅")

    lock_info = lock_dataset_version(fake_files, [], {}, "/tmp/rg_data_lock.json")
    assert os.path.exists("/tmp/rg_data_lock.json")
    ok = verify_dataset_version("/tmp/rg_data_lock.json", fake_files)
    assert ok, "版本验证失败"
    # 修改文件后应报不一致
    import time as _time
    _time.sleep(0.01)
    with open(fake_files[0], "w") as fp:
        fp.write("modified")
    ok2 = verify_dataset_version("/tmp/rg_data_lock.json", fake_files)
    assert not ok2, "应检测到文件变更"
    print("数据版本验证（变更检测）✅")

    # ---- 结构化日志测试 ----
    slog = StructuredLogger("/tmp/rg_test_run.json")
    slog.start(
        config       = {"project": {"seed": 42}},
        data_version = lock_info,
        model_params = {"stgcn": 5_200_000, "fno": 300_000, "risk": 12_000},
    )
    slog.log_pass_selection({"pass0": 8500, "pass1": 1200, "pass2": 300})
    slog.log_physics_stats(mean_pw=1.0, std_pw=0.35)
    for ep in range(5):
        slog.log_epoch(
            train_loss=1.0 - ep * 0.1,
            val_loss=0.9 - ep * 0.08,
            val_grf_mae=150.0 - ep * 10,
            val_risk_acc=0.6 + ep * 0.05,
            learned_lag=50 - ep * 2,
        )
    slog.finish(early_stopped=False)
    assert os.path.exists("/tmp/rg_test_run.json")
    with open("/tmp/rg_test_run.json") as f:
        log = json.load(f)
    assert "pass_selection" in log
    assert "physics_weight_stats" in log
    assert len(log["loss_curve"]) == 5
    print(f"结构化日志: run_id={log['run_id']}, best_val={log['best_val_loss']} ✅")

    # ---- 消融框架测试 ----
    base_cfg = {"train": {"num_epochs": 5, "learning_rate": 0.001}}

    def dummy_train(cfg):
        # 模拟：有对称损失时稍好
        base_v = 0.45
        if cfg.get("_ablation_no_sym"):
            base_v += 0.03
        if cfg.get("_ablation_no_pinn"):
            base_v += 0.02
        if cfg.get("_ablation_no_lag"):
            base_v += 0.05
        return {"val_loss": base_v, "val_grf_mae": 120.0,
                "val_risk_acc": 0.88, "learned_lag_ms": 45.0}

    ablation = AblationRunner(base_cfg, quick_epochs=5)
    results  = ablation.run(dummy_train)
    ablation.print_table(results)
    ablation.save_results(results, "/tmp/rg_ablation.json")
    assert os.path.exists("/tmp/rg_ablation.json")
    print("消融实验框架 ✅")

    print("\nreproducibility.py OK ✅")