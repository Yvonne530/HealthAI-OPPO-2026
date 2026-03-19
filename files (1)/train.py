"""
train.py - 物理增强训练脚本
RehabGuardian 10.0 | 适配 RTX 4070 8GB

答辩要点：
- 基于AddBiomechanics官方数据，按受试者划分避免数据泄露
- AMP混合精度训练，batch_size=32适配8GB显存
- 物理损失确保GRF符合生物力学约束
"""

import os
import re
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import FNO1D, PhysicsInformedLoss


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_subject_id(filepath: str) -> str:
    """
    从文件路径提取受试者ID（如 AB06, AB08 等）
    
    支持格式：
        /data/AB06_trial1.npy
        /data/subjects/AB08/walking_01.npy
    """
    fname = Path(filepath).stem  # 去掉扩展名
    match = re.search(r'(AB\d{2}|subject_?\d+)', fname, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    # fallback：取文件名前4个字符作为ID
    return fname[:4].upper()


def subject_split(
    dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    ⚠️ 补丁3：按受试者ID划分训练/验证集，防止数据泄露
    
    确保同一受试者的所有试验段都在同一集合中，
    避免同一受试者数据同时出现在训练集和验证集。
    
    Args:
        dataset: ACLDataset 实例（需有 .file_list 属性）
        val_ratio: 验证集受试者比例（默认20%）
        seed: 随机种子
    
    Returns:
        train_indices, val_indices
    """
    # 获取每个样本对应的文件路径
    if hasattr(dataset, 'file_list'):
        file_list = dataset.file_list
    elif hasattr(dataset, 'samples'):
        file_list = [s[0] for s in dataset.samples]
    else:
        raise AttributeError("Dataset 需有 .file_list 或 .samples 属性")

    # 建立 受试者ID → 样本索引列表 的映射
    subject_to_indices: Dict[str, List[int]] = {}
    for idx, fp in enumerate(file_list):
        sid = extract_subject_id(str(fp))
        subject_to_indices.setdefault(sid, []).append(idx)

    # 按受试者ID排序，确保结果可复现
    all_subjects = sorted(subject_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(all_subjects)

    # 按比例划分受试者
    n_val = max(1, int(len(all_subjects) * val_ratio))
    val_subjects  = set(all_subjects[:n_val])
    train_subjects = set(all_subjects[n_val:])

    train_indices, val_indices = [], []
    for sid, indices in subject_to_indices.items():
        if sid in val_subjects:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    print(f"受试者划分: 训练 {len(train_subjects)} 人 / 验证 {len(val_subjects)} 人")
    print(f"样本数量:   训练 {len(train_indices)} / 验证 {len(val_indices)}")
    return train_indices, val_indices


def plot_grf_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    epoch: int,
    writer: SummaryWriter,
    sample_idx: int = 0,
) -> None:
    """
    绘制 GRF 预测 vs 真值对比图，写入 TensorBoard
    每10epoch保存一次
    
    Args:
        pred:   [B, 50, 6] numpy array
        target: [B, 50, 6] numpy array
        epoch:  当前 epoch
        writer: TensorBoard SummaryWriter
        sample_idx: 可视化哪个batch样本
    """
    labels = ['右脚Fx', '右脚Fy', '右脚Fz', '左脚Fx', '左脚Fy', '左脚Fz']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'GRF 预测 vs 真值 (Epoch {epoch})', fontsize=14)

    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.plot(target[sample_idx, :, i], label='真值', color='blue', linewidth=2)
        ax.plot(pred[sample_idx, :, i],   label='预测', color='red',  linewidth=2, linestyle='--')
        ax.set_title(label)
        ax.set_xlabel('时间帧')
        ax.set_ylabel('GRF (归一化)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    writer.add_figure('GRF对比/预测vs真值', fig, global_step=epoch)
    plt.close(fig)


# ============================================================
# 早停器
# ============================================================

class EarlyStopping:
    """早停机制，防止过拟合"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns:
            True 表示应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        """恢复最优权重"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print("✅ 已恢复最优模型权重")


# ============================================================
# 训练/验证单步函数
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedLoss,
    scaler: GradScaler,
    device: torch.device,
    clip_norm: float = 1.0,
) -> Dict[str, float]:
    """训练一个 epoch，使用 AMP 混合精度"""
    model.train()
    total_losses = {'total': 0., 'mse': 0., 'delta': 0., 'anatomical': 0.}
    n_batches = 0

    for X, Y in loader:
        X = X.to(device, non_blocking=True)  # [B, 50, 69]
        Y = Y.to(device, non_blocking=True)  # [B, 50, 6]

        optimizer.zero_grad(set_to_none=True)

        # AMP 自动混合精度
        with autocast():
            pred = model(X)              # [B, 50, 6]
            losses = criterion(pred, Y)  # dict

        # 反向传播 + 梯度裁剪
        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """验证，返回损失字典 + 最后一批次预测/真值（用于可视化）"""
    model.eval()
    total_losses = {'total': 0., 'mse': 0., 'delta': 0., 'anatomical': 0.}
    n_batches = 0
    last_pred = last_target = None

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        pred = model(X)
        losses = criterion(pred, Y)

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1
        last_pred   = pred.cpu().numpy()
        last_target = Y.cpu().numpy()

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    return avg_losses, last_pred, last_target


# ============================================================
# 主训练函数
# ============================================================

def train(args: argparse.Namespace) -> None:
    """完整训练流程"""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 加载 Dataset ---
    try:
        from dataset import ACLDataset  # 用户已实现的数据集
        dataset = ACLDataset(root=args.data_root)
    except ImportError:
        raise ImportError("请确保 dataset.py 中实现了 ACLDataset 类")

    # ⚠️ 按受试者划分，防止数据泄露
    train_idx, val_idx = subject_split(dataset, val_ratio=0.2, seed=args.seed)
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,    # 加速数据传输到GPU
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- 模型 ---
    model = FNO1D(
        in_channels=69,
        out_channels=6,
        width=64,
        modes=16,
        n_layers=4,
        seq_len=50,
    ).to(device)
    print(f"模型参数量: {model.count_parameters():,}")
    print(f"FP32大小:   {model.model_size_mb('fp32'):.2f} MB")

    # --- 损失 / 优化器 / 调度器 ---
    criterion = PhysicsInformedLoss(delta_weight=0.3, anatomical_weight=0.15)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler    = GradScaler()  # AMP 梯度缩放器
    early_stopper = EarlyStopping(patience=args.patience)

    # --- TensorBoard ---
    log_dir = Path(args.log_dir) / time.strftime('%Y%m%d_%H%M%S')
    writer  = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard 日志: {log_dir}")

    # --- 检查点目录 ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 训练循环
    # ============================================================
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 训练
        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # 验证
        val_losses, val_pred, val_target = validate(
            model, val_loader, criterion, device
        )

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # --- TensorBoard 记录 ---
        for split, losses in [('train', train_losses), ('val', val_losses)]:
            for k, v in losses.items():
                writer.add_scalar(f'loss/{split}_{k}', v, epoch)
        writer.add_scalar('lr', current_lr, epoch)

        # 每10epoch保存GRF曲线对比图
        if epoch % 10 == 0 and val_pred is not None:
            plot_grf_comparison(val_pred, val_target, epoch, writer)

        # --- 打印进度 ---
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_losses['total']:.4f} "
            f"(MSE={train_losses['mse']:.4f} Δ={train_losses['delta']:.4f} "
            f"Anat={train_losses['anatomical']:.4f}) | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"耗时: {elapsed:.1f}s"
        )

        # --- 保存最优模型 ---
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'args': vars(args),
                },
                ckpt_dir / 'best_model.pth'
            )
            print(f"  ✅ 保存最优模型 (val_loss={best_val_loss:.4f})")

        # --- 早停检测 ---
        if early_stopper.step(val_losses['total'], model):
            print(f"⏹  早停触发（patience={args.patience}），停止训练")
            early_stopper.restore_best(model)
            break

    # --- 训练结束 ---
    writer.close()
    print(f"\n✅ 训练完成！最优验证损失: {best_val_loss:.4f}")
    print(f"   权重保存于: {ckpt_dir / 'best_model.pth'}")
    print(f"   TensorBoard 日志: {log_dir}")
    print(f"   运行: tensorboard --logdir={args.log_dir}")


# ============================================================
# 入口
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RehabGuardian FNO-1D 训练脚本')
    parser.add_argument('--data_root',    type=str, default='./data',      help='数据集根目录')
    parser.add_argument('--ckpt_dir',     type=str, default='./checkpoints', help='权重保存目录')
    parser.add_argument('--log_dir',      type=str, default='./runs',       help='TensorBoard 日志目录')
    parser.add_argument('--epochs',       type=int, default=200)
    parser.add_argument('--batch_size',   type=int, default=32,   help='RTX 4070 8GB实测可用')
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience',     type=int, default=10,   help='早停patience')
    parser.add_argument('--seed',         type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
