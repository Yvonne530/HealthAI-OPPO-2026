"""
utils/kinematics.py
可微运动学工具函数
- finite_diff: 时间差分（支持 torch.autograd 梯度回传）
- compute_com:  从关节角度估算质心（简化 Dempster 分节模型）
全部基于 PyTorch，确保端到端梯度贯通
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


# -----------------------------------------------------------------------
# 有限差分（可微）
# -----------------------------------------------------------------------

def finite_diff(
    x: torch.Tensor,   # (B, T, D)
    dt: float = 1.0 / 60.0,
    order: int = 1,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """
    沿时间轴 (dim=1) 做有限差分，保持形状 (B, T, D)。
    pad_mode="replicate" 保持序列长度不变（边界复制）。
    支持 autograd 梯度回传。

    Args:
        x:       (B, T, D)
        dt:      时间步长（秒）
        order:   阶数（1=速度, 2=加速度）
        pad_mode: 边界填充方式
    Returns:
        (B, T, D) 与输入同形状
    """
    assert x.ndim == 3, f"期望 (B,T,D)，得到 {x.shape}"
    assert order in (1, 2)

    if order == 1:
        # 中心差分：dx/dt ≈ (x[t+1] - x[t-1]) / (2dt)
        # 边界用前向/后向差分
        x_pad = F.pad(x.permute(0,2,1), (1, 1), mode=pad_mode).permute(0,2,1)
        diff = (x_pad[:, 2:, :] - x_pad[:, :-2, :]) / (2 * dt)
    else:
        # 二阶差分：d²x/dt² ≈ (x[t+1] - 2x[t] + x[t-1]) / dt²
        x_pad = F.pad(x.permute(0,2,1), (1, 1), mode=pad_mode).permute(0,2,1)
        diff = (x_pad[:, 2:, :] - 2 * x[:, :, :] + x_pad[:, :-2, :]) / (dt ** 2)

    assert diff.shape == x.shape
    return diff


# -----------------------------------------------------------------------
# 质心估算（可微，基于 Dempster 分节模型简化版）
# -----------------------------------------------------------------------

# OpenSim gait2392 关节索引 → 身体部位贡献
# 简化：用髋、膝、踝关节角度估算质心垂直位移
# 参考：Winter, D.A. "Biomechanics and Motor Control of Human Movement"
_HIP_L  = 3   # hip_flexion_l
_HIP_R  = 10  # hip_flexion_r
_KNEE_L = 6   # knee_angle_l
_KNEE_R = 13  # knee_angle_r
_ANKL_L = 7   # ankle_angle_l
_ANKL_R = 14  # ankle_angle_r

# 体节质量比（Dempster 1955）
_THIGH_MASS = 0.100   # 大腿/全身
_SHANK_MASS = 0.0465  # 小腿
_FOOT_MASS  = 0.0145

# 链长归一化（相对于身高，默认1.70m）
_L_THIGH = 0.245  # m
_L_SHANK = 0.246


def compute_com(
    joint_angles: torch.Tensor,     # (B, T, 23) 弧度
    height: float = 1.70,
    dt: float = 1.0 / 60.0,
) -> torch.Tensor:
    """
    基于 Dempster 分节模型简化估算质心位置 (B, T, 3)。
    支持梯度回传。

    简化假设：
    - 只估算矢状面（y 轴）质心高度变化
    - x/z 分量用骨盆角度（pelvis_tilt, pelvis_list）估算
    - 绝对值用 0 归一化，只预测相对偏移

    Physical reference:
      Gard & Childress (1997) "The influence of stance-phase knee
      flexion on the vertical displacement of the trunk during normal
      walking", Arch Phys Med Rehabil.
    """
    B, T, D = joint_angles.shape
    assert D == 23, f"期望 23 维，得到 {D}"

    # 提取关键角度
    hip_l  = joint_angles[:, :, _HIP_L]    # (B, T)
    hip_r  = joint_angles[:, :, _HIP_R]
    knee_l = joint_angles[:, :, _KNEE_L]
    knee_r = joint_angles[:, :, _KNEE_R]
    ankl_l = joint_angles[:, :, _ANKL_L]
    ankl_r = joint_angles[:, :, _ANKL_R]

    # 双腿平均（步态对称假设）
    hip_avg  = (hip_l  + hip_r)  * 0.5
    knee_avg = (knee_l + knee_r) * 0.5

    # 质心高度简化估算（矢状面运动学）
    # ΔyCOM ≈ L_thigh*(1-cos(hip)) + L_shank*(1-cos(knee))
    l_th = _L_THIGH * height / 1.70
    l_sh = _L_SHANK * height / 1.70

    y_com = (l_th * (1.0 - torch.cos(hip_avg))
           + l_sh * (1.0 - torch.cos(knee_avg)))  # (B, T)

    # x 方向：骨盆前后倾（pelvis_tilt = index 0）
    pelvis_tilt = joint_angles[:, :, 0]
    x_com = l_th * torch.sin(pelvis_tilt) * 0.5

    # z 方向：骨盆侧倾（pelvis_list = index 1）
    pelvis_list = joint_angles[:, :, 1]
    z_com = l_th * torch.sin(pelvis_list) * 0.5

    com = torch.stack([x_com, y_com, z_com], dim=-1)  # (B, T, 3)
    assert com.shape == (B, T, 3)
    return com


# -----------------------------------------------------------------------
# 构建 FNO 输入特征（端到端可微）
# -----------------------------------------------------------------------

def build_fno_features(
    pred_ja: torch.Tensor,    # (B, T, 23)  来自 ST-GCN（或 GT）
    dt: float = 1.0 / 60.0,
) -> torch.Tensor:
    """
    构建 FNO 72 维输入特征，全程可微：
      [joint_angles(23) | vel(23) | acc(23) | com(3)] → (B, T, 72)

    vel/acc 通过有限差分从 pred_ja 计算，梯度可回传到 ST-GCN。
    com 通过简化运动学模型从 pred_ja 计算。
    """
    B, T, D = pred_ja.shape
    assert D == 23

    vel = finite_diff(pred_ja, dt=dt, order=1)      # (B, T, 23)
    acc = finite_diff(pred_ja, dt=dt, order=2)      # (B, T, 23)
    com = compute_com(pred_ja)                       # (B, T, 3)

    feat = torch.cat([pred_ja, vel, acc, com], dim=-1)   # (B, T, 72)
    assert feat.shape == (B, T, 72), f"特征维度错误: {feat.shape}"
    return feat


# -----------------------------------------------------------------------
# OOD 检测（Mahalanobis 距离）
# -----------------------------------------------------------------------

class MahalanobisDetector:
    """
    基于 Mahalanobis 距离的分布外检测器。
    在训练集特征上 fit，推理时检测输入是否 OOD。

    reference:
      Lee et al. 2018 "A Simple Unified Framework for Detecting
      Out-of-Distribution Samples and Adversarial Attacks"
    """

    def __init__(self, threshold_percentile: float = 99.0):
        self.threshold_percentile = threshold_percentile
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: float = float("inf")
        self._fitted = False

    def fit(self, features: np.ndarray) -> "MahalanobisDetector":
        """
        features: (N, D) 训练集特征
        """
        self.mean    = features.mean(axis=0)
        cov          = np.cov(features.T) + np.eye(features.shape[1]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)

        # 计算训练集自身的距离分布，设定阈值
        dists = self._batch_distance(features)
        self.threshold = float(np.percentile(dists, self.threshold_percentile))
        self._fitted   = True
        return self

    def _batch_distance(self, x: np.ndarray) -> np.ndarray:
        diff = x - self.mean                          # (N, D)
        left = diff @ self.cov_inv                    # (N, D)
        dist = np.sqrt((left * diff).sum(axis=1))     # (N,)
        return dist

    def is_ood(self, feat: np.ndarray) -> bool:
        """feat: (D,) 单帧特征"""
        if not self._fitted:
            return False
        dist = float(self._batch_distance(feat[np.newaxis])[0])
        return dist > self.threshold

    def save(self, path: str) -> None:
        np.savez(path,
                 mean=self.mean,
                 cov_inv=self.cov_inv,
                 threshold=np.array([self.threshold]))

    def load(self, path: str) -> "MahalanobisDetector":
        data = np.load(path)
        self.mean      = data["mean"]
        self.cov_inv   = data["cov_inv"]
        self.threshold = float(data["threshold"][0])
        self._fitted   = True
        return self


if __name__ == "__main__":
    import torch

    # finite_diff 测试
    B, T, D = 4, 20, 23
    x = torch.randn(B, T, D, requires_grad=True)
    vel = finite_diff(x, dt=1/60, order=1)
    acc = finite_diff(x, dt=1/60, order=2)
    assert vel.shape == (B, T, D) and acc.shape == (B, T, D)
    # 梯度测试
    vel.sum().backward()
    assert x.grad is not None
    print("finite_diff 梯度测试 ✅")

    # compute_com 测试
    x2 = torch.randn(B, T, 23, requires_grad=True)
    com = compute_com(x2)
    assert com.shape == (B, T, 3)
    com.sum().backward()
    assert x2.grad is not None
    print("compute_com 梯度测试 ✅")

    # build_fno_features 测试
    x3 = torch.randn(B, T, 23, requires_grad=True)
    feat = build_fno_features(x3)
    assert feat.shape == (B, T, 72)
    feat.sum().backward()
    assert x3.grad is not None
    print("build_fno_features 梯度贯通 ✅")

    # Mahalanobis 测试
    rng = np.random.default_rng(0)
    train_feats = rng.normal(0, 1, (1000, 10))
    det = MahalanobisDetector(threshold_percentile=99.0)
    det.fit(train_feats)
    normal_feat = rng.normal(0, 1, (10,))
    ood_feat    = rng.normal(10, 1, (10,))   # 远离训练分布
    assert not det.is_ood(normal_feat), "正常样本误判为 OOD"
    assert det.is_ood(ood_feat),        "OOD 样本未检出"
    print("MahalanobisDetector 测试 ✅")

    print("\nkinematics.py OK ✅")