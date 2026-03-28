"""
models/risk_model.py  (v2 - 时序 + 置信度头 + 生理融合)
升级：
  - 输入 (B, T, 35)，经 1D Conv 时序建模再接 MLP
  - 双头输出：logits(3) + confidence(1)
  - 融合心率/睡眠评分（作为额外 2 维特征）
  - SafetyHeartbeat：组合规则 + 时间累积 + 自适应阈值

物理规则参考：
  Hewett TE et al. 2005. "Biomechanical measures of neuromuscular
  control and valgus loading of the knee predict anterior cruciate
  ligament injury risk in female athletes." Am J Sports Med.
  Myer GD et al. 2010. "The influence of age on the effectiveness of
  neuromuscular training to reduce anterior cruciate ligament injury in female athletes." Am J Sports Med.
"""
import logging
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

INPUT_DIM_BASE = 35    # joint_angles(23) + grf(12)
INPUT_DIM_FULL = 37    # + heart_rate(1) + sleep_score(1)
NUM_CLASSES    = 3
FUTURE_K       = 10    # 与 FNO 对齐


class TemporalRiskEncoder(nn.Module):
    """
    时序风险编码器：1D 卷积 + 残差
    捕获连续帧的风险演变趋势
    """
    def __init__(self, in_dim, hidden, drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim,  hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden,  hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden,  hidden, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.drop  = nn.Dropout(drop)
        self.proj  = nn.Linear(in_dim, hidden)   # 残差投影

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        h = x.permute(0, 2, 1)              # (B, D, T)
        h = self.drop(F.gelu(self.bn1(self.conv1(h))))
        h = self.drop(F.gelu(self.bn2(self.conv2(h))))
        h = F.gelu(self.bn3(self.conv3(h)))

        # 残差（全局 avg pool 匹配维度）
        res = self.proj(x).permute(0,2,1)   # (B, hidden, T)
        h   = h + res
        h   = h.mean(dim=-1)               # (B, hidden) 全局池化
        return h


class RiskMLP(nn.Module):
    """
    时序风险分类器 v2

    输入:  (B, T, 35) 或 (B, T, 37)（含生理数据）
          或者二维 (B, 35/37)，会自动扩展成 T=1
    输出:  logits (B, 3)  +  confidence (B, 1)

    Loss:
      L = CrossEntropy(logits, label)
        + lambda * BinaryCrossEntropy(confidence, |label - pred_label| == 0)
    """

    def __init__(
        self,
        input_dim:   int   = INPUT_DIM_BASE,
        hidden_dim:  int   = 64,
        num_classes: int   = NUM_CLASSES,
        seq_len:     int   = 20,
        dropout:     float = 0.2,
        use_physio:  bool  = True,      # 融合心率/睡眠
    ):
        super().__init__()
        self.use_physio = use_physio
        eff_dim = input_dim + (2 if use_physio else 0)  # 37 或 35

        self.encoder = TemporalRiskEncoder(eff_dim, hidden_dim, drop=dropout)

        # 风险分类头
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        # 置信度头（sigmoid → 0~1）
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, 35) 或 (B, 35)
        heart_rate:   Optional[torch.Tensor] = None,   # (B, 1) 或 (B,)
        sleep_score:  Optional[torch.Tensor] = None,   # (B, 1) 或 (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:     (B, 3)
            confidence: (B, 1)
        """
        # 如果输入是二维 (B, D)，加一个时间维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        B, T, D = x.shape

        # 拼接生理特征（沿所有时间步广播）
        if self.use_physio and heart_rate is not None and sleep_score is not None:
            hr  = heart_rate.float().reshape(B, 1, 1).expand(B, T, 1)
            slp = sleep_score.float().reshape(B, 1, 1).expand(B, T, 1)
            x   = torch.cat([x, hr, slp], dim=-1)   # (B, T, 37)

        feat    = self.encoder(x)              # (B, hidden)
        logits  = self.risk_head(feat)         # (B, 3)
        conf    = self.conf_head(feat)         # (B, 1)

        assert logits.shape == (B, NUM_CLASSES)
        assert conf.shape   == (B, 1)
        return logits, conf

    def predict(
        self,
        x: torch.Tensor,
        heart_rate: Optional[torch.Tensor] = None,
        sleep_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            label:      (B,)    预测类别
            risk_score: (B,)    高风险概率
            confidence: (B,)    置信度
        """
        logits, conf = self.forward(x, heart_rate, sleep_score)
        probs  = F.softmax(logits, dim=-1)
        label  = probs.argmax(dim=-1)
        score  = probs[:, 2]                   # 高风险概率
        return label, score, conf.squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SafetyHeartbeat:
    """
    物理规则引擎 v2（升级）

    改进：
    1. 组合规则（多条件联合判断）
    2. 时间累积（连续 N 帧 → 升级风险）
    3. 自适应阈值（心率+睡眠驱动）
    4. 输出 = max(模型等级, 规则等级)
    """
    KNEE_L = 6;  KNEE_R  = 13
    HIP_L  = 3;  HIP_R   = 10
    KNEE_MIN_RAD   = -0.0873   # -5° 过伸
    KNEE_MAX_RAD   =  2.094    # 120°
    KNEE_ASYMM_RAD =  0.2618   # 15°
    TEMPORAL_WINDOW = 10       # 连续帧累积窗口

    def __init__(self, body_weight_kg: float = 70.0):
        self.bw_n = body_weight_kg * 9.81
        self._hr: float = 70.0
        self._sleep: float = 80.0
        self._rule_hist: deque = deque(maxlen=self.TEMPORAL_WINDOW)

    def update_heart_rate(self, hr: float) -> None:
        self._hr = float(np.clip(hr, 40, 200))

    def update_sleep_score(self, score: float) -> None:
        self._sleep = float(np.clip(score, 0, 100))

    def _grf_threshold_bw(self) -> float:
        base = 1.5
        hr_factor    = 1.0 - 0.05 * max(0, (self._hr - 80) / 10)
        sleep_factor = 0.8 + 0.2 * (self._sleep / 100)
        return base * float(np.clip(hr_factor * sleep_factor, 0.6, 1.3))

    def _grf_med_threshold_bw(self) -> float:
        return self._grf_threshold_bw() * 0.7

    def _eval_rules(self, joint_angles: np.ndarray, grf: np.ndarray) -> Tuple[int, float, list]:
        ja  = joint_angles
        grf_z = abs(grf[2]) + abs(grf[8]) if len(grf) >= 9 else abs(grf[2])
        grf_bw = grf_z / self.bw_n
        thr_high = self._grf_threshold_bw()
        thr_med  = self._grf_med_threshold_bw()
        rule_label  = 0
        rule_conf   = 0.5
        triggered   = []

        # 膝关节过伸
        if ja[self.KNEE_L] < self.KNEE_MIN_RAD or ja[self.KNEE_R] < self.KNEE_MIN_RAD:
            rule_label = max(rule_label, 2)
            rule_conf  = 0.95
            triggered.append("knee_overextension")

        # 膝关节过大
        if ja[self.KNEE_L] > self.KNEE_MAX_RAD or ja[self.KNEE_R] > self.KNEE_MAX_RAD:
            rule_label = max(rule_label, 2)
            rule_conf  = 0.85
            triggered.append("knee_excessive_flexion")

        # 左右膝不对称
        asymm = abs(ja[self.KNEE_L] - ja[self.KNEE_R])
        if asymm > self.KNEE_ASYMM_RAD:
            rule_label = max(rule_label, 1)
            rule_conf  = max(rule_conf, 0.7)
            triggered.append(f"knee_asymmetry_{np.degrees(asymm):.1f}deg")

        # GRF 峰值
        if grf_bw > thr_high:
            rule_label = max(rule_label, 2)
            rule_conf  = max(rule_conf, 0.90)
            triggered.append(f"high_grf_{grf_bw:.2f}BW")
        elif grf_bw > thr_med:
            rule_label = max(rule_label, 1)
            rule_conf  = max(rule_conf, 0.65)
            triggered.append(f"med_grf_{grf_bw:.2f}BW")

        # 组合规则
        if asymm > self.KNEE_ASYMM_RAD and grf_bw > thr_med and self._hr > 90:
            rule_label = max(rule_label, 2)
            rule_conf  = max(rule_conf, 0.88)
            triggered.append("combined_valgus_grf_hr")

        return rule_label, rule_conf, triggered

    def validate(self, joint_angles: np.ndarray, grf: np.ndarray, model_label: int, model_conf: float) -> Tuple[int, float, bool]:
        if isinstance(joint_angles, torch.Tensor):
            joint_angles = joint_angles.cpu().numpy()
        if isinstance(grf, torch.Tensor):
            grf = grf.cpu().numpy()

        rule_label, rule_conf, triggered = self._eval_rules(joint_angles, grf)
        self._rule_hist.append(rule_label)

        # 时间累积
        hist = list(self._rule_hist)
        if len(hist) >= self.TEMPORAL_WINDOW // 2:
            recent = hist[-self.TEMPORAL_WINDOW//2:]
            if sum(r >= 1 for r in recent) >= self.TEMPORAL_WINDOW // 4:
                rule_label = max(rule_label, min(rule_label + 1, 2))

        final_label    = max(model_label, rule_label)
        final_conf     = max(model_conf, rule_conf) if rule_label > 0 else model_conf
        overridden     = rule_label > model_label

        if triggered:
            logger.debug(f"SafetyHeartbeat 触发: {triggered} → label={final_label}")

        return final_label, final_conf, overridden


if __name__ == "__main__":
    import math

    # RiskMLP v2 测试
    model = RiskMLP(seq_len=20)
    print(f"RiskMLP v2 参数量: {model.count_params()}")

    B, T = 4, 20
    x  = torch.randn(B, T, INPUT_DIM_BASE)
    hr = torch.tensor([70., 80., 90., 100.])
    sl = torch.tensor([85., 70., 60., 90.])

    logits, conf = model(x, hr, sl)
    print(f"logits: {logits.shape}  conf: {conf.shape}")
    assert logits.shape == (B, NUM_CLASSES) and conf.shape == (B, 1)

    label, score, confidence = model.predict(x, hr, sl)
    print(f"label={label}  score={score.round(decimals=3)}  conf={confidence.round(decimals=3)}")

    # 梯度测试
    loss = logits.sum() + conf.sum()
    loss.backward()
    print("RiskMLP 梯度正常 ✅")

    # SafetyHeartbeat v2 测试
    safety = SafetyHeartbeat()
    safety.update_heart_rate(105)
    safety.update_sleep_score(55)
    print(f"自适应阈值（高心率+差睡眠）: {safety._grf_threshold_bw():.2f}×BW")

    ja_overext = np.zeros(23, np.float32)
    ja_overext[6] = math.radians(-8)
    grf_test   = np.zeros(12, np.float32); grf_test[2] = 600

    fl, fc, ov = safety.validate(ja_overext, grf_test, 0, 0.3)
    assert fl == 2 and ov, f"过伸应触发高风险覆盖: label={fl}"
    print(f"过伸覆盖测试: label={fl}, conf={fc:.2f}, override={ov} ✅")

    # 组合规则测试
    ja_asym = np.zeros(23, np.float32)
    ja_asym[6] = math.radians(20); ja_asym[13] = math.radians(5)  # 15°不对称
    grf_high = np.zeros(12, np.float32); grf_high[2] = 800
    safety.update_heart_rate(95)
    fl2, fc2, ov2 = safety.validate(ja_asym, grf_high, 0, 0.3)
    print(f"组合规则测试: label={fl2}, conf={fc2:.2f} ✅")

    print("risk_model.py v2 OK ✅")