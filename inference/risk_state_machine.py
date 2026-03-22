"""
inference/risk_state_machine.py
风险状态机（Risk State Machine）
将单帧模型输出转化为时间连续、稳健的风险决策系统。

状态：SAFE → WARNING → DANGER（及回退路径）
特性：
  1. 迟滞（Hysteresis）：升级需连续 N 帧，降级需更多帧
     避免在阈值附近震荡
  2. 锁定（Lock）：DANGER 状态持续最少 M 帧才能退出
     对应生理预警"一旦触发，不能立即消除"
  3. 置信度门控：低置信度输出不触发状态升级
  4. OPPO 健康联动：心率/睡眠影响升级阈值

状态转移矩阵：
  SAFE    → WARNING  : risk_score > t_warn 连续 K_up=3 帧
  WARNING → DANGER   : risk_score > t_danger 连续 K_up=2 帧
  DANGER  → WARNING  : risk_score < t_warn 连续 K_dn=8 帧（锁定 lock=10）
  WARNING → SAFE     : risk_score < t_safe 连续 K_dn=5 帧
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RiskState(IntEnum):
    SAFE    = 0
    WARNING = 1
    DANGER  = 2


@dataclass
class StateTransition:
    """记录一次状态转移事件"""
    from_state:  RiskState
    to_state:    RiskState
    frame_idx:   int
    risk_score:  float
    trigger:     str        # 触发原因描述


@dataclass
class RiskStateMachineConfig:
    """状态机超参数（全部可从 config.yaml 覆盖）"""
    # 升级阈值
    thresh_warn:   float = 0.40   # score > 此值才可升入 WARNING
    thresh_danger: float = 0.70   # score > 此值才可升入 DANGER

    # 降级阈值（带迟滞）
    thresh_safe:   float = 0.25   # score < 此值才可降回 SAFE
    thresh_warn_dn:float = 0.50   # score < 此值才可降回 WARNING

    # 连续帧计数
    k_up_warn:   int = 3    # 升入 WARNING 需连续多少帧
    k_up_danger: int = 2    # 升入 DANGER 需连续多少帧
    k_dn_safe:   int = 5    # 降回 SAFE 需连续多少帧
    k_dn_warn:   int = 8    # 降回 WARNING 需连续多少帧

    # 锁定：DANGER 状态最少保持帧数（@60Hz: 10帧=167ms）
    danger_lock_frames: int = 10

    # 置信度门控：置信度 < 此值的帧不触发升级
    min_confidence_for_upgrade: float = 0.5

    # 评分平滑（EWMA）
    ewma_alpha: float = 0.3   # 0=无平滑, 1=无记忆


class RiskStateMachine:
    """
    时间连续风险状态机

    设计原则：
    - 宁可假阳性（多报警）不可假阴性（漏报警）→ 升级快，降级慢
    - 置信度低时保守处理，不主动升级
    - OPPO 健康数据动态调整阈值

    典型时序（@60Hz）：
      帧0-2:  SAFE  （score=0.3）
      帧3-4:  SAFE→WARNING  （连续3帧 score>0.4）
      帧5-6:  WARNING→DANGER  （连续2帧 score>0.7）
      帧7-16: DANGER（锁定10帧）
      帧17-24:DANGER→WARNING  （连续8帧 score<0.5）
      帧25-29:WARNING→SAFE  （连续5帧 score<0.25）
    """

    def __init__(self, cfg: Optional[RiskStateMachineConfig] = None):
        self.cfg   = cfg or RiskStateMachineConfig()
        self._state: RiskState = RiskState.SAFE
        self._frame_idx:    int = 0
        self._danger_since: int = -1   # 进入 DANGER 的帧序号

        # 升降级计数器
        self._up_counter:   int = 0
        self._dn_counter:   int = 0

        # 评分历史（EWMA）
        self._ewma_score: float = 0.0

        # 事件日志
        self._transitions: List[StateTransition] = []

        # 当前帧信息（供外部读取）
        self.current_risk_score: float = 0.0

    # ------------------------------------------------------------------
    # OPPO Health 联动
    # ------------------------------------------------------------------

    def _adjusted_thresholds(
        self,
        heart_rate:  float = 70.0,
        sleep_score: float = 80.0,
    ) -> tuple:
        """
        根据心率和睡眠质量动态调整阈值：
          心率↑ → 阈值↓（更容易触发）
          睡眠↓ → 阈值↓
        """
        cfg = self.cfg
        # 心率因子：HR>80 开始降低阈值
        hr_f    = 1.0 - 0.15 * max(0, (heart_rate  - 80) / 40)
        # 睡眠因子：睡眠<70 开始降低阈值
        sleep_f = 0.85 + 0.15 * min(1.0, sleep_score / 80)
        factor  = float(np.clip(hr_f * sleep_f, 0.6, 1.1))

        return (
            cfg.thresh_warn   * factor,
            cfg.thresh_danger * factor,
            cfg.thresh_safe   * factor,
        )

    # ------------------------------------------------------------------
    # 状态更新
    # ------------------------------------------------------------------

    def update(
        self,
        risk_score:  float,
        confidence:  float = 1.0,
        heart_rate:  float = 70.0,
        sleep_score: float = 80.0,
    ) -> RiskState:
        """
        输入单帧模型输出，更新状态机，返回当前状态。

        Args:
            risk_score:  0~1 高风险概率（来自 RiskMLP）
            confidence:  0~1 模型置信度
            heart_rate:  来自 OPPO Health SDK（bpm）
            sleep_score: 来自 OPPO Health SDK（0-100）

        Returns:
            当前 RiskState
        """
        cfg = self.cfg

        # EWMA 平滑
        self._ewma_score = (cfg.ewma_alpha * risk_score
                          + (1 - cfg.ewma_alpha) * self._ewma_score)
        score = self._ewma_score
        self.current_risk_score = score

        # 动态阈值
        t_warn, t_danger, t_safe = self._adjusted_thresholds(heart_rate, sleep_score)
        t_warn_dn = cfg.thresh_warn_dn   # 降级阈值不随状态调整

        old_state = self._state

        # ---------- 状态转移逻辑 ----------

        if self._state == RiskState.SAFE:
            if score > t_warn and confidence >= cfg.min_confidence_for_upgrade:
                self._up_counter += 1
                if self._up_counter >= cfg.k_up_warn:
                    self._transition_to(RiskState.WARNING, score, "score_threshold")
            else:
                self._up_counter = max(0, self._up_counter - 1)

        elif self._state == RiskState.WARNING:
            if score > t_danger and confidence >= cfg.min_confidence_for_upgrade:
                self._up_counter += 1
                if self._up_counter >= cfg.k_up_danger:
                    self._transition_to(RiskState.DANGER, score, "score_threshold")
            elif score < t_safe:
                self._dn_counter += 1
                self._up_counter = 0
                if self._dn_counter >= cfg.k_dn_safe:
                    self._transition_to(RiskState.SAFE, score, "score_below_safe")
            else:
                self._up_counter = max(0, self._up_counter - 1)
                self._dn_counter = max(0, self._dn_counter - 1)

        elif self._state == RiskState.DANGER:
            # 锁定检查：必须在 DANGER 够久才能降级
            frames_in_danger = self._frame_idx - self._danger_since
            lock_ok = frames_in_danger >= cfg.danger_lock_frames

            if score < t_warn_dn and lock_ok:
                self._dn_counter += 1
                if self._dn_counter >= cfg.k_dn_warn:
                    self._transition_to(RiskState.WARNING, score, "score_below_warn")
            else:
                self._dn_counter = max(0, self._dn_counter - 1)

        self._frame_idx += 1
        return self._state

    def _transition_to(self, new_state: RiskState, score: float, trigger: str) -> None:
        ev = StateTransition(
            from_state=self._state,
            to_state=new_state,
            frame_idx=self._frame_idx,
            risk_score=score,
            trigger=trigger,
        )
        self._transitions.append(ev)
        logger.debug(
            f"[FSM] {self._state.name} → {new_state.name} "
            f"(score={score:.3f}, {trigger})"
        )

        if new_state == RiskState.DANGER:
            self._danger_since = self._frame_idx

        self._state      = new_state
        self._up_counter = 0
        self._dn_counter = 0

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------

    @property
    def state(self) -> RiskState:
        return self._state

    @property
    def state_int(self) -> int:
        return int(self._state)

    @property
    def state_name(self) -> str:
        return self._state.name

    @property
    def transitions(self) -> List[StateTransition]:
        return self._transitions

    def reset(self) -> None:
        """重置状态机（新用户开始使用时调用）"""
        self.__init__(self.cfg)

    def summary(self) -> dict:
        """输出状态机统计（用于训练日志）"""
        return {
            "current_state":   self.state_name,
            "total_frames":    self._frame_idx,
            "n_transitions":   len(self._transitions),
            "ewma_score":      round(self._ewma_score, 4),
            "transitions":     [
                {"from": t.from_state.name, "to": t.to_state.name,
                 "frame": t.frame_idx, "score": round(t.risk_score, 3),
                 "trigger": t.trigger}
                for t in self._transitions
            ],
        }


if __name__ == "__main__":
    print("=== RiskStateMachine 测试 ===")
    fsm = RiskStateMachine()

    # 模拟场景：正常→警告→危险→恢复
    scenario = (
        [0.1] * 5        # SAFE
      + [0.6] * 5        # 升入 WARNING（>0.4 连续3帧）
      + [0.85] * 5       # 升入 DANGER（>0.7 连续2帧）
      + [0.3] * 15       # 尝试降级（需锁定10帧 + 连续8帧）
      + [0.1] * 10       # 降回 SAFE
    )

    states_seen = []
    for i, score in enumerate(scenario):
        state = fsm.update(score, confidence=0.9)
        states_seen.append(state)
        if i in [4, 9, 14, 29, 39]:
            print(f"  帧{i:02d}: {state.name} (score={score:.2f} ewma={fsm._ewma_score:.3f})")

    # 验证必须经历所有三个状态
    state_names = set(s.name for s in states_seen)
    assert "SAFE"    in state_names, "未经历 SAFE"
    assert "WARNING" in state_names, "未经历 WARNING"
    assert "DANGER"  in state_names, "未经历 DANGER"
    print(f"  状态历程: {[s.name for s in states_seen[::8]]}")

    # 验证锁定机制
    fsm2 = RiskStateMachine()
    for _ in range(10):
        fsm2.update(0.9, confidence=1.0)
    assert fsm2.state == RiskState.DANGER
    # 立刻输入低分，不应立即退出
    fsm2.update(0.1, confidence=1.0)
    assert fsm2.state == RiskState.DANGER, "锁定机制失效"
    print("  锁定机制正常 ✅")

    # OPPO 健康联动
    fsm3 = RiskStateMachine()
    t_warn_n, t_d_n, _ = fsm3._adjusted_thresholds(70, 80)
    t_warn_f, t_d_f, _ = fsm3._adjusted_thresholds(110, 50)
    assert t_warn_f < t_warn_n, "疲劳状态阈值应更低"
    print(f"  正常阈值 warn={t_warn_n:.3f} | 疲劳阈值 warn={t_warn_f:.3f} ✅")

    summary = fsm.summary()
    print(f"  状态机摘要: {summary['n_transitions']} 次转移")
    print("risk_state_machine.py OK ✅")