"""
inference/heytap_health_adapter.py
OPPO 健康服务 SDK Python 适配器
在 PC 训练环境中提供 mock 实现，Android 端替换为真实 JNI 调用
"""
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HealthDataType:
    """对应 OPPO SDK DataType"""
    HEART_RATE       = "TYPE_HEART_RATE"
    SLEEP_COUNT      = "TYPE_SLEEP_COUNT"
    BLOOD_OXYGEN     = "TYPE_BLOOD_OXYGEN"
    DAILY_ACTIVITY   = "TYPE_DAILY_ACTIVITY"


class HeytapHealthAdapter:
    """
    OPPO HeytapHealth SDK 适配器
    - PC 环境：Mock 模式（随机生成生理数据）
    - Android 环境：替换为 JNI/Kotlin 回调

    集成点：
      1. 实时心率 → SafetyHeartbeat.update_heart_rate()
      2. 睡眠评分 → SafetyHeartbeat.update_sleep_score()
      3. 运动数据写回 → write_risk_analysis()
    """

    def __init__(self, mock: bool = True):
        self._mock      = mock
        self._hr        = 70.0
        self._sleep     = 80.0
        self._callbacks: dict = {}
        self._thread: Optional[threading.Thread] = None
        self._running   = False

        if mock:
            logger.info("[HeytapHealth] Mock 模式启动")
        else:
            logger.info("[HeytapHealth] 真实 SDK 模式（需 Android 环境）")

    # ------------------------------------------------------------------
    # 授权
    # ------------------------------------------------------------------

    def request_authorization(self, callback: Callable) -> None:
        """
        请求用户授权（Android 端跳转授权页面）
        对应 SDK：HeytapHealthApi.getInstance().authorityApi().request()
        """
        if self._mock:
            logger.info("[HeytapHealth] Mock 授权成功")
            callback(success=True)
        else:
            # Android JNI 调用占位
            raise NotImplementedError("在 Android 端通过 JNI 调用 SDK")

    def validate_authorization(self) -> list:
        """
        校验授权权限范围
        对应 SDK：HeytapHealthApi.getInstance().authorityApi().valid()
        """
        if self._mock:
            return [HealthDataType.HEART_RATE, HealthDataType.SLEEP_COUNT]
        raise NotImplementedError("在 Android 端通过 JNI 调用 SDK")

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------

    def get_heart_rate(self) -> float:
        """读取最新实时心率（bpm）"""
        if self._mock:
            # 模拟心率在 60-100 之间波动
            import numpy as np
            self._hr += np.random.normal(0, 0.5)
            self._hr  = float(np.clip(self._hr, 50, 120))
            return self._hr
        raise NotImplementedError("在 Android 端通过 JNI 调用 SDK")

    def get_sleep_score(self) -> float:
        """读取昨晚睡眠评分（0-100）"""
        if self._mock:
            return self._sleep
        raise NotImplementedError("在 Android 端通过 JNI 调用 SDK")

    def start_realtime_hr(self, callback: Callable[[float], None]) -> None:
        """
        启动实时心率订阅
        对应 SDK：TYPE_HEART_RATE @ 60Hz
        """
        if not self._mock:
            raise NotImplementedError("在 Android 端通过 JNI 调用 SDK")

        self._running = True
        def _worker():
            while self._running:
                hr = self.get_heart_rate()
                callback(hr)
                time.sleep(1.0)   # 1 Hz 心率更新

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        logger.info("[HeytapHealth] 实时心率订阅已启动（Mock）")

    def stop_realtime_hr(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("[HeytapHealth] 实时心率订阅已停止")

    # ------------------------------------------------------------------
    # 数据写回
    # ------------------------------------------------------------------

    def write_risk_analysis(
        self,
        start_time_ms: int,
        duration_ms:   int,
        risk_score:    float,
        risk_level:    int,
        calories:      float = 0.0,
    ) -> bool:
        """
        将 ACL 风险分析结果写回 OPPO 健康 App。

        对应 SDK：
          DataType.TYPE_GYM_STRENGTH_TRAINING
          DataType.TYPE_TRAINING_ACTION
          HeytapHealthApi.getInstance().dataApi().insert()

        Android Kotlin 等效代码：
          val baseDataPoint = DataPoint.builder(DataType.TYPE_GYM_STRENGTH_TRAINING)
              .setStartTimeStamp(startTime)
              .setElement(Element.ELEMENT_TRAINING_TITLE, "ACL风险监测")
              .setElement(Element.ELEMENT_DURATION, duration)
              .setElement(Element.ELEMENT_CALORIE, calories)
              .setElement(Element.ELEMENT_SPORT_MODE, SportMode.GYM_STRENGTH_TRAINING)
              .build()
          val actionDataPoint = DataPoint.builder(DataType.TYPE_TRAINING_ACTION)
              .setStartTimeStamp(startTime)
              .setElement(Element.ELEMENT_TRAINING_ACTION, "ACL风险评估")
              .setElement(Element.ELEMENT_ACTION_COUNTERWEIGHT_1, riskScore)
              .setElement(Element.ELEMENT_TIMES, riskLevel)
              .build()
        """
        if self._mock:
            logger.info(
                f"[HeytapHealth] Mock 写回: 风险={risk_level} "
                f"分数={risk_score:.3f} 时长={duration_ms}ms"
            )
            return True
        raise NotImplementedError("在 Android 端通过 JNI 调用 SDK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    adapter = HeytapHealthAdapter(mock=True)
    scope = adapter.validate_authorization()
    print(f"授权范围: {scope}")

    def on_hr(hr):
        print(f"  心率: {hr:.1f} bpm")

    adapter.start_realtime_hr(on_hr)
    time.sleep(3)
    adapter.stop_realtime_hr()

    ok = adapter.write_risk_analysis(
        start_time_ms=int(time.time() * 1000),
        duration_ms=30000,
        risk_score=0.75,
        risk_level=2,
    )
    print(f"写回结果: {ok}")
    print("heytap_health_adapter.py OK ✅")