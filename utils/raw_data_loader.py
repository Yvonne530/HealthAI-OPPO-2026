"""
utils/raw_data_loader.py
OpenSim Camargo2021 原始数据加载器
支持格式：AB06/10_09_18/{levelground|ramp|stair|treadmill}/
  - ik/   → 关节角度 (.sto)
  - id/   → 关节力矩 (.sto)
  - fp/   → 测力板 GRF (.sto/.mat)
  - markers/ → 标记点 (.sto/.mat)
  - conditions/ → trial 裁剪索引 (.mat)
  - emg/  → 肌电（可选）

输出格式与 .b3d loader 完全一致（RGSample），可直接接入训练流水线。
优势：
  - 可使用原始 200Hz 原始 GRF（fp/）而非 .b3d 的重建值
  - EMG 可作为额外监督信号（physics weight）
  - id/ 的关节力矩可用于 PINN 约束
"""
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .rg_sample import RGSample

logger = logging.getLogger(__name__)

TARGET_FPS    = 60
N_MARKERS     = 28
N_DOFS        = 23

# OpenSim gait2392 关节名（按标准顺序）
DOF_ORDER = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "neck_flexion", "neck_bending", "neck_rotation",
]

# 测力板 GRF 列名（左右脚合并为 12 维）
GRF_COLS_L = ["ground_force_vx", "ground_force_vy", "ground_force_vz",
              "ground_torque_x", "ground_torque_y", "ground_torque_z"]
GRF_COLS_R = ["1_ground_force_vx", "1_ground_force_vy", "1_ground_force_vz",
              "1_ground_torque_x", "1_ground_torque_y", "1_ground_torque_z"]


# =====================================================================
# .sto 文件解析
# =====================================================================

def parse_sto(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    解析 OpenSim .sto 文件 → (column_names, data_array)
    .sto 格式：
      行1: 文件名
      行2~N: "key=value" 元数据
      endheader 后为 tab 分隔数据
    """
    headers_done = False
    col_names: List[str] = []
    rows: List[List[float]] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "endheader":
                headers_done = True
                continue
            if not headers_done:
                continue
            # 第一行数据行是列名
            if not col_names:
                col_names = line.split("\t")
                continue
            try:
                rows.append([float(x) for x in line.split("\t")])
            except ValueError:
                continue

    if not rows:
        return col_names, np.zeros((0, max(len(col_names), 1)), dtype=np.float32)
    return col_names, np.array(rows, dtype=np.float32)


def parse_mat(filepath: str) -> Optional[Dict]:
    """解析 .mat 文件（尝试 scipy.io，失败时返回 None）"""
    try:
        from scipy.io import loadmat
        return loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        logger.debug(f"  .mat 解析失败 {filepath}: {e}")
        return None


# =====================================================================
# 单 trial 加载
# =====================================================================

def _get_sto_in_dir(dirpath: str, keyword: str = "") -> List[str]:
    """获取目录下匹配关键词的 .sto 文件列表"""
    if not os.path.isdir(dirpath):
        return []
    return sorted(
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if f.endswith(".sto") and keyword.lower() in f.lower()
    )


def load_opensim_trial(
    subject_dir: str,
    activity:    str,           # "levelground" / "ramp" / "stair" / "treadmill"
    session:     str = "",      # 日期文件夹，如 "10_09_18"（空=自动寻找）
    target_fps:  int = TARGET_FPS,
) -> Iterator[RGSample]:
    """
    加载单个 subject 的单个 activity 的所有 trial。

    目录结构：
      subject_dir/[session]/[activity]/
        ik/   *.sto  → 关节角度（度，需转弧度）
        fp/   *.sto  → 测力板 GRF（牛顿）
        id/   *.sto  → 关节力矩（用于 physics_weight）
        markers/ *.sto → 标记点坐标（可选）

    Returns:
        Iterator[RGSample]（已降采样至 target_fps）
    """
    subj_path = Path(subject_dir)

    # 自动寻找会话目录
    if session:
        sess_path = subj_path / session
    else:
        sessions = [d for d in subj_path.iterdir()
                    if d.is_dir() and not d.name.startswith(".")]
        if not sessions:
            logger.warning(f"[RawLoader] 无会话目录: {subject_dir}")
            return
        sess_path = sessions[0]

    act_path = sess_path / activity
    if not act_path.exists():
        logger.warning(f"[RawLoader] 活动目录不存在: {act_path}")
        return

    ik_dir      = act_path / "ik"
    fp_dir      = act_path / "fp"
    id_dir      = act_path / "id"
    markers_dir = act_path / "markers"

    # 获取 ik 文件列表（以 ik 为主索引）
    ik_files = _get_sto_in_dir(str(ik_dir))
    if not ik_files:
        logger.warning(f"[RawLoader] 无 IK 数据: {ik_dir}")
        return

    logger.info(f"[RawLoader] {subj_path.name}/{activity}: {len(ik_files)} 个 trial")

    for ik_file in ik_files:
        trial_name = Path(ik_file).stem
        try:
            yield from _load_one_trial(
                trial_name, ik_file,
                str(fp_dir), str(id_dir), str(markers_dir),
                target_fps
            )
        except Exception as e:
            logger.warning(f"  trial {trial_name} 失败: {e}")


def _load_one_trial(
    trial_name:  str,
    ik_file:     str,
    fp_dir:      str,
    id_dir:      str,
    markers_dir: str,
    target_fps:  int,
) -> Iterator[RGSample]:
    """解析单个 trial 的所有数据，yield RGSample"""

    # ---- 关节角度（IK，度→弧度）----
    ik_cols, ik_data = parse_sto(ik_file)
    if ik_data.shape[0] < 10:
        return

    time_idx = ik_cols.index("time") if "time" in ik_cols else 0
    timestamps = ik_data[:, time_idx].astype(np.float64)

    joint_angles_mat = _extract_dofs(ik_cols, ik_data)
    if joint_angles_mat is None:
        logger.debug(f"  {trial_name}: IK 列不足，跳过")
        return
    joint_angles_mat = np.deg2rad(joint_angles_mat)   # 度→弧度

    n_frames = joint_angles_mat.shape[0]
    if n_frames < 30:
        return

    # vel/acc 有限差分
    dt = float(np.median(np.diff(timestamps)))
    if dt <= 0:
        dt = 1.0 / 200.0
    vel_mat = np.gradient(joint_angles_mat, dt, axis=0).astype(np.float32)
    acc_mat = np.gradient(vel_mat, dt, axis=0).astype(np.float32)

    # ---- 测力板 GRF ----
    fp_file  = _find_matching_sto(fp_dir, trial_name)
    grf_mat  = np.zeros((n_frames, 12), dtype=np.float32)
    grf_mask_arr = np.ones(n_frames, dtype=np.float32)

    if fp_file:
        fp_cols, fp_data = parse_sto(fp_file)
        grf_mat = _extract_grf(fp_cols, fp_data, timestamps)
    else:
        grf_mask_arr[:] = 0.5   # 无 GRF 数据

    # ---- 关节力矩（逆动力学），用于物理权重 ----
    id_file = _find_matching_sto(id_dir, trial_name)
    physics_weights = np.ones(n_frames, dtype=np.float32)

    if id_file:
        _, id_data = parse_sto(id_file)
        if id_data.shape[0] >= n_frames:
            # 残差越小（力矩越一致）→ 权重越高
            residual = np.abs(id_data[:n_frames, 1:]).mean(axis=1)
            residual = residual / (residual.mean() + 1e-8)
            physics_weights = 1.0 / (residual + 0.1)
            physics_weights = np.clip(physics_weights, 0.1, 10.0).astype(np.float32)

    # ---- 标记点 ----
    mk_file = _find_matching_sto(markers_dir, trial_name)
    markers_mat = np.full((n_frames, N_MARKERS, 3), np.nan, dtype=np.float32)
    if mk_file:
        mk_cols, mk_data = parse_sto(mk_file)
        markers_mat = _extract_markers(mk_cols, mk_data, n_frames)

    # ---- 质心（从骨盆关节角估算）----
    com_mat = _estimate_com(joint_angles_mat)   # (N, 3)

    # ---- 重采样到 target_fps ----
    resampled = _resample_all(
        joint_angles_mat, vel_mat, acc_mat,
        grf_mat, com_mat, markers_mat, grf_mask_arr, physics_weights,
        timestamps, target_fps
    )
    if resampled is None:
        return

    ja_r, vel_r, acc_r, grf_r, com_r, mk_r, mask_r, pw_r, ts_r = resampled
    n_out = ja_r.shape[0]
    seq_id = f"{Path(ik_file).parent.parent.parent.name}__{trial_name}"

    for i in range(n_out):
        s = RGSample()
        s.t             = float(ts_r[i])
        s.original_t    = float(ts_r[i])
        s.sequence_id   = seq_id
        s.joint_angles  = ja_r[i].copy()
        s.joint_vel     = vel_r[i].copy()
        s.joint_acc     = acc_r[i].copy()
        s.markers       = mk_r[i].copy()
        s.grf_left      = grf_r[i, :6].copy()
        s.grf_right     = grf_r[i, 6:12].copy()
        s.grf_mask      = float(mask_r[i])
        s.physics_weight= float(pw_r[i])
        s.com           = com_r[i].copy()
        s.contact_left  = bool(grf_r[i, 2] > 50)
        s.contact_right = bool(grf_r[i, 8] > 50)
        s.valid = True

        assert s.joint_angles.shape == (N_DOFS,)
        assert s.grf_left.shape     == (6,)
        assert s.com.shape          == (3,)
        yield s


# =====================================================================
# 辅助函数
# =====================================================================

def _find_matching_sto(dirpath: str, trial_name: str) -> Optional[str]:
    """在目录中寻找与 trial_name 对应的 .sto 文件"""
    if not os.path.isdir(dirpath):
        return None
    for f in os.listdir(dirpath):
        if f.endswith(".sto") and trial_name in f:
            return os.path.join(dirpath, f)
    return None


def _extract_dofs(cols: List[str], data: np.ndarray) -> Optional[np.ndarray]:
    """从 IK 列中按 DOF_ORDER 提取关节角度矩阵"""
    result = np.zeros((data.shape[0], N_DOFS), dtype=np.float32)
    found  = 0
    for i, dof in enumerate(DOF_ORDER):
        if dof in cols:
            result[:, i] = data[:, cols.index(dof)]
            found += 1
    if found < N_DOFS // 2:   # 至少找到一半才认为有效
        return None
    return result


def _extract_grf(
    cols:      List[str],
    fp_data:   np.ndarray,
    target_ts: np.ndarray,
) -> np.ndarray:
    """
    从测力板 .sto 提取 GRF，插值对齐到 IK 时间轴。
    返回 (N, 12) float32
    """
    n_target = len(target_ts)
    result   = np.zeros((n_target, 12), dtype=np.float32)

    if fp_data.shape[0] < 2 or "time" not in cols:
        return result

    t_fp = fp_data[:, cols.index("time")].astype(np.float64)

    def _get_col(name: str) -> np.ndarray:
        return fp_data[:, cols.index(name)].astype(np.float64) if name in cols \
               else np.zeros(len(t_fp))

    grf_cols = GRF_COLS_L + GRF_COLS_R
    for j, col in enumerate(grf_cols):
        raw_col = _get_col(col)
        if len(t_fp) >= 2:
            try:
                f = interp1d(t_fp, raw_col, kind="linear", bounds_error=False,
                             fill_value=(raw_col[0], raw_col[-1]))
                result[:, j] = f(target_ts).astype(np.float32)
            except Exception:
                pass

    return result


def _extract_markers(
    cols:     List[str],
    mk_data:  np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """提取标记点坐标 → (N, 28, 3)"""
    result = np.full((n_frames, N_MARKERS, 3), np.nan, dtype=np.float32)

    # 寻找 X/Y/Z 三列组成的标记点
    # 命名格式通常为 "LASIS_X", "LASIS_Y", "LASIS_Z" 或 "L_ASIS_x"
    used = 0
    i    = 0
    while i < len(cols) - 2 and used < N_MARKERS:
        c = cols[i]
        if c.lower() in ("time", "nans"):
            i += 1; continue
        # 尝试 X/Y/Z 三列
        for sfx in [("_X","_Y","_Z"), ("_x","_y","_z"), ("X","Y","Z")]:
            base = c.rstrip(sfx[0][-1]).rstrip("_")
            cx = f"{base}{sfx[0]}"; cy = f"{base}{sfx[1]}"; cz = f"{base}{sfx[2]}"
            if cx in cols and cy in cols and cz in cols:
                idx = [cols.index(cx), cols.index(cy), cols.index(cz)]
                n   = min(mk_data.shape[0], n_frames)
                result[:n, used, :] = mk_data[:n, idx].astype(np.float32)
                used += 1
                break
        i += 1

    return result


def _estimate_com(joint_angles: np.ndarray) -> np.ndarray:
    """简化质心估算（与 kinematics.py 保持一致，但 numpy 实现）"""
    HIP_L = 3; HIP_R = 10; KNEE_L = 6; KNEE_R = 13
    L_TH  = 0.245; L_SH  = 0.246
    hip_avg  = (joint_angles[:, HIP_L]  + joint_angles[:, HIP_R])  * 0.5
    knee_avg = (joint_angles[:, KNEE_L] + joint_angles[:, KNEE_R]) * 0.5
    y = L_TH * (1 - np.cos(hip_avg)) + L_SH * (1 - np.cos(knee_avg))
    x = L_TH * np.sin(joint_angles[:, 0]) * 0.5
    z = L_TH * np.sin(joint_angles[:, 1]) * 0.5
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _resample_all(
    ja, vel, acc, grf, com, markers, mask, pw, ts, target_fps
) -> Optional[tuple]:
    """插值重采样所有特征到 target_fps"""
    dur = float(ts[-1] - ts[0])
    if dur <= 0 or len(ts) < 4:
        return None
    n_out  = max(int(round(dur * target_fps)), 2)
    ts_out = np.linspace(ts[0], ts[-1], n_out)

    def _rs1d(arr):
        f = interp1d(ts, arr.astype(np.float64), kind="linear",
                     bounds_error=False, fill_value=(arr[0], arr[-1]))
        return f(ts_out).astype(np.float32)

    def _rs2d(arr):
        out = np.zeros((n_out, arr.shape[1]), np.float32)
        for d in range(arr.shape[1]):
            f = interp1d(ts, arr[:, d].astype(np.float64), kind="linear",
                         bounds_error=False, fill_value=(arr[0,d], arr[-1,d]))
            out[:, d] = f(ts_out).astype(np.float32)
        return out

    mk_out = np.full((n_out, N_MARKERS, 3), np.nan, np.float32)
    for mi in range(N_MARKERS):
        for ci in range(3):
            col = markers[:, mi, ci]
            vld = ~np.isnan(col)
            if vld.sum() < 2:
                continue
            f = interp1d(ts[vld], col[vld].astype(np.float64), kind="linear",
                         bounds_error=False, fill_value=np.nan)
            mk_out[:, mi, ci] = f(ts_out).astype(np.float32)

    return (_rs2d(ja), _rs2d(vel), _rs2d(acc), _rs2d(grf),
            _rs2d(com), mk_out, _rs1d(mask), _rs1d(pw), ts_out)


# =====================================================================
# 批量加载器
# =====================================================================

class RawDataLoader:
    """
    Camargo2021 原始数据批量加载器

    用法：
        loader = RawDataLoader("/mnt/d/Subjects_Part1_AB06-AB14/")
        samples = loader.get_all_samples()

    支持与 B3DLoader 混合使用：
        all_samples = b3d_loader.get_all_samples() + raw_loader.get_all_samples()
    """

    ACTIVITIES = ["levelground", "ramp", "stair", "treadmill"]

    def __init__(
        self,
        root_dir:   str,
        activities: List[str] = None,
        target_fps: int       = TARGET_FPS,
    ):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"根目录不存在: {root_dir}")
        self.root_dir   = root_dir
        self.activities = activities or self.ACTIVITIES
        self.target_fps = target_fps
        self.subject_dirs = self._scan_subjects()
        logger.info(f"[RawDataLoader] {len(self.subject_dirs)} 个受试者")

    def _scan_subjects(self) -> List[str]:
        dirs = []
        for d in sorted(os.listdir(self.root_dir)):
            full = os.path.join(self.root_dir, d)
            if os.path.isdir(full) and d.startswith("AB"):
                dirs.append(full)
        return dirs

    def get_all_samples(self) -> List[RGSample]:
        all_samples: List[RGSample] = []
        for subj_dir in self.subject_dirs:
            subj_name = os.path.basename(subj_dir)
            logger.info(f"  加载受试者 {subj_name}")
            for act in self.activities:
                n_before = len(all_samples)
                try:
                    for s in load_opensim_trial(subj_dir, act,
                                                target_fps=self.target_fps):
                        all_samples.append(s)
                    logger.debug(
                        f"    {act}: +{len(all_samples)-n_before} 帧"
                    )
                except Exception as e:
                    logger.warning(f"    {act} 失败: {e}")
        logger.info(f"[RawDataLoader] 共 {len(all_samples)} 帧")
        return all_samples

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __repr__(self) -> str:
        return (f"RawDataLoader(root={self.root_dir!r}, "
                f"subjects={len(self.subject_dirs)}, "
                f"activities={self.activities})")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Subjects_Part1_AB06-AB14/"
    if not os.path.isdir(root):
        print(f"目录不存在: {root}，跳过真实加载测试")
        # 测试解析函数
        import tempfile, os
        # 创建临时 .sto 文件
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sto', delete=False)
        tmp.write("test\nnRows=3\nnColumns=4\nendheader\n")
        tmp.write("time\tpelvis_tilt\tpelvis_list\tknee_angle_l\n")
        tmp.write("0.0\t5.0\t2.0\t15.0\n")
        tmp.write("0.005\t5.1\t2.1\t15.5\n")
        tmp.write("0.010\t5.2\t2.2\t16.0\n")
        tmp.close()
        cols, data = parse_sto(tmp.name)
        os.unlink(tmp.name)
        print(f"parse_sto 测试: cols={cols[:4]}, shape={data.shape}")
        assert "time" in cols and data.shape == (3, 4)
        print("raw_data_loader.py 结构测试 OK ✅")
        sys.exit(0)

    loader = RawDataLoader(root)
    print(loader)
    samples = loader.get_all_samples()
    if samples:
        s = samples[0]
        print(f"第1帧: {s}")
        print(f"  joint_angles {s.joint_angles.shape}")
        print(f"  grf_left     {s.grf_left.shape}: {np.round(s.grf_left, 1)}")
        print(f"  physics_weight={s.physics_weight:.2f}")
    print(f"共 {len(samples)} 帧")
    print("raw_data_loader.py OK ✅")