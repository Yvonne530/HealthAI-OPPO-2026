import os
import json
import numpy as np
import nimblephysics as nimble

def safe_shape(x):
    try:
        if hasattr(x, 'shape'): return list(x.shape)
        if isinstance(x, (list, dict)): return [len(x)]
        return "unknown"
    except:
        return "unknown"

def inspect_b3d_full(file_path):
    print("=" * 60)
    print(f"📂 解析: {file_path}")
    print("=" * 60)

    subject = nimble.biomechanics.SubjectOnDisk(file_path)
    summary = {"num_trials": subject.getNumTrials(), "num_dofs": subject.getNumDofs()}
    
    marker_names = []
    trials_info = []
    global_stats = {
        "total_frames": 0,
        "missing_marker_frames": 0,
        "has_emg": False,
        "has_imu": False,
    }

    for trial_idx in range(subject.getNumTrials()):
        trial_len = subject.getTrialLength(trial_idx)
        global_stats["total_frames"] += trial_len
        
        # 读取该 Trial 的所有帧
        frames = subject.readFrames(trial_idx, 0, trial_len)
        if not frames: continue

        # 使用第一帧作为结构参考
        f0 = frames[0]
        trial_stat = {"trial_id": trial_idx, "length": trial_len}

        # --- Marker 处理 ---
        if hasattr(f0, "markerObservations"):
            m_obs = f0.markerObservations # 这是一个列表
            trial_stat["marker_count"] = len(m_obs)
            
            # 如果还没拿到 marker 名字，尝试从第一个有效帧提取
            if len(m_obs) > 0 and not marker_names:
                # Nimble 的 MarkerObservation 对象通常有 name 属性
                marker_names = [m.name if hasattr(m, 'name') else f"marker_{i}" for i, m in enumerate(m_obs)]

        # --- DOF / 状态 ---
        if hasattr(f0, "pos"):
            trial_stat["dof_shape"] = safe_shape(f0.pos)

        # --- EMG ---
        if hasattr(f0, "emgSignals") and len(f0.emgSignals) > 0:
            global_stats["has_emg"] = True
            trial_stat["emg_dim"] = safe_shape(f0.emgSignals)

        # --- IMU (Acc/Gyro) ---
        if hasattr(f0, "accObservations") and len(f0.accObservations) > 0:
            global_stats["has_imu"] = True
            trial_stat["acc_dim"] = safe_shape(f0.accObservations)

        # --- 缺失检测 ---
        missing_count = 0
        for f in frames:
            if not hasattr(f, "markerObservations") or len(f.markerObservations) == 0:
                missing_count += 1
        global_stats["missing_marker_frames"] += missing_count

        trials_info.append(trial_stat)
        print(f"✅ Trial {trial_idx} 完成 | 帧数: {trial_len} | Markers: {trial_stat.get('marker_count', 0)}")

    summary["marker_names"] = marker_names
    summary["num_markers"] = len(marker_names)
    summary["trials"] = trials_info
    summary["global_stats"] = global_stats

    return summary

def save_summary(summary, path="b3d_full_summary.json"):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)): return int(obj)
            if isinstance(obj, (np.floating, np.float64)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=MyEncoder)
    print(f"\n🚀 报告已生成: {path}")

if __name__ == "__main__":
    file_path = "/mnt/d/Camargo2021_Formatted_No_Arm/AB30_split5/AB30_split5.b3d"
    
    if os.path.exists(file_path):
        try:
            data_summary = inspect_b3d_full(file_path)
            save_summary(data_summary)
        except Exception as e:
            print(f"❌ 运行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"文件未找到: {file_path}")
