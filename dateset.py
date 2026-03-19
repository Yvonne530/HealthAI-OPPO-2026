import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class ACLDataset(Dataset):
    def __init__(self, data_root="/mnt/d/cam", seq_len=50, stride=5, cache_file="data_index.pkl", refresh_cache=False):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.stride = stride
        self.input_dof = 23  # 对齐 Gait2392 的关键下肢/躯干关节
        
        if not refresh_cache and Path(cache_file).exists():
            print(f"📦 加载索引缓存: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.index = cache_data['index']
                self.file_map = cache_data['file_map']
        else:
            self.index, self.file_map = self._build_index()
            with open(cache_file, 'wb') as f:
                pickle.dump({'index': self.index, 'file_map': self.file_map}, f)

        # 核心：提供给 train.py 进行 Subject-wise Split
        self.subjects = [item[3] for item in self.index]
        print(f"✅ 数据集就绪: {len(self.index)} 条序列, 来自 {len(set(self.subjects))} 个独立受试者")

    def _build_index(self):
        index = []
        file_map = {}
        # 兼容 Windows/Linux 路径扫描
        b3d_files = list(self.data_root.rglob("*.b3d"))
        if not b3d_files:
            print(f"❌ 错误：在 {self.data_root} 下未找到 .b3d 文件，请检查路径。")
            
        for f_idx, path in enumerate(tqdm(b3d_files, desc="🔍 扫描物理数据")):
            try:
                subject = nimble.biomechanics.SubjectOnDisk(str(path))
                # 自动获取受试者体重，默认70kg
                mass = 70.0
                try:
                    if hasattr(subject, 'getMassKG'): mass = subject.getMassKG()
                    elif hasattr(subject, 'getMass'): mass = subject.getMass()
                except: pass
                
                # 提取受试者ID（假设文件夹名为ID，如 AB06）
                sub_id = path.parent.name if path.parent.name != self.data_root.name else path.stem
                
                file_map[f_idx] = {'path': str(path), 'mass': mass}
                
                for t_idx in range(subject.getNumTrials()):
                    t_len = subject.getTrialLength(t_idx)
                    if t_len >= self.seq_len:
                        for start in range(0, t_len - self.seq_len + 1, self.stride):
                            # 索引包含：文件ID, Trial ID, 起始帧, 受试者ID
                            index.append((f_idx, t_idx, start, sub_id))
            except Exception as e:
                print(f"⚠️ 跳过文件 {path.name}: {e}")
        return index, file_map

    def __len__(self):
        return len(self.index)

    def _get_grf_from_frame(self, frame, mass):
        """地反力标准化处理 (N/kg)"""
        forces = []
        if hasattr(frame, 'externalForces'): forces = frame.externalForces
        elif hasattr(frame, 'grf'): forces = frame.grf
        
        if len(forces) >= 2:
            # 提取双脚 3轴力并除以体重标准化
            g_r = np.array(forces[0].force) / mass if forces[0].force is not None else np.zeros(3)
            g_l = np.array(forces[1].force) / mass if forces[1].force is not None else np.zeros(3)
            return np.concatenate([g_r, g_l])
        return np.zeros(6)

    def __getitem__(self, idx):
        try:
            f_idx, t_idx, start, _ = self.index[idx]
            file_info = self.file_map[f_idx]
            
            # 重新加载对象以保证线程安全（适配 num_workers > 0）
            subject = nimble.biomechanics.SubjectOnDisk(file_info['path'])
            frames = subject.readFrames(t_idx, start, self.seq_len)
            mass = file_info['mass']

            pos_list, grf_list = [], []
            for f in frames:
                # 1. 提取自由度坐标 (Pos)
                # 针对 Gait2392 提取前 23 个核心自由度，确保维度对齐
                raw_p = f.processingPasses[0].pos if len(f.processingPasses) > 0 else np.zeros(self.input_dof)
                pos_list.append(raw_p[:self.input_dof]) 
                
                # 2. 提取地反力 (GRF)
                grf_list.append(self._get_grf_from_frame(f, mass))

            pos = np.array(pos_list, dtype=np.float32) # [50, 23]
            
            # 3. 计算一阶/二阶差分 (对齐 FNO 输入的 69 维)
            dt = 1/60.0 
            vel = np.zeros_like(pos)
            vel[1:] = (pos[1:] - pos[:-1]) / dt
            acc = np.zeros_like(vel)
            acc[1:] = (vel[1:] - vel[:-1]) / dt
            
            # 拼接特征: [50, 23*3=69]
            X = np.concatenate([pos, vel, acc], axis=-1)
            
            # 返回元组 (X, y)，适配 train.py
            return torch.from_numpy(X).float(), torch.from_numpy(np.array(grf_list, dtype=np.float32)).float()
            
        except Exception as e:
            # 容错处理：若读取失败则尝试下一条
            return self.__getitem__((idx + 1) % len(self.index))

if __name__ == "__main__":
    # 测试代码
    import os
    test_path = "/mnt/d/cam" if os.path.exists("/mnt/d/cam") else "D:/cam"
    ds = ACLDataset(data_root=test_path)
    if len(ds) > 0:
        X, y = ds[0]
        print(f"🚀 数据集对齐成功!")
        print(f"输入 X (pos+vel+acc): {X.shape} (预期 [50, 69])")
        print(f"标签 Y (双脚GRF): {y.shape} (预期 [50, 6])")
        print(f"受试者 ID 示例: {ds.subjects[0]}")