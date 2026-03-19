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

        print(f"✅ 数据集就绪: {len(self.index)} 条序列, 来自 {len(self.file_map)} 个文件")

    def _build_index(self):
        index = []
        file_map = {}
        b3d_files = list(self.data_root.rglob("*.b3d"))
        for f_idx, path in enumerate(tqdm(b3d_files, desc="🔍 扫描物理数据")):
            try:
                subject = nimble.biomechanics.SubjectOnDisk(str(path))
                mass = getattr(subject, 'getMassKG', lambda: getattr(subject, 'getMass', lambda: 70.0)())()
                file_map[f_idx] = {'path': str(path), 'mass': mass}
                for t_idx in range(subject.getNumTrials()):
                    t_len = subject.getTrialLength(t_idx)
                    if t_len >= self.seq_len:
                        for start in range(0, t_len - self.seq_len + 1, self.stride):
                            index.append((f_idx, t_idx, start))
            except Exception as e:
                print(f"⚠️ 跳过文件 {path.name}: {e}")
        return index, file_map

    def __len__(self):
        return len(self.index)

    def _get_grf_from_frame(self, frame, mass):
        """兼容性获取地反力"""
        # 尝试几种可能的属性名
        forces = []
        if hasattr(frame, 'externalForces'):
            forces = frame.externalForces
        elif hasattr(frame, 'grf'):
            forces = frame.grf
        
        # 如果找到了力数据
        if len(forces) >= 2:
            # 提取前两个力（通常是右脚和左脚）
            g_r = forces[0].force / mass if forces[0].force is not None else np.zeros(3)
            g_l = forces[1].force / mass if forces[1].force is not None else np.zeros(3)
            return np.concatenate([g_r, g_l])
        return np.zeros(6)

    def __getitem__(self, idx):
        try:
            f_idx, t_idx, start = self.index[idx]
            file_info = self.file_map[f_idx]
            subject = nimble.biomechanics.SubjectOnDisk(file_info['path'])
            frames = subject.readFrames(t_idx, start, self.seq_len)
            mass = file_info['mass']

            pos_list, grf_list, tau_list = [], [], []
            for f in frames:
                # 1. Pos
                p = f.processingPasses[0].pos if len(f.processingPasses) > 0 else np.zeros(50)
                pos_list.append(p)
                
                # 2. GRF (使用兼容逻辑)
                grf_list.append(self._get_grf_from_frame(f, mass))
                
                # 3. Tau
                t = f.processingPasses[-1].tau if len(f.processingPasses) > 0 else np.zeros(50)
                tau_list.append(t)

            pos = np.array(pos_list, dtype=np.float32)
            dt = 1/60.0 
            vel = np.zeros_like(pos)
            vel[1:] = (pos[1:] - pos[:-1]) / dt
            acc = np.zeros_like(vel)
            acc[1:] = (vel[1:] - vel[:-1]) / dt
            
            X = np.concatenate([pos, vel, acc], axis=-1)
            return {
                'X': torch.from_numpy(X),
                'y_grf': torch.from_numpy(np.array(grf_list, dtype=np.float32)),
                'y_tau': torch.from_numpy(np.array(tau_list, dtype=np.float32))
            }
        except Exception as e:
            # 避免死循环：如果连续失败，可能数据路径有问题
            if idx > len(self.index) + 100: raise e
            return self.__getitem__((idx + 1) % len(self.index))

if __name__ == "__main__":
    ds = ACLDataset(data_root="/mnt/d/cam")
    if len(ds) > 0:
        sample = ds[0]
        print(f"🚀 测试成功! X: {sample['X'].shape}, GRF: {sample['y_grf'].shape}")