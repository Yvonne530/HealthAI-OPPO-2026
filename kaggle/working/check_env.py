"""
Kaggle 环境检查脚本
验证 GPU、磁盘空间、依赖包等
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_gpu():
    """检查 GPU 信息"""
    print_section("GPU 信息")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        print(result.stdout.strip())
        
        # 检查是否是 A100
        if "A100" in result.stdout:
            print("✓ 检测到 A100 GPU")
        else:
            print("⚠ 警告: 不是 A100 GPU，配置可能需要调整")
    except Exception as e:
        print(f"✗ GPU 检查失败: {e}")

def check_disk_space():
    """检查磁盘空间"""
    print_section("磁盘空间")
    
    paths = ["/kaggle/working", "/kaggle/temp", "/kaggle/input"]
    
    for path in paths:
        if os.path.exists(path):
            stat = os.statvfs(path)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            used_percent = ((total_gb - free_gb) / total_gb) * 100
            
            print(f"\n{path}:")
            print(f"  总空间: {total_gb:.2f} GB")
            print(f"  可用空间: {free_gb:.2f} GB")
            print(f"  使用率: {used_percent:.1f}%")
            
            # 警告检查
            if path == "/kaggle/working" and free_gb < 15:
                print(f"  ⚠ 警告: 可用空间不足 15GB")
            if path == "/kaggle/temp" and free_gb < 30:
                print(f"  ⚠ 警告: 临时空间不足 30GB")

def check_python_packages():
    """检查关键 Python 包"""
    print_section("Python 包检查")
    
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT",
        "datasets": "Datasets",
        "accelerate": "Accelerate",
        "PIL": "Pillow"
    }
    
    for package, name in packages.items():
        try:
            if package == "PIL":
                import PIL
                version = PIL.__version__
            else:
                mod = __import__(package)
                version = mod.__version__
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: 未安装")

def check_cuda():
    """检查 CUDA 和 PyTorch"""
    print_section("CUDA 和 PyTorch")
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name(0)}")
            
            # 显存信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU 显存: {total_memory:.2f} GB")
    except Exception as e:
        print(f"✗ CUDA 检查失败: {e}")

def check_data_files():
    """检查数据文件"""
    print_section("数据文件检查")
    
    input_dir = "/kaggle/input/fitness-rehab-vlm"
    
    if not os.path.exists(input_dir):
        print(f"✗ 数据集目录不存在: {input_dir}")
        print("  请确保已添加 fitness-rehab-vlm 数据集")
        return
    
    print(f"✓ 数据集目录存在: {input_dir}")
    
    # 检查必需文件
    required_files = ["train.json", "val.json"]
    for filename in required_files:
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            
            # 读取样本数
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    count = len(data)
                print(f"✓ {filename}: {size_mb:.2f} MB, {count} 个样本")
            except:
                print(f"✓ {filename}: {size_mb:.2f} MB")
        else:
            print(f"✗ {filename}: 不存在")
    
    # 检查图片目录
    images_dir = os.path.join(input_dir, "images")
    if os.path.exists(images_dir):
        subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        print(f"✓ images/ 目录: {len(subdirs)} 个子目录 ({', '.join(subdirs)})")
        
        # 统计图片数量
        total_images = 0
        for subdir in subdirs:
            subdir_path = os.path.join(images_dir, subdir)
            images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            total_images += len(images)
        print(f"  总图片数: {total_images}")
    else:
        print(f"✗ images/ 目录不存在")

def check_llamafactory():
    """检查 LLaMA-Factory 安装"""
    print_section("LLaMA-Factory 检查")
    
    llamafactory_dir = "/kaggle/working/LLaMA-Factory"
    
    if os.path.exists(llamafactory_dir):
        print(f"✓ LLaMA-Factory 目录存在")
        
        # 检查关键文件
        key_files = [
            "src/llamafactory/__init__.py",
            "data/dataset_info.json"
        ]
        
        for filename in key_files:
            filepath = os.path.join(llamafactory_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename}")
    else:
        print(f"✗ LLaMA-Factory 未安装")
        print("  请先运行: bash setup.sh")
    
    # 检查命令行工具
    try:
        result = subprocess.run(
            ['llamafactory-cli', '--version'],
            capture_output=True, text=True, check=True
        )
        print(f"✓ llamafactory-cli 可用")
    except:
        print(f"✗ llamafactory-cli 不可用")

def check_config_files():
    """检查配置文件"""
    print_section("配置文件检查")
    
    config_files = [
        "/kaggle/working/train_config.yaml",
        "/kaggle/working/export_config.yaml",
        "/kaggle/working/preprocess_data.py"
    ]
    
    for filepath in config_files:
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"✓ {os.path.basename(filepath)}: {size_kb:.2f} KB")
        else:
            print(f"✗ {os.path.basename(filepath)}: 不存在")

def main():
    print("\n" + "="*60)
    print("  Kaggle Qwen2-VL 微调环境检查")
    print("="*60)
    
    check_gpu()
    check_cuda()
    check_disk_space()
    check_python_packages()
    check_data_files()
    check_llamafactory()
    check_config_files()
    
    print("\n" + "="*60)
    print("  环境检查完成")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()