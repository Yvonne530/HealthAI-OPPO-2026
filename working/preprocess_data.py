"""
鲁棒的数据预处理脚本 - Qwen2-VL 医疗助手数据集
严格校验 + 路径修正 + 格式标准化
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any

# ========== 配置区 ==========
INPUT_DIR = "/kaggle/input/fitness-rehab-vlm"
OUTPUT_DIR = "/kaggle/temp"
TRAIN_FILE = "train.json"
VAL_FILE = "val.json"

def validate_data_sample(sample: Dict[str, Any], base_path: str) -> tuple[bool, str]:
    """
    硬校验单个数据样本
    
    Returns:
        (is_valid, error_message)
    """
    # 1. 检查必需字段
    if "conversations" not in sample:
        return False, "缺少 conversations 字段"
    
    if "images" not in sample:
        return False, "缺少 images 字段"
    
    # 2. 校验 images 必须是列表
    if not isinstance(sample["images"], list):
        return False, f"images 字段必须是列表，当前类型: {type(sample['images'])}"
    
    if len(sample["images"]) == 0:
        return False, "images 列表不能为空"
    
    # 3. 校验 conversations 格式
    conversations = sample["conversations"]
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False, "conversations 必须是非空列表"
    
    # 4. 校验角色交替 (human -> gpt -> human -> gpt ...)
    expected_roles = ["human", "gpt"] * (len(conversations) // 2 + 1)
    for i, conv in enumerate(conversations):
        if "from" not in conv or "value" not in conv:
            return False, f"conversations[{i}] 缺少 from 或 value 字段"
        
        if conv["from"] not in ["human", "gpt"]:
            return False, f"非法角色: {conv['from']}"
        
        if i < len(expected_roles) and conv["from"] != expected_roles[i]:
            return False, f"角色顺序错误，位置 {i} 应为 {expected_roles[i]}，实际为 {conv['from']}"
    
    # 5. 物理路径校验 - 检查图片文件是否存在
    for img_path in sample["images"]:
        # 移除可能的前缀路径，只保留相对路径
        relative_path = img_path.replace(INPUT_DIR + "/", "").replace("\\", "/")
        full_path = os.path.join(base_path, relative_path)
        
        if not os.path.exists(full_path):
            return False, f"图片文件不存在: {full_path}"
    
    return True, ""


def fix_image_paths(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    修正图片路径：
    1. 替换所有反斜杠为正斜杠
    2. 添加 Kaggle 路径前缀
    3. 确保格式为列表
    """
    fixed_sample = sample.copy()
    
    # 处理 images 字段
    if "images" in fixed_sample:
        images = fixed_sample["images"]
        
        # 如果是字符串，转换为列表
        if isinstance(images, str):
            images = [images]
        
        # 修正路径
        fixed_images = []
        for img in images:
            # 替换反斜杠
            img = img.replace("\\", "/")
            
            # 如果没有前缀，添加 Kaggle 路径
            if not img.startswith("/kaggle/input/"):
                # 移除可能已存在的 "images/" 前缀，避免重复
                if img.startswith("images/"):
                    img = img
                fixed_images.append(f"{INPUT_DIR}/{img}")
            else:
                fixed_images.append(img)
        
        fixed_sample["images"] = fixed_images
    
    # 处理旧格式的 image 字段（单数形式）
    if "image" in fixed_sample and "images" not in fixed_sample:
        img = fixed_sample["image"].replace("\\", "/")
        if not img.startswith("/kaggle/input/"):
            img = f"{INPUT_DIR}/{img}"
        fixed_sample["images"] = [img]
        del fixed_sample["image"]  # 删除旧字段
    
    return fixed_sample


def process_dataset(input_file: str, output_file: str):
    """处理单个数据集文件"""
    print(f"\n{'='*60}")
    print(f"处理文件: {input_file}")
    print(f"{'='*60}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 加载了 {len(data)} 个样本")
    
    # 处理每个样本
    processed_data = []
    valid_count = 0
    error_count = 0
    
    for idx, sample in enumerate(data):
        # 1. 修正路径
        fixed_sample = fix_image_paths(sample)
        
        # 2. 校验数据
        is_valid, error_msg = validate_data_sample(fixed_sample, INPUT_DIR)
        
        if is_valid:
            processed_data.append(fixed_sample)
            valid_count += 1
        else:
            error_count += 1
            print(f"✗ 样本 {idx} 校验失败: {error_msg}")
            if error_count <= 5:  # 只显示前5个错误
                print(f"  样本ID: {sample.get('id', 'unknown')}")
    
    # 保存处理后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 统计报告
    print(f"\n{'='*60}")
    print(f"处理完成统计:")
    print(f"  ✓ 有效样本: {valid_count}")
    print(f"  ✗ 无效样本: {error_count}")
    print(f"  → 保存路径: {output_file}")
    print(f"{'='*60}\n")
    
    # 展示第一个样本作为示例
    if processed_data:
        print("✓ 第一个样本预览:")
        print(json.dumps(processed_data[0], ensure_ascii=False, indent=2)[:500] + "...")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Qwen2-VL 数据预处理启动")
    print("="*60)
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理训练集和验证集
    for filename in [TRAIN_FILE, VAL_FILE]:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(input_path):
            process_dataset(input_path, output_path)
        else:
            print(f"⚠ 警告: 文件不存在 {input_path}")
    
    print("\n" + "="*60)
    print("✓ 数据预处理全部完成！")
    print(f"✓ 输出目录: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()