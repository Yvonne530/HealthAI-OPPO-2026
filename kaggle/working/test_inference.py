"""
Qwen2-VL 微调后模型推理测试脚本
用于验证训练效果
"""
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

def load_model_and_processor(model_path, adapter_path=None):
    """
    加载模型和处理器
    
    Args:
        model_path: 基座模型路径
        adapter_path: LoRA 适配器路径（可选）
    """
    print(f"加载模型: {model_path}")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 如果提供了 LoRA 适配器，加载它
    if adapter_path and os.path.exists(adapter_path):
        print(f"加载 LoRA 适配器: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # 合并权重
    
    model.eval()
    print("✓ 模型加载完成")
    
    return model, processor

def inference(model, processor, image_path, prompt, max_new_tokens=512):
    """
    执行推理
    
    Args:
        model: 模型
        processor: 处理器
        image_path: 图片路径
        prompt: 文本提示
        max_new_tokens: 最大生成 token 数
    """
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 构建对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    # 移动到 GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0
        )
    
    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def test_model():
    """测试模型"""
    print("\n" + "="*60)
    print("  Qwen2-VL 推理测试")
    print("="*60 + "\n")
    
    # 配置路径
    base_model = "Qwen/Qwen2-VL-7B-Instruct"
    adapter_path = "/kaggle/working/rehab_lora"
    
    # 测试图片路径（请根据实际修改）
    test_image = "/kaggle/input/fitness-rehab-vlm/images/squat/squat__side__standard__none__0016.jpg"
    test_prompt = "请检查我的深蹲动作。"
    
    # 检查文件是否存在
    if not os.path.exists(test_image):
        print(f"✗ 测试图片不存在: {test_image}")
        print("请修改 test_image 变量指向有效的图片路径")
        return
    
    # 加载模型
    try:
        model, processor = load_model_and_processor(base_model, adapter_path)
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 执行推理
    print(f"\n测试输入:")
    print(f"  图片: {test_image}")
    print(f"  提示: {test_prompt}")
    print(f"\n生成中...")
    print("-" * 60)
    
    try:
        output = inference(model, processor, test_image, test_prompt)
        print(f"\n模型输出:\n{output}")
        print("-" * 60)
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        return
    
    print("\n✓ 推理测试完成")
    print("="*60 + "\n")

def batch_test():
    """批量测试多个样本"""
    print("\n" + "="*60)
    print("  批量推理测试")
    print("="*60 + "\n")
    
    base_model = "Qwen/Qwen2-VL-7B-Instruct"
    adapter_path = "/kaggle/working/rehab_lora"
    
    # 定义测试样本
    test_samples = [
        {
            "image": "/kaggle/input/fitness-rehab-vlm/images/squat/squat__side__standard__none__0016.jpg",
            "prompt": "请检查我的深蹲动作。"
        },
        {
            "image": "/kaggle/input/fitness-rehab-vlm/images/lunge/lunge__side__standard__none__0001.jpg",
            "prompt": "请检查我的弓步蹲动作。"
        }
    ]
    
    # 加载模型
    try:
        model, processor = load_model_and_processor(base_model, adapter_path)
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 批量推理
    for i, sample in enumerate(test_samples, 1):
        print(f"\n[样本 {i}/{len(test_samples)}]")
        print(f"图片: {sample['image']}")
        print(f"提示: {sample['prompt']}")
        
        if not os.path.exists(sample['image']):
            print(f"✗ 图片不存在，跳过")
            continue
        
        try:
            output = inference(model, processor, sample['image'], sample['prompt'])
            print(f"输出:\n{output}")
            print("-" * 60)
        except Exception as e:
            print(f"✗ 推理失败: {e}")
    
    print("\n✓ 批量测试完成")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_test()
    else:
        test_model()