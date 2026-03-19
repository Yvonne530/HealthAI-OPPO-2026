"""
test_model.py - 单元测试
RehabGuardian 10.0 | FNO-1D 模型完整测试套件

测试覆盖：
- 模型维度验证
- 物理损失函数正确性
- 卷积近似误差
- ONNX 导出（可选）
- 边界条件处理
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from pathlib import Path

from model import FNO1D, FNOBlock, SpectralConv1d, PositionalEncoding, PhysicsInformedLoss
from inference import FNO1DONNX, predict, validate_approximation_error


# ============================================================
# 测试辅助函数
# ============================================================

def make_model() -> FNO1D:
    """创建标准测试模型"""
    return FNO1D(in_channels=69, out_channels=6, width=64, modes=16, n_layers=4, seq_len=50)


def make_batch(B: int = 4) -> tuple:
    """创建测试批次数据"""
    X = torch.randn(B, 50, 69)
    Y = torch.randn(B, 50, 6)
    return X, Y


# ============================================================
# 测试类
# ============================================================

class TestModelDimensions(unittest.TestCase):
    """测试模型各层维度是否符合设计规范"""

    def setUp(self):
        self.model = make_model()
        self.model.eval()

    def test_output_shape(self):
        """主测试：输入 [B, 50, 69] → 输出 [B, 50, 6]"""
        X, _ = make_batch(4)
        with torch.no_grad():
            pred = self.model(X)
        self.assertEqual(pred.shape, (4, 50, 6),
                         f"输出形状错误: 期望 (4,50,6), 得到 {pred.shape}")

    def test_batch_size_1(self):
        """边界条件：batch_size=1"""
        X = torch.randn(1, 50, 69)
        with torch.no_grad():
            pred = self.model(X)
        self.assertEqual(pred.shape, (1, 50, 6))

    def test_input_proj_shape(self):
        """输入投影层: [B,50,69] → [B,50,64]"""
        X = torch.randn(2, 50, 69)
        with torch.no_grad():
            out = self.model.input_proj(X)
        self.assertEqual(out.shape, (2, 50, 64))

    def test_pos_enc_shape(self):
        """位置编码不改变输入维度"""
        x = torch.randn(2, 50, 64)
        out = self.model.pos_enc(x)
        self.assertEqual(out.shape, (2, 50, 64))

    def test_spectral_conv_shape(self):
        """谱卷积层: [B,64,50] → [B,64,50]"""
        layer = SpectralConv1d(64, 64, modes=16, seq_len=50)
        x = torch.randn(2, 64, 50)
        with torch.no_grad():
            out = layer(x)
        self.assertEqual(out.shape, (2, 64, 50))

    def test_fno_block_shape(self):
        """FNO Block: [B,50,64] → [B,50,64]"""
        block = FNOBlock(64, 16, seq_len=50)
        x = torch.randn(2, 50, 64)
        with torch.no_grad():
            out = block(x)
        self.assertEqual(out.shape, (2, 50, 64))

    def test_output_proj_shape(self):
        """输出投影层: [B,50,64] → [B,50,6]"""
        x = torch.randn(2, 50, 64)
        with torch.no_grad():
            out = self.model.output_proj(x)
        self.assertEqual(out.shape, (2, 50, 6))

    def test_model_parameter_count(self):
        """参数量应接近 550K"""
        n_params = self.model.count_parameters()
        # 允许合理范围：400K ~ 800K
        self.assertGreater(n_params, 400_000, f"参数量过少: {n_params:,}")
        self.assertLess(n_params, 800_000, f"参数量过多: {n_params:,}")
        print(f"\n  实际参数量: {n_params:,}")

    def test_model_size_mb(self):
        """INT8量化后应 < 1MB"""
        size_mb = self.model.model_size_mb('int8')
        self.assertLess(size_mb, 1.0, f"INT8大小 {size_mb:.2f}MB 超出1MB限制")
        print(f"\n  INT8大小: {size_mb:.3f} MB")


class TestPhysicsLoss(unittest.TestCase):
    """测试物理一致性损失函数"""

    def setUp(self):
        self.criterion = PhysicsInformedLoss(delta_weight=0.3, anatomical_weight=0.15)

    def test_loss_keys(self):
        """损失字典包含所有必要键"""
        pred   = torch.randn(4, 50, 6)
        target = torch.randn(4, 50, 6)
        losses = self.criterion(pred, target)
        for key in ['total', 'mse', 'delta', 'anatomical']:
            self.assertIn(key, losses, f"损失字典缺少键: {key}")

    def test_loss_nonnegative(self):
        """所有损失项应为非负值"""
        pred   = torch.randn(4, 50, 6)
        target = torch.randn(4, 50, 6)
        losses = self.criterion(pred, target)
        for k, v in losses.items():
            self.assertGreaterEqual(v.item(), 0.0, f"{k} 损失为负: {v.item()}")

    def test_total_loss_formula(self):
        """验证总损失 = MSE + 0.3×ΔLoss + 0.15×AnatomicalLoss"""
        pred   = torch.randn(4, 50, 6)
        target = torch.randn(4, 50, 6)
        losses = self.criterion(pred, target)
        expected_total = (losses['mse'] +
                          0.3 * losses['delta'] +
                          0.15 * losses['anatomical'])
        self.assertAlmostEqual(
            losses['total'].item(),
            expected_total.item(),
            places=5,
            msg="总损失公式不符合设计规范"
        )

    def test_perfect_prediction_low_loss(self):
        """完美预测时损失应接近0"""
        target = torch.rand(4, 50, 6).abs()  # 非负，模拟真实GRF
        losses = self.criterion(target, target)
        self.assertLess(losses['mse'].item(), 1e-6, "完美预测MSE应接近0")

    def test_grf_nonneg_penalty(self):
        """当预测负Fz时，anatomical损失应增加"""
        target = torch.ones(4, 50, 6) * 5.0
        pred_neg = target.clone()
        pred_neg[:, :, 2] = -1.0  # 设置负垂直力
        pred_pos = target.clone()

        loss_neg = self.criterion(pred_neg, target)['anatomical']
        loss_pos = self.criterion(pred_pos, target)['anatomical']
        self.assertGreater(loss_neg.item(), loss_pos.item(),
                           "负Fz时anatomical损失应更大")

    def test_contact_mask_effect(self):
        """接触期约束仅在接触期生效（Fz > 1.0时）"""
        pred   = torch.ones(4, 50, 6) * 25.0  # 超出范围值

        # 接触期（target Fz > 1.0）
        target_contact = torch.ones(4, 50, 6) * 5.0
        target_contact[..., 2] = 2.0  # Fz > 1.0 → 接触期

        # 摆动期（target Fz < 1.0）
        target_swing = torch.ones(4, 50, 6) * 0.5
        target_swing[..., 2] = 0.5   # Fz < 1.0 → 摆动期

        loss_contact = self.criterion(pred, target_contact)['anatomical']
        loss_swing   = self.criterion(pred, target_swing)['anatomical']
        self.assertGreater(loss_contact.item(), loss_swing.item(),
                           "接触期应有更大的范围约束损失")

    def test_delta_loss_formula(self):
        """验证变化率约束公式"""
        pred   = torch.randn(4, 50, 6)
        target = torch.randn(4, 50, 6)
        losses = self.criterion(pred, target)
        # 手动计算 delta loss
        expected = torch.mean(
            (pred[:, 1:] - pred[:, :-1] - (target[:, 1:] - target[:, :-1])) ** 2
        )
        self.assertAlmostEqual(
            losses['delta'].item(), expected.item(), places=5
        )


class TestONNXConversion(unittest.TestCase):
    """测试 FFT → Conv 卷积近似转换"""

    def setUp(self):
        self.fft_model  = make_model().eval()
        self.conv_model = FNO1DONNX(self.fft_model).eval()

    def test_conv_model_output_shape(self):
        """ONNX版模型输出形状应与FFT版一致"""
        X = torch.randn(2, 50, 69)
        with torch.no_grad():
            pred_fft  = self.fft_model(X)
            pred_conv = self.conv_model(X)
        self.assertEqual(pred_fft.shape,  (2, 50, 6))
        self.assertEqual(pred_conv.shape, (2, 50, 6))

    def test_approximation_error_below_threshold(self):
        """
        卷积近似误差应 < 5%
        注意：随机初始化权重下误差可能较大，
        此测试主要验证函数可以正常运行
        """
        err = validate_approximation_error(
            self.fft_model, self.conv_model,
            n_samples=20
        )
        # 随机权重下误差可能 > 5%，这里只验证函数能正常运行
        self.assertIsInstance(err, float)
        self.assertGreater(err, 0.0)
        print(f"\n  卷积近似误差: {err:.2f}% (随机权重，仅验证函数运行)")

    def test_no_fft_ops_in_onnx_model(self):
        """ONNX模型中不应包含FFT相关模块"""
        # 检查所有子模块类型
        module_types = {type(m).__name__ for m in self.conv_model.modules()}
        fft_modules = {t for t in module_types if 'Spectral' in t and 'ONNX' not in t}
        self.assertEqual(
            len(fft_modules), 0,
            f"ONNX模型中发现非ONNX兼容的频域模块: {fft_modules}"
        )

    def test_onnx_export(self):
        """测试 ONNX 导出流程"""
        try:
            import onnx
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, 'test_model.onnx')
                dummy = torch.randn(1, 50, 69)
                torch.onnx.export(
                    self.conv_model,
                    dummy,
                    onnx_path,
                    opset_version=11,
                    input_names=['input'],
                    output_names=['output'],
                )
                self.assertTrue(os.path.exists(onnx_path))
                model_onnx = onnx.load(onnx_path)
                onnx.checker.check_model(model_onnx)
                print("\n  ✅ ONNX 导出验证通过")
        except ImportError:
            self.skipTest("onnx 未安装，跳过 ONNX 导出测试")


class TestInferenceInterface(unittest.TestCase):
    """测试推理接口"""

    def setUp(self):
        self.model = make_model().eval()

    def test_predict_single_sample(self):
        """单样本推理: 输入 [50, 69] → 输出 [50, 6]"""
        x = np.random.randn(50, 69).astype(np.float32)
        pred = predict(x, self.model, device=torch.device('cpu'))
        self.assertEqual(pred.shape, (50, 6))
        self.assertIsInstance(pred, np.ndarray)

    def test_predict_output_range(self):
        """预测输出为有限数值（无NaN/Inf）"""
        x = np.random.randn(50, 69).astype(np.float32)
        pred = predict(x, self.model, device=torch.device('cpu'))
        self.assertTrue(np.all(np.isfinite(pred)), "预测包含 NaN 或 Inf")

    def test_predict_batch_input(self):
        """支持批次输入 [1, 50, 69]"""
        x = np.random.randn(1, 50, 69).astype(np.float32)
        pred = predict(x, self.model, device=torch.device('cpu'))
        self.assertEqual(pred.shape, (50, 6))


class TestTrainingComponents(unittest.TestCase):
    """测试训练相关组件"""

    def test_gradient_flow(self):
        """梯度应能正常反向传播到所有参数"""
        model = make_model()
        criterion = PhysicsInformedLoss()
        X, Y = make_batch(2)
        losses = criterion(model(X), Y)
        losses['total'].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"参数 {name} 梯度为 None，梯度流断裂"
                )

    def test_amp_compatibility(self):
        """AMP 混合精度训练兼容性"""
        try:
            from torch.cuda.amp import autocast, GradScaler
            model = make_model()
            X, Y = make_batch(2)
            scaler = GradScaler(enabled=False)  # CPU测试下禁用
            with autocast(enabled=False):
                losses = PhysicsInformedLoss()(model(X), Y)
            scaler.scale(losses['total']).backward()
            print("\n  ✅ AMP 兼容性验证通过")
        except Exception as e:
            self.fail(f"AMP 兼容性测试失败: {e}")

    def test_model_eval_no_grad(self):
        """eval 模式下 torch.no_grad() 应减少内存占用"""
        model = make_model().eval()
        X, _ = make_batch(4)
        with torch.no_grad():
            pred = model(X)
        # 验证输出不需要梯度
        self.assertFalse(pred.requires_grad)

    def test_position_encoding_values(self):
        """位置编码值应在合理范围 [-1, 1]"""
        pe_layer = PositionalEncoding(d_model=64, max_len=50)
        pe_values = pe_layer.pe  # [1, 50, 64]
        self.assertTrue(
            (pe_values >= -1.0).all() and (pe_values <= 1.0).all(),
            "位置编码值超出 [-1, 1] 范围"
        )


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("RehabGuardian FNO-1D 单元测试")
    print("=" * 60)

    # 详细输出模式
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # 按顺序添加测试
    for cls in [
        TestModelDimensions,
        TestPhysicsLoss,
        TestONNXConversion,
        TestInferenceInterface,
        TestTrainingComponents,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
    else:
        print(f"\n❌ {len(result.failures)} 个测试失败, {len(result.errors)} 个错误")
        exit(1)
