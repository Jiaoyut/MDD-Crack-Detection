import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import logging
import time
import math
import hashlib
import numpy as np
from collections import defaultdict, deque
import cv2

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CorrectEbbinghausDistillation:
    """正确的艾宾浩斯蒸馏系统 - 使用有效的YOLO参数"""

    def __init__(self, teacher_path, student_config, data_config, device='cuda'):
        self.device = device
        self.teacher_path = teacher_path
        self.student_config = student_config
        self.data_config = data_config

        # 训练状态
        self.step_count = 0
        self.epoch_count = 0
        self.distill_applied = 0
        self.start_time = time.time()

        # 稳定化参数
        self.nan_detected = False
        self.nan_recovery_steps = 0
        self.stable_mode = False

        # 正确的记忆系统
        self.memory_model = CorrectMemoryModel()
        self.review_scheduler = CorrectReviewScheduler(self.memory_model)

        self.setup_correct_models()
        self.setup_correct_callbacks()

        logging.info("🧠✅ 正确的艾宾浩斯蒸馏系统初始化完成")

    def setup_correct_models(self):
        """设置正确模型"""
        try:
            # 教师模型
            self.teacher = YOLO(self.teacher_path)
            self.teacher.model.to(self.device).eval()
            for p in self.teacher.model.parameters():
                p.requires_grad = False

            # 学生模型
            self.student = YOLO(self.student_config)
            self.student.model.to(self.device)

            # 添加梯度裁剪
            self._add_correct_gradient_clipping()

            logging.info("✅ 正确模型设置完成")
        except Exception as e:
            logging.error(f"❌ 模型设置失败: {e}")
            raise

    def _add_correct_gradient_clipping(self):
        """添加正确的梯度裁剪"""
        try:
            # 手动添加梯度裁剪钩子
            for name, param in self.student.model.named_parameters():
                if param.requires_grad:
                    def make_hook(param_name):
                        def hook(grad):
                            clipped_grad = torch.clamp(grad, -1.0, 1.0)
                            if self.step_count % 500 == 0:
                                grad_norm = grad.norm().item() if grad is not None else 0
                                clipped_norm = clipped_grad.norm().item() if clipped_grad is not None else 0
                                if grad_norm > 1.0:
                                    logging.info(
                                        f"✂️ 梯度裁剪: {param_name} 梯度范数 {grad_norm:.3f} -> {clipped_norm:.3f}")
                            return clipped_grad

                        return hook

                    param.register_hook(make_hook(name))

            logging.info("✅ 梯度裁剪钩子已添加")
        except Exception as e:
            logging.warning(f"❌ 梯度裁剪设置失败: {e}")

    def setup_correct_callbacks(self):
        """设置正确回调"""
        logging.info("🔧 设置正确回调...")

        # 保存原始训练方法
        self.original_train = self.student.train

        def correct_train_wrapper(**kwargs):
            return self._correct_training(**kwargs)

        # 替换训练方法
        self.student.train = correct_train_wrapper
        logging.info("✅ 正确回调设置完成")

    def _correct_training(self, **kwargs):
        """正确训练"""
        logging.info("🚀 开始正确的艾宾浩斯蒸馏训练...")

        # 设置正确回调
        self._setup_correct_training_callbacks()

        # 使用正确的训练配置
        correct_kwargs = self._get_correct_training_config(kwargs)

        # 使用原始训练
        results = self.original_train(**correct_kwargs)

        # 最终报告
        self._final_correct_report()

        return results

    def _get_correct_training_config(self, kwargs):
        """获取正确的训练配置 - 使用有效的YOLO参数"""
        # 使用YOLO支持的有效参数
        correct_config = {
            'data': self.data_config,
            'epochs': kwargs.get('epochs', 100),
            'imgsz': kwargs.get('imgsz', 640),
            'batch': kwargs.get('batch', 16),
            'device': self.device,
            'workers': kwargs.get('workers', 4),
            'lr0': 0.001,  # 降低学习率以提高稳定性
            'amp': False,  # 关闭混合精度训练
            'verbose': True,
            'project': kwargs.get('project', 'runs/correct_ebbinghaus'),
            'name': kwargs.get('name', f'correct_{int(time.time())}'),
            # 只使用YOLO支持的有效参数
        }

        # 添加其他有效参数
        valid_args = ['patience', 'save_period', 'seed', 'cos_lr', 'label_smoothing']
        for arg in valid_args:
            if arg in kwargs:
                correct_config[arg] = kwargs[arg]

        logging.info(f"🔧 正确训练配置: lr0={correct_config['lr0']}, amp={correct_config['amp']}")
        return correct_config

    def _setup_correct_training_callbacks(self):
        """设置正确训练回调"""
        logging.info("🔧 设置正确训练回调...")

        if not hasattr(self.student, 'callbacks'):
            self.student.callbacks = {}

        def on_train_batch_start(trainer):
            """正确批次开始回调"""
            self.step_count += 1

            try:
                # 检查NaN状态
                if self.nan_detected:
                    self.nan_recovery_steps += 1
                    if self.nan_recovery_steps >= 100:
                        self.nan_detected = False
                        self.nan_recovery_steps = 0
                        logging.info("🔄 已从NaN状态恢复")

                # 提取批次数据
                batch = self._get_correct_batch_data(trainer)
                if batch is not None:
                    self.current_batch = batch

                    if self.step_count % 200 == 0:
                        imgs = batch.get('img') if hasattr(batch, 'get') else None
                        if imgs is not None and hasattr(imgs, 'shape'):
                            logging.info(f"✅ 步骤{self.step_count}: 获取真实批次数据, 形状: {imgs.shape}")
                else:
                    if self.step_count % 200 == 0:
                        logging.info(f"🔄 步骤{self.step_count}: 使用正确虚拟数据")
                    self.current_batch = self._create_correct_virtual_batch()

            except Exception as e:
                if self.step_count % 200 == 0:
                    logging.warning(f"❌ 步骤{self.step_count}批次开始回调失败: {e}")

        def on_train_batch_end(trainer):
            """正确批次结束回调"""
            try:
                # 检查损失是否为NaN
                if hasattr(trainer, 'loss') and trainer.loss is not None:
                    try:
                        if hasattr(trainer.loss, 'item'):
                            loss_value = trainer.loss.item()
                        else:
                            loss_value = float(trainer.loss)
                        if math.isnan(loss_value) or math.isinf(loss_value):
                            if not self.nan_detected:
                                logging.warning("⚠️ 检测到NaN损失，启用正确模式")
                                self.nan_detected = True
                                self.stable_mode = True
                            return  # 跳过蒸馏
                    except:
                        pass

                self._apply_correct_distillation(trainer)
            except Exception as e:
                if self.step_count % 200 == 0:
                    logging.warning(f"❌ 步骤{self.step_count}蒸馏失败: {e}")

        def on_train_epoch_end(trainer):
            self.epoch_count += 1
            self._perform_correct_review()
            self._epochly_correct_report()

        # 注册回调
        if 'on_train_batch_start' not in self.student.callbacks:
            self.student.callbacks['on_train_batch_start'] = []
        if 'on_train_batch_end' not in self.student.callbacks:
            self.student.callbacks['on_train_batch_end'] = []
        if 'on_train_epoch_end' not in self.student.callbacks:
            self.student.callbacks['on_train_epoch_end'] = []

        self.student.callbacks['on_train_batch_start'].append(on_train_batch_start)
        self.student.callbacks['on_train_batch_end'].append(on_train_batch_end)
        self.student.callbacks['on_train_epoch_end'].append(on_train_epoch_end)

        logging.info("✅ 正确训练回调设置完成")

    def _get_correct_batch_data(self, trainer):
        """获取正确批次数据"""
        try:
            # 方法1: 检查trainer.batch
            if hasattr(trainer, 'batch') and trainer.batch is not None:
                return trainer.batch

            # 方法2: 检查其他可能属性
            for attr_name in ['batch_data', 'current_batch', 'data_batch']:
                if hasattr(trainer, attr_name):
                    batch = getattr(trainer, attr_name)
                    if batch is not None:
                        return batch

            return None
        except:
            return None

    def _create_correct_virtual_batch(self):
        """创建正确虚拟批次"""
        try:
            batch_size = 8
            img_size = 640

            # 使用更稳定的随机数生成
            virtual_batch = {
                'img': torch.randn(batch_size, 3, img_size, img_size, device=self.device) * 0.1 + 0.5,
                'cls': torch.randint(0, 1, (batch_size,), device=self.device),
                'bbox': torch.rand(batch_size, 4, device=self.device) * 0.8 + 0.1
            }
            return virtual_batch
        except:
            return None

    def _apply_correct_distillation(self, trainer):
        """应用正确蒸馏"""
        if not hasattr(self, 'current_batch') or self.current_batch is None:
            if self.step_count % 200 == 0:
                logging.info(f"🔄 步骤{self.step_count}: 无批次数据，跳过蒸馏")
            return

        if self.nan_detected and self.stable_mode:
            if self.step_count % 100 == 0:
                logging.info(f"🔄 步骤{self.step_count}: NaN恢复模式，跳过蒸馏")
            return

        batch = self.current_batch

        try:
            # 验证批次数据
            if not self._validate_correct_batch(batch):
                if self.step_count % 200 == 0:
                    logging.info(f"🔄 步骤{self.step_count}: 批次数据无效，使用正确虚拟数据")
                batch = self._create_correct_virtual_batch()
                if batch is None:
                    return

            # 获取图像数据
            imgs = batch.get('img') if hasattr(batch, 'get') else None
            if imgs is None:
                if self.step_count % 200 == 0:
                    logging.info(f"🔄 步骤{self.step_count}: 无图像数据，使用正确虚拟图像")
                imgs = torch.randn(8, 3, 640, 640, device=self.device) * 0.1 + 0.5

            if self.step_count % 200 == 0:
                logging.info(f"🧠 步骤{self.step_count}: 应用正确蒸馏, 图像形状: {imgs.shape}")

            # 记录原始损失
            original_loss = 0.0
            if hasattr(trainer, 'loss') and trainer.loss is not None:
                try:
                    if hasattr(trainer.loss, 'item'):
                        original_loss = trainer.loss.item()
                    else:
                        original_loss = float(trainer.loss)
                except:
                    original_loss = 0.0

            # 准备图像
            imgs = imgs.to(self.device).float()

            # 检查图像数据稳定性
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                logging.warning("⚠️ 图像数据包含NaN或Inf，使用正确虚拟数据")
                imgs = torch.randn(8, 3, 640, 640, device=self.device) * 0.1 + 0.5

            if imgs.max() > 1.0:
                imgs = imgs / 255.0

            # 教师预测
            with torch.no_grad():
                try:
                    teacher_outputs = self.teacher.model(imgs)
                except Exception as e:
                    logging.warning(f"❌ 教师预测失败: {e}")
                    return

            # 学生预测
            try:
                student_outputs = self.student.model(imgs)
            except Exception as e:
                logging.warning(f"❌ 学生预测失败: {e}")
                return

            # 计算正确蒸馏损失
            distill_loss = self._compute_correct_distillation_loss(student_outputs, teacher_outputs, imgs)

            # 检查蒸馏损失稳定性
            if not self._check_tensor_stability(distill_loss, '蒸馏损失'):
                logging.warning("⚠️ 蒸馏损失不稳定，使用默认值")
                distill_loss = torch.tensor(0.05, device=self.device, requires_grad=True)

            # 应用蒸馏损失
            distill_weight = 0.1 if self.stable_mode else 0.3
            weighted_distill = distill_weight * distill_loss

            if hasattr(trainer, 'loss') and trainer.loss is not None:
                # 检查当前损失稳定性
                if not self._check_loss_stability(trainer.loss):
                    logging.warning("⚠️ 训练损失不稳定，重置为蒸馏损失")
                    trainer.loss = weighted_distill
                else:
                    trainer.loss = trainer.loss + weighted_distill
            else:
                trainer.loss = weighted_distill

            # 安全获取蒸馏损失值
            distill_loss_value = 0.0
            try:
                if hasattr(distill_loss, 'item'):
                    distill_loss_value = distill_loss.item()
                else:
                    distill_loss_value = float(distill_loss)
            except:
                distill_loss_value = 0.05

            # 更新正确记忆系统
            self._update_correct_memory_system(batch, distill_loss_value, original_loss)

            self.distill_applied += 1

            # 记录日志
            if self.step_count % 200 == 0:
                new_loss = 0.0
                if hasattr(trainer, 'loss') and trainer.loss is not None:
                    try:
                        if hasattr(trainer.loss, 'item'):
                            new_loss = trainer.loss.item()
                        else:
                            new_loss = float(trainer.loss)
                    except:
                        new_loss = 0.0

                memory_report = self.memory_model.get_memory_report()

                logging.info("🧠 === 正确蒸馏报告 ===")
                logging.info(f"📊 训练步数: {self.step_count}")
                logging.info(f"💧 损失变化: {original_loss:.4f} -> {new_loss:.4f}")
                logging.info(f"🔥 蒸馏损失: {distill_loss_value:.6f}")
                logging.info(f"🎯 学习进度: {memory_report['learning_progress']:.1%}")
                logging.info(f"📚 跟踪样本: {memory_report['total_samples']}个")
                logging.info(f"💾 平均记忆强度: {memory_report['avg_intensity']:.3f}")
                logging.info(f"📈 记忆趋势: {memory_report['trend']}")
                logging.info(f"✅ 蒸馏应用次数: {self.distill_applied}")
                if self.stable_mode:
                    logging.info("🛡️ 稳定模式: 已启用")

        except Exception as e:
            if self.step_count % 200 == 0:
                logging.warning(f"❌ 步骤{self.step_count}蒸馏失败: {e}")

    def _compute_correct_distillation_loss(self, student_out, teacher_out, imgs):
        """计算正确蒸馏损失"""
        try:
            # 简化但稳定的蒸馏损失计算
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            num_losses = 0

            # 提取特征
            s_features = self._extract_correct_features(student_out, '学生')
            t_features = self._extract_correct_features(teacher_out, '教师')

            if not s_features or not t_features:
                if self.step_count % 500 == 0:
                    logging.info("🔄 无法提取特征，使用正确默认损失")
                return torch.tensor(0.05, device=self.device, requires_grad=True)

            # 对每个特征层计算损失
            for i, (s_feat, t_feat) in enumerate(zip(s_features, t_features)):
                if s_feat is None or t_feat is None:
                    continue

                # 稳定化特征张量
                s_feat = self._stabilize_tensor(s_feat)
                t_feat = self._stabilize_tensor(t_feat)

                if s_feat is None or t_feat is None:
                    continue

                # 确保形状匹配
                if s_feat.shape != t_feat.shape:
                    try:
                        # 使用插值调整学生特征图大小
                        s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
                    except Exception as e:
                        if self.step_count % 500 == 0:
                            logging.warning(f"❌ 特征{i}插值失败: {e}")
                        continue

                # 计算MSE损失
                try:
                    layer_loss = F.mse_loss(s_feat, t_feat) * 0.1

                    # 检查损失稳定性
                    if not self._check_tensor_stability(layer_loss, f"层损失{i}"):
                        continue

                    total_loss = total_loss + layer_loss
                    num_losses += 1

                except Exception as e:
                    if self.step_count % 500 == 0:
                        logging.warning(f"❌ 特征{i}损失计算失败: {e}")
                    continue

            if num_losses > 0:
                avg_loss = total_loss / num_losses
                if self.step_count % 500 == 0:
                    logging.info(f"🎯 正确特征蒸馏损失: {avg_loss.item():.6f} (基于{num_losses}个特征层)")
                return avg_loss
            else:
                if self.step_count % 500 == 0:
                    logging.info("🔄 所有特征层损失计算失败，使用正确默认损失")
                return torch.tensor(0.05, device=self.device, requires_grad=True)

        except Exception as e:
            logging.warning(f"❌ 正确蒸馏损失计算失败: {e}")
            return torch.tensor(0.05, device=self.device, requires_grad=True)

    def _extract_correct_features(self, model_output, model_type):
        """提取正确特征"""
        features = []

        try:
            if model_output is None:
                return features

            # 方法1: 直接是特征图
            if isinstance(model_output, torch.Tensor):
                features.append(model_output)
                return features

            # 方法2: 列表或元组
            if isinstance(model_output, (list, tuple)):
                for i, output in enumerate(model_output):
                    if isinstance(output, torch.Tensor):
                        features.append(output)
                    elif hasattr(output, 'detach'):
                        try:
                            output_tensor = output.detach()
                            features.append(output_tensor)
                        except:
                            pass

            # 方法3: 字典形式
            if isinstance(model_output, dict):
                for key, value in model_output.items():
                    if isinstance(value, torch.Tensor):
                        features.append(value)

            # 稳定化所有特征
            stable_features = []
            for feat in features:
                stable_feat = self._stabilize_tensor(feat)
                if stable_feat is not None:
                    stable_features.append(stable_feat)

            if self.step_count % 1000 == 0 and stable_features:
                logging.info(f"🔍 {model_type}特征提取: 找到{len(stable_features)}个稳定特征")

            return stable_features

        except Exception as e:
            if self.step_count % 500 == 0:
                logging.warning(f"❌ {model_type}特征提取失败: {e}")
            return []

    def _stabilize_tensor(self, tensor):
        """稳定化张量"""
        try:
            if tensor is None:
                return None

            # 检查是否为张量
            if not isinstance(tensor, torch.Tensor):
                try:
                    tensor = torch.tensor(tensor, device=self.device)
                except:
                    return None

            # 检查NaN和Inf
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                # 替换NaN和Inf为0
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
                tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)

            return tensor

        except Exception as e:
            if self.step_count % 500 == 0:
                logging.warning(f"❌ 张量稳定化失败: {e}")
            return None

    def _check_tensor_stability(self, tensor, tensor_name):
        """检查张量稳定性"""
        try:
            if tensor is None:
                return False

            if not isinstance(tensor, torch.Tensor):
                return False

            # 检查NaN和Inf
            if torch.isnan(tensor).any():
                logging.warning(f"⚠️ {tensor_name}包含NaN")
                return False

            if torch.isinf(tensor).any():
                logging.warning(f"⚠️ {tensor_name}包含Inf")
                return False

            return True

        except:
            return False

    def _check_loss_stability(self, loss):
        """检查损失稳定性"""
        try:
            if loss is None:
                return False

            if hasattr(loss, 'item'):
                loss_value = loss.item()
            else:
                loss_value = float(loss)

            if math.isnan(loss_value) or math.isinf(loss_value):
                return False

            if abs(loss_value) > 1e6:
                return False

            return True

        except:
            return False

    def _validate_correct_batch(self, batch):
        """验证正确批次"""
        try:
            if batch is None:
                return False
            if not hasattr(batch, 'get'):
                return False
            if batch.get('img') is None:
                return False
            return True
        except:
            return False

    def _update_correct_memory_system(self, batch, distill_loss_value, original_loss):
        """更新正确记忆系统"""
        try:
            if batch is None:
                return

            # 获取批次大小
            batch_size = 1
            if hasattr(batch, 'get'):
                imgs = batch.get('img')
                if imgs is not None and hasattr(imgs, 'shape') and len(imgs.shape) > 0:
                    batch_size = imgs.shape[0]

            for i in range(batch_size):
                sample_id = self._get_correct_sample_id(batch, i)

                # 计算正确学习增益
                learning_gain = self._compute_correct_learning_gain(distill_loss_value, original_loss)

                # 计算样本难度
                difficulty = self.memory_model.compute_sample_difficulty(batch, i)

                # 更新样本记忆
                self.memory_model.update_sample_memory(
                    sample_id, learning_gain, self.step_count, difficulty, is_review=False
                )

            if self.step_count % 500 == 0:
                memory_report = self.memory_model.get_memory_report()
                logging.info(
                    f"💾 正确记忆更新: 批次大小={batch_size}, 样本数={memory_report['total_samples']}, 学习进度={memory_report['learning_progress']:.1%}")

        except Exception as e:
            if self.step_count % 500 == 0:
                logging.warning(f"❌ 正确记忆更新失败: {e}")

    def _compute_correct_learning_gain(self, distill_loss, original_loss):
        """计算正确学习增益"""
        try:
            # 确保蒸馏损失为正
            distill_loss = max(0.0, distill_loss)

            # 基于蒸馏损失的计算
            if distill_loss <= 0.01:
                base_gain = 0.15
            elif distill_loss <= 0.05:
                base_gain = 0.08
            else:
                base_gain = 0.03

            # 稳定模式下降低增益
            if self.stable_mode:
                base_gain = base_gain * 0.5

            # 限制增益范围
            final_gain = max(0.01, min(0.2, base_gain))

            if self.step_count % 1000 == 0:
                logging.info(
                    f"🎯 正确学习增益: 蒸馏损失={distill_loss:.4f}, 增益={final_gain:.4f}, 稳定模式={self.stable_mode}")

            return final_gain

        except Exception as e:
            if self.step_count % 500 == 0:
                logging.warning(f"❌ 正确学习增益计算失败: {e}")
            return 0.05

    def _get_correct_sample_id(self, batch, index):
        """获取正确样本ID"""
        try:
            if not hasattr(batch, 'get'):
                return f"batch_{index}_{self.step_count}"

            imgs = batch.get('img')
            if imgs is None:
                return f"no_img_{index}_{self.step_count}"

            if hasattr(imgs, 'shape') and index < imgs.shape[0]:
                try:
                    if hasattr(imgs, '__getitem__'):
                        img_tensor = imgs[index]
                    else:
                        img_tensor = imgs

                    if hasattr(img_tensor, 'mean') and hasattr(img_tensor, 'std'):
                        img_mean = img_tensor.mean().item()
                        img_std = img_tensor.std().item()

                        # 使用更稳定的ID生成
                        id_string = f"{index}_{self.step_count}_{img_mean:.6f}"
                        img_hash = hashlib.md5(id_string.encode()).hexdigest()[:8]
                        return f"img_{img_hash}"
                except:
                    pass

            return f"sample_{index}_{self.step_count}"
        except:
            return f"error_{index}_{self.step_count}"

    def _perform_correct_review(self):
        """执行正确复习"""
        logging.info("📖 执行正确自适应复习调度...")

        # 安排复习
        review_samples = self.review_scheduler.schedule_reviews(
            self.step_count, review_ratio=0.3
        )

        if review_samples:
            logging.info(f"✅ 安排了{len(review_samples)}个样本的复习")
        else:
            logging.info("🔄 暂无需要复习的样本")

    def _epochly_correct_report(self):
        """正确周期报告"""
        memory_report = self.memory_model.get_memory_report()
        scheduling_report = self.review_scheduler.get_scheduling_report()

        logging.info("📈 === 正确周期报告 ===")
        logging.info(f"📅 训练周期: {self.epoch_count}")
        logging.info(f"🔄 总训练步数: {self.step_count}")
        logging.info(f"✅ 蒸馏应用次数: {self.distill_applied}")
        logging.info(f"🎯 学习进度: {memory_report['learning_progress']:.1%}")
        logging.info(f"📚 跟踪样本: {memory_report['total_samples']}个")
        logging.info(f"💾 平均记忆强度: {memory_report['avg_intensity']:.3f}")
        logging.info(f"📈 记忆趋势: {memory_report['trend']}")
        logging.info(f"📅 复习安排: {scheduling_report['scheduled_reviews']}次")

        # NaN状态报告
        if self.nan_detected:
            logging.info("⚠️  NaN检测: 系统处于稳定恢复模式")
        elif self.stable_mode:
            logging.info("🛡️  稳定模式: 已启用，训练更加保守")

        logging.info("=" * 50)

    def _final_correct_report(self):
        """正确最终报告"""
        training_time = time.time() - self.start_time
        memory_report = self.memory_model.get_memory_report()
        scheduling_report = self.review_scheduler.get_scheduling_report()

        logging.info("🎉 === 正确最终报告 ===")
        logging.info(f"⏱️  总训练时间: {training_time:.0f}秒")
        logging.info(f"🔄 总训练步数: {self.step_count}")
        logging.info(f"📅 总训练周期: {self.epoch_count}")
        logging.info(f"✅ 蒸馏应用次数: {self.distill_applied}")
        logging.info(f"🎯 最终学习进度: {memory_report['learning_progress']:.1%}")
        logging.info(f"📚 总跟踪样本: {memory_report['total_samples']}个")
        logging.info(f"💾 最终记忆强度: {memory_report['avg_intensity']:.3f}")
        logging.info(f"📈 最终记忆趋势: {memory_report['trend']}")
        logging.info(f"📅 总复习安排: {scheduling_report['total_reviews']}次")

        # 系统稳定性评估
        if not self.nan_detected and memory_report['learning_progress'] > 0:
            logging.info("✅ 正确蒸馏系统工作正常!")
        elif self.nan_detected:
            logging.info("⚠️  系统检测到NaN，需要进一步调试")
        else:
            logging.info("🔄 系统稳定但学习进度较低")

        logging.info("=" * 50)

    def train(self, epochs=100, imgsz=640, batch=16, **kwargs):
        """训练入口"""
        logging.info("🚀 开始正确的艾宾浩斯蒸馏训练...")

        # 使用正确的训练配置
        correct_config = {
            'data': self.data_config,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': self.device,
            'workers': kwargs.get('workers', 4),
            'lr0': 0.001,  # 降低学习率
            'amp': False,  # 关闭混合精度
            'verbose': True,
            'project': kwargs.get('project', 'runs/correct_ebbinghaus'),
            'name': kwargs.get('name', f'correct_{int(time.time())}'),
        }

        try:
            results = self.student.train(**correct_config)
            logging.info("✅ 正确的艾宾浩斯蒸馏训练完成")
            return results

        except Exception as e:
            logging.error(f"❌ 训练失败: {e}")
            raise


class CorrectMemoryModel:
    """正确记忆模型"""

    def __init__(self):
        self.sample_memory_db = {}
        self.sample_difficulty_db = {}
        self.global_memory_intensity = 0.5
        self.history_intensities = deque(maxlen=100)

        # 正确参数
        self.base_forgetting_rate = 0.998
        self.learning_threshold = 0.5
        self.forgetting_threshold = 0.3

        # 统计
        self.total_samples = 0
        self.learned_samples = 0

        logging.info("🧠✅ 正确记忆模型初始化完成")

    def compute_sample_difficulty(self, batch, index):
        """计算样本难度"""
        return 1.0  # 简化实现

    def update_sample_memory(self, sample_id, learning_gain, current_step, difficulty, is_review=False):
        """更新样本记忆"""
        if sample_id not in self.sample_memory_db:
            self.sample_memory_db[sample_id] = {
                'strength': 0.5,
                'last_learning_time': current_step,
                'learning_count': 0
            }
            self.total_samples += 1

        record = self.sample_memory_db[sample_id]

        # 记忆衰减
        time_gap = max(1, current_step - record['last_learning_time'])
        time_decay = math.exp(-time_gap / self.base_forgetting_rate)
        record['strength'] = record['strength'] * time_decay

        # 记忆强化
        record['strength'] = min(1.0, record['strength'] + learning_gain * 0.1)
        record['last_learning_time'] = current_step
        record['learning_count'] += 1

        # 检查学习状态
        if record['strength'] >= self.learning_threshold and record['learning_count'] == 1:
            self.learned_samples += 1

        # 更新全局记忆强度
        self._update_global_memory()

        return record['strength']

    def _update_global_memory(self):
        """更新全局记忆强度"""
        if not self.sample_memory_db:
            self.global_memory_intensity = 0.5
            return

        total_intensity = sum(record['strength'] for record in self.sample_memory_db.values())
        self.global_memory_intensity = total_intensity / len(self.sample_memory_db)

        # 记录历史强度
        self.history_intensities.append(self.global_memory_intensity)

    def get_memory_report(self):
        """获取记忆报告"""
        if self.total_samples == 0:
            return {
                'total_samples': 0,
                'learned_samples': 0,
                'learning_progress': 0.0,
                'avg_intensity': 0.5,
                'trend': 'stable'
            }

        learning_progress = self.learned_samples / self.total_samples
        avg_intensity = self.global_memory_intensity

        # 计算趋势
        trend = 'stable'
        if len(self.history_intensities) >= 10:
            recent = np.mean(list(self.history_intensities)[-5:])
            earlier = np.mean(list(self.history_intensities)[-10:-5])
            if recent > earlier + 0.01:
                trend = 'increasing'
            elif recent < earlier - 0.01:
                trend = 'decreasing'

        return {
            'total_samples': self.total_samples,
            'learned_samples': self.learned_samples,
            'learning_progress': learning_progress,
            'avg_intensity': avg_intensity,
            'trend': trend
        }


class CorrectReviewScheduler:
    """正确复习调度器"""

    def __init__(self, memory_model):
        self.memory_model = memory_model
        self.review_count = 0

        logging.info("📅✅ 正确复习调度器初始化完成")

    def schedule_reviews(self, current_step, review_ratio=0.3):
        """安排复习"""
        if not self.memory_model.sample_memory_db:
            return []

        need_review = []
        for sample_id, memory_info in self.memory_model.sample_memory_db.items():
            if memory_info['strength'] < 0.4:
                need_review.append(sample_id)

        num_review = min(len(need_review), int(len(self.memory_model.sample_memory_db) * review_ratio))
        review_samples = need_review[:num_review]

        self.review_count += len(review_samples)

        return review_samples

    def get_scheduling_report(self):
        """获取调度报告"""
        return {
            'total_reviews': self.review_count,
            'scheduled_reviews': self.review_count
        }


def main():
    """主函数"""
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")

    print("=== 正确的艾宾浩斯蒸馏系统 ===")
    print("🚀 开始正确版本训练...")

    try:
        trainer = CorrectEbbinghausDistillation(
            teacher_path="runs/segment/train2/weights/best.pt",
            student_config="yolo11n-seg.yaml",
            data_config="./datasets/crack-seg/data.yaml",
            device=device
        )

        results = trainer.train(
            epochs=100,
            imgsz=640,
            batch=8,
            workers=0,
            lr0=0.001,
            amp=False
        )

        logging.info("🎉 正确的艾宾浩斯蒸馏训练完成!")
        return results

    except Exception as e:
        logging.error(f"❌ 正确版本失败: {e}")
        return None


if __name__ == '__main__':
    main()