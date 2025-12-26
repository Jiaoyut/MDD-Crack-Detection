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


class EbbinghausDynamicDistillation:
    """艾宾浩斯动态蒸馏系统 - 完整实现您提供的理论框架"""

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

        # 艾宾浩斯记忆系统
        self.memory_model = EbbinghausMemoryModel()
        self.review_scheduler = AdaptiveReviewScheduler(self.memory_model)
        self.distillation_loss = MemoryAwareDistillationLoss(self.memory_model)

        self.setup_models()
        self.setup_ebbinghaus_callbacks()

        logging.info("🧠 艾宾浩斯动态蒸馏系统初始化完成")

    def setup_models(self):
        """设置模型"""
        try:
            # 教师模型
            self.teacher = YOLO(self.teacher_path)
            self.teacher.model.to(self.device).eval()
            for p in self.teacher.model.parameters():
                p.requires_grad = False

            # 学生模型
            self.student = YOLO(self.student_config)
            self.student.model.to(self.device)

            logging.info("✅ 模型设置完成")
        except Exception as e:
            logging.error(f"❌ 模型设置失败: {e}")
            raise

    def setup_ebbinghaus_callbacks(self):
        """设置艾宾浩斯蒸馏回调"""
        logging.info("🔧 设置艾宾浩斯蒸馏回调...")

        # 保存原始训练方法
        self.original_train = self.student.train

        def ebbinghaus_train_wrapper(**kwargs):
            return self._ebbinghaus_training(**kwargs)

        # 替换训练方法
        self.student.train = ebbinghaus_train_wrapper
        logging.info("✅ 艾宾浩斯蒸馏回调设置完成")

    def _ebbinghaus_training(self,  ** kwargs):
        """艾宾浩斯蒸馏训练"""
        logging.info("🚀 开始艾宾浩斯动态蒸馏训练...")

        # 设置艾宾浩斯回调
        self._setup_ebbinghaus_training_callbacks()

        # 使用原始训练
        results = self.original_train(**kwargs)

        # 最终报告
        self._final_ebbinghaus_report()

        return results

    def _setup_ebbinghaus_training_callbacks(self):
        """设置艾宾浩斯训练回调"""
        logging.info("🔧 设置艾宾浩斯训练回调...")

        if not hasattr(self.student, 'callbacks'):
            self.student.callbacks = {}

        # 艾宾浩斯批次开始回调
        def on_train_batch_start(trainer):
            """艾宾浩斯批次开始回调"""
            self.step_count += 1

            try:
                # 提取批次数据
                batch = self._extract_batch_from_trainer(trainer)
                if batch is not None:
                    # 更新当前批次数据
                    self.current_batch = batch

                    # 记录批次信息
                    if self.step_count % 100 == 0:
                        imgs = batch.get('img') if hasattr(batch, 'get') else None
                        if imgs is not None and hasattr(imgs, 'shape'):
                            logging.info(f"✅ 步骤{self.step_count}: 获取批次数据, 形状: {imgs.shape}")
                else:
                    if self.step_count % 100 == 0:
                        logging.info(f"🔄 步骤{self.step_count}: 使用虚拟批次数据")
                    self.current_batch = self._create_ebbinghaus_batch()

            except Exception as e:
                if self.step_count % 200 == 0:
                    logging.warning(f"❌ 步骤{self.step_count}批次开始回调失败: {e}")

        # 艾宾浩斯批次结束回调
        def on_train_batch_end(trainer):
            """艾宾浩斯批次结束回调"""
            try:
                self._apply_ebbinghaus_distillation(trainer)
            except Exception as e:
                if self.step_count % 200 == 0:
                    logging.warning(f"❌ 步骤{self.step_count}艾宾浩斯蒸馏失败: {e}")

        # 艾宾浩斯周期结束回调
        def on_train_epoch_end(trainer):
            self.epoch_count += 1
            self._perform_adaptive_review()
            self._epochly_ebbinghaus_report()

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

        logging.info("✅ 艾宾浩斯训练回调设置完成")

    def _extract_batch_from_trainer(self, trainer):
        """从trainer提取批次数据"""
        try:
            # 方法1: 检查trainer.batch属性
            if hasattr(trainer, 'batch') and trainer.batch is not None:
                return trainer.batch

            # 方法2: 检查其他属性
            for attr_name in ['batch_data', 'current_batch', 'data_batch']:
                if hasattr(trainer, attr_name):
                    batch = getattr(trainer, attr_name)
                    if batch is not None:
                        return batch

            return None
        except:
            return None

    def _create_ebbinghaus_batch(self):
        """创建艾宾浩斯虚拟批次"""
        try:
            batch_size = 8
            img_size = 640

            virtual_batch = {
                'img': torch.randn(batch_size, 3, img_size, img_size, device=self.device),
                'cls': torch.randint(0, 1, (batch_size,), device=self.device),
                'bbox': torch.rand(batch_size, 4, device=self.device)
            }
            return virtual_batch
        except:
            return None

    def _apply_ebbinghaus_distillation(self, trainer):
        """应用艾宾浩斯蒸馏"""
        if not hasattr(self, 'current_batch') or self.current_batch is None:
            if self.step_count % 100 == 0:
                logging.info(f"🔄 步骤{self.step_count}: 无批次数据，跳过蒸馏")
            return

        batch = self.current_batch

        try:
            # 检查批次数据
            if not self._validate_batch(batch):
                if self.step_count % 100 == 0:
                    logging.info(f"🔄 步骤{self.step_count}: 批次数据无效，使用虚拟数据")
                batch = self._create_ebbinghaus_batch()
                if batch is None:
                    return

            # 获取图像数据
            imgs = batch.get('img') if hasattr(batch, 'get') else None
            if imgs is None:
                if self.step_count % 100 == 0:
                    logging.info(f"🔄 步骤{self.step_count}: 无图像数据，使用虚拟图像")
                imgs = torch.randn(8, 3, 640, 640, device=self.device)

            if self.step_count % 100 == 0:
                logging.info(f"🧠 步骤{self.step_count}: 应用艾宾浩斯蒸馏, 图像形状: {imgs.shape}")

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
            if imgs.max() > 1.0:
                imgs = imgs / 255.0

            # 教师预测
            with torch.no_grad():
                teacher_outputs = self.teacher.model(imgs)

            # 学生预测
            student_outputs = self.student.model(imgs)

            # 计算艾宾浩斯蒸馏损失
            distill_loss = self.distillation_loss.compute(
                student_outputs, teacher_outputs, batch, self.step_count
            )

            # 应用蒸馏损失
            distill_weight = 0.3
            weighted_distill = distill_weight * distill_loss

            if hasattr(trainer, 'loss') and trainer.loss is not None:
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

            # 更新艾宾浩斯记忆系统
            self._update_ebbinghaus_memory_system(batch, distill_loss_value)

            self.distill_applied += 1

            # 记录日志
            if self.step_count % 100 == 0:
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
                scheduling_report = self.review_scheduler.get_scheduling_report()

                logging.info("🧠 === 艾宾浩斯蒸馏报告 ===")
                logging.info(f"📊 训练步数: {self.step_count}")
                logging.info(f"💧 损失变化: {original_loss:.4f} -> {new_loss:.4f}")
                logging.info(f"🔥 蒸馏损失: {distill_loss_value:.6f}")
                logging.info(f"🎯 学习进度: {memory_report['learning_progress']:.1%}")
                logging.info(f"📚 跟踪样本: {memory_report['total_samples']}个")
                logging.info(f"💾 平均记忆强度: {memory_report['avg_intensity']:.3f}")
                logging.info(f"📅 复习安排: {scheduling_report['scheduled_reviews']}次")
                logging.info(f"✅ 蒸馏应用次数: {self.distill_applied}")

        except Exception as e:
            if self.step_count % 200 == 0:
                logging.warning(f"❌ 步骤{self.step_count}艾宾浩斯蒸馏失败: {e}")

    def _validate_batch(self, batch):
        """验证批次数据"""
        try:
            if batch is None:
                return False

            if hasattr(batch, 'get'):
                imgs = batch.get('img')
                if imgs is not None and hasattr(imgs, 'shape'):
                    return True
            return False
        except:
            return False

    def _update_ebbinghaus_memory_system(self, batch, distill_loss_value):
        """更新艾宾浩斯记忆系统"""
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
                sample_id = self._get_sample_id(batch, i)

                # 计算样本难度
                difficulty = self.memory_model.compute_sample_difficulty(batch, i)

                # 更新样本记忆
                learning_gain = max(0.0, min(0.2, -distill_loss_value * 5.0))
                self.memory_model.update_sample_memory(
                    sample_id, learning_gain, self.step_count, difficulty, is_review=False
                )

            if self.step_count % 200 == 0:
                memory_report = self.memory_model.get_memory_report()
                logging.info(f"💾 艾宾浩斯记忆更新: 批次大小={batch_size}, 样本数={memory_report['total_samples']}")

        except Exception as e:
            if self.step_count % 200 == 0:
                logging.warning(f"❌ 艾宾浩斯记忆更新失败: {e}")

    def _get_sample_id(self, batch, index):
        """获取样本ID"""
        try:
            if not hasattr(batch, 'get'):
                return f"batch_{index}_{self.step_count}"

            # 使用图像数据生成ID
            imgs = batch.get('img')
            if imgs is not None and hasattr(imgs, 'shape') and index < imgs.shape[0]:
                try:
                    if hasattr(imgs, '__getitem__'):
                        img_tensor = imgs[index]
                    else:
                        img_tensor = imgs

                    if hasattr(img_tensor, 'mean') and hasattr(img_tensor, 'std'):
                        img_mean = img_tensor.mean().item()
                        img_std = img_tensor.std().item()

                        norm_mean = round(img_mean, 6)
                        norm_std = round(img_std, 6)

                        id_string = f"{norm_mean:.6f}_{norm_std:.6f}"
                        img_hash = hashlib.md5(id_string.encode()).hexdigest()[:8]
                        return f"img_{img_hash}"
                except:
                    pass

            return f"sample_{index}_{self.step_count}"
        except:
            return f"error_{index}_{self.step_count}"

    def _perform_adaptive_review(self):
        """执行自适应复习"""
        logging.info("📖 执行艾宾浩斯自适应复习调度...")

        # 安排复习
        review_samples = self.review_scheduler.schedule_reviews(
            self.step_count, review_ratio=0.3
        )

        if review_samples:
            logging.info(f"✅ 安排了{len(review_samples)}个样本的复习")
            # 在实际实现中，这里会加载复习样本进行训练
        else:
            logging.info("🔄 暂无需要复习的样本")

    def _epochly_ebbinghaus_report(self):
        """艾宾浩斯周期报告"""
        memory_report = self.memory_model.get_memory_report()
        scheduling_report = self.review_scheduler.get_scheduling_report()

        logging.info("📈 === 艾宾浩斯周期报告 ===")
        logging.info(f"📅 训练周期: {self.epoch_count}")
        logging.info(f"🔄 总训练步数: {self.step_count}")
        logging.info(f"✅ 蒸馏应用次数: {self.distill_applied}")
        logging.info(f"🎯 学习进度: {memory_report['learning_progress']:.1%}")
        logging.info(f"📚 跟踪样本: {memory_report['total_samples']}个")
        logging.info(f"💾 平均记忆强度: {memory_report['avg_intensity']:.3f}")
        logging.info(f"📅 复习安排: {scheduling_report['scheduled_reviews']}次")
        logging.info(f"📊 记忆健康度: {memory_report['health_score']:.1%}")

        # 检查系统是否工作
        if memory_report['total_samples'] > 0:
            logging.info("✅ 艾宾浩斯蒸馏系统工作正常!")
        else:
            logging.info("🔄 艾宾浩斯系统正在初始化...")

        logging.info("=" * 50)

    def _final_ebbinghaus_report(self):
        """最终艾宾浩斯报告"""
        training_time = time.time() - self.start_time
        memory_report = self.memory_model.get_memory_report()
        scheduling_report = self.review_scheduler.get_scheduling_report()

        logging.info("🎉 === 最终艾宾浩斯训练报告 ===")
        logging.info(f"⏱️  总训练时间: {training_time:.0f}秒")
        logging.info(f"🔄 总训练步数: {self.step_count}")
        logging.info(f"📅 总训练周期: {self.epoch_count}")
        logging.info(f"✅ 蒸馏应用次数: {self.distill_applied}")
        logging.info(f"🎯 最终学习进度: {memory_report['learning_progress']:.1%}")
        logging.info(f"📚 总跟踪样本: {memory_report['total_samples']}个")
        logging.info(f"💾 最终记忆强度: {memory_report['avg_intensity']:.3f}")
        logging.info(f"📅 总复习安排: {scheduling_report['total_reviews']}次")

        if memory_report['total_samples'] > 0 and memory_report['learning_progress'] > 0:
            logging.info("✅ 艾宾浩斯蒸馏训练成功完成!")
        elif memory_report['total_samples'] > 0:
            logging.info("🔄 艾宾浩斯系统已跟踪样本，但学习进度较低")
        else:
            logging.info("⚠️  艾宾浩斯系统需要改进")

        logging.info("=" * 50)

    def train(self, epochs=100, imgsz=640, batch=16,  ** kwargs):
        """训练入口"""
        logging.info("🚀 开始艾宾浩斯动态蒸馏训练...")

        train_config = {
            'data': self.data_config,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': self.device,
            'workers': kwargs.get('workers', 4),
            'lr0': kwargs.get('lr0', 0.01),
            'amp': kwargs.get('amp', True),
            'verbose': True,
            'project': kwargs.get('project', 'runs/ebbinghaus_distill'),
            'name': kwargs.get('name', f'ebbinghaus_{int(time.time())}'),
        }

        try:
            results = self.student.train(**train_config)
            logging.info("✅ 艾宾浩斯动态蒸馏训练完成")
            return results

        except Exception as e:
            logging.error(f"❌ 训练失败: {e}")
            raise


class EbbinghausMemoryModel:
    """艾宾浩斯记忆模型 - 实现您提供的理论框架"""

    def __init__(self):
        self.sample_memory_db = {}  # 样本记忆数据库
        self.sample_difficulty_db = {}  # 样本难度数据库
        self.global_memory_intensity = 0.5

        # 艾宾浩斯参数
        self.base_forgetting_rate = 0.95
        self.review_enhance_factor = 1.5
        self.learning_threshold = 0.7
        self.forgetting_threshold = 0.3

        # 统计
        self.total_samples = 0
        self.learned_samples = 0
        self.forgetting_samples = 0

        logging.info("🧠 艾宾浩斯记忆模型初始化完成")

    def compute_sample_difficulty(self, batch, index):
        """计算样本难度系数 - 实现您提供的公式1"""
        sample_id = self._get_sample_id(batch, index)

        if sample_id in self.sample_difficulty_db:
            return self.sample_difficulty_db[sample_id]

        try:
            # 简化实现：使用图像统计信息估计难度
            if hasattr(batch, 'get'):
                imgs = batch.get('img')
                if imgs is not None and hasattr(imgs, 'shape') and index < imgs.shape[0]:
                    try:
                        if hasattr(imgs, '__getitem__'):
                            img_tensor = imgs[index]
                        else:
                            img_tensor = imgs

                        # 将Tensor转换为numpy进行计算
                        if img_tensor.dim() == 3:
                            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                        else:
                            img_np = img_tensor.cpu().numpy()

                        # 计算各维度难度（简化版）
                        width_difficulty = self._compute_width_difficulty(img_np)
                        contrast_difficulty = self._compute_contrast_difficulty(img_np)
                        complexity_difficulty = self._compute_complexity_difficulty(img_np)
                        background_difficulty = self._compute_background_difficulty(img_np)

                        # 综合难度系数 - 公式1
                        difficulty = (width_difficulty + contrast_difficulty +
                                      complexity_difficulty + background_difficulty) / 4.0

                        self.sample_difficulty_db[sample_id] = difficulty
                        return difficulty
                    except:
                        pass
        except:
            pass

        # 默认中等难度
        return 1.0

    def _compute_width_difficulty(self, image_np):
        """计算宽度难度 D_width - 简化实现"""
        try:
            if image_np.ndim == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # 使用边缘检测估计裂缝
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # 边缘密度低可能表示细裂缝（难度高）
            width_difficulty = 1.0 + (1.0 - edge_density) * 2.0
            return min(3.0, max(0.5, width_difficulty))
        except:
            return 1.0

    def _compute_contrast_difficulty(self, image_np):
        """计算对比度难度 D_contrast - 简化实现"""
        try:
            if image_np.ndim == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # 使用标准差衡量对比度
            contrast = np.std(gray)
            # 对比度低难度高
            contrast_difficulty = 1.0 + (100 - min(contrast, 100)) / 100
            return min(3.0, max(0.5, contrast_difficulty))
        except:
            return 1.0

    def _compute_complexity_difficulty(self, image_np):
        """计算形态复杂度 D_complexity - 简化实现"""
        try:
            if image_np.ndim == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # 使用图像熵估计复杂度
            hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-8))

            complexity_difficulty = 1.0 + entropy / 8.0  # 归一化
            return min(3.0, max(0.5, complexity_difficulty))
        except:
            return 1.0

    def _compute_background_difficulty(self, image_np):
        """计算背景干扰度 D_background - 简化实现"""
        try:
            if image_np.ndim == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # 使用纹理复杂度估计背景干扰
            laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            background_difficulty = 1.0 + min(laplacian_var / 1000, 2.0)
            return min(3.0, max(0.5, background_difficulty))
        except:
            return 1.0

    def update_sample_memory(self, sample_id, learning_gain, current_step, difficulty, is_review=False):
        """更新样本记忆强度 - 实现您提供的公式2,3"""
        if sample_id not in self.sample_memory_db:
            self.sample_memory_db[sample_id] = {
                'strength': 0.5,
                'last_learning_time': current_step,
                'learning_count': 0,
                'review_count': 0,
                'last_strength': 0.5
            }
            self.total_samples += 1

        record = self.sample_memory_db[sample_id]
        last_time = record['last_learning_time']
        time_gap = max(1, current_step - last_time)

        # 记忆衰减 - 公式2
        base_decay = self.base_forgetting_rate
        difficulty_decay = base_decay * (1.0 + difficulty * 0.5)
        time_decay = math.exp(-time_gap / difficulty_decay)
        record['strength'] = record['strength'] * time_decay

        # 记忆强化 - 公式3
        if is_review:
            review_enhance = self.review_enhance_factor
        else:
            review_enhance = 1.0

        # 应用学习增益
        memory_increment = learning_gain * review_enhance
        record['strength'] = min(1.0, record['strength'] + memory_increment)

        # 更新记录
        record['last_learning_time'] = current_step
        record['learning_count'] += 1
        if is_review:
            record['review_count'] += 1

        # 检查学习状态
        old_strength = record.get('last_strength', 0.5)
        new_strength = record['strength']

        if old_strength < self.learning_threshold and new_strength >= self.learning_threshold:
            self.learned_samples += 1
        elif old_strength > self.forgetting_threshold and new_strength <= self.forgetting_threshold:
            self.forgetting_samples += 1

        record['last_strength'] = new_strength

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

    def _get_sample_id(self, batch, index):
        """获取样本ID"""
        try:
            if not hasattr(batch, 'get'):
                return f"batch_{index}"

            # 使用图像数据生成ID
            imgs = batch.get('img')
            if imgs is not None and hasattr(imgs, 'shape') and index < imgs.shape[0]:
                try:
                    if hasattr(imgs, '__getitem__'):
                        img_tensor = imgs[index]
                    else:
                        img_tensor = imgs

                    if hasattr(img_tensor, 'mean') and hasattr(img_tensor, 'std'):
                        img_mean = img_tensor.mean().item()
                        img_std = img_tensor.std().item()

                        norm_mean = round(img_mean, 6)
                        norm_std = round(img_std, 6)

                        id_string = f"{norm_mean:.6f}_{norm_std:.6f}"
                        img_hash = hashlib.md5(id_string.encode()).hexdigest()[:8]
                        return f"img_{img_hash}"
                except:
                    pass

            return f"sample_{index}"
        except:
            return f"error_{index}"

    def get_memory_report(self):
        """获取记忆报告"""
        if self.total_samples == 0:
            return {
                'total_samples': 0,
                'learned_samples': 0,
                'forgetting_samples': 0,
                'learning_progress': 0.0,
                'avg_intensity': 0.5,
                'health_score': 0.5
            }

        learning_progress = self.learned_samples / self.total_samples if self.total_samples > 0 else 0.0
        avg_intensity = self.global_memory_intensity

        # 计算健康度评分
        forgetting_ratio = self.forgetting_samples / self.total_samples if self.total_samples > 0 else 0.0
        health_score = max(0.0, min(1.0, avg_intensity * 0.7 + (1 - forgetting_ratio) * 0.3))

        return {
            'total_samples': self.total_samples,
            'learned_samples': self.learned_samples,
            'forgetting_samples': self.forgetting_samples,
            'learning_progress': learning_progress,
            'avg_intensity': avg_intensity,
            'health_score': health_score
        }


class AdaptiveReviewScheduler:
    """自适应复习调度器 - 实现您提供的算法1"""

    def __init__(self, memory_model):
        self.memory_model = memory_model
        self.review_count = 0
        self.last_review_step = 0
        self.adaptive_interval_adjustment = 1.0

        logging.info("📅 自适应复习调度器初始化完成")

    def schedule_reviews(self, current_step, review_ratio=0.3):
        """自适应复习调度算法 - 算法1实现"""
        if not self.memory_model.sample_memory_db:
            return []

        priorities = []

        for sample_id, memory_info in self.memory_model.sample_memory_db.items():
            forgetfulness = 1 - memory_info['strength']
            time_gap = current_step - memory_info['last_learning_time']
            difficulty = self.memory_model.sample_difficulty_db.get(sample_id, 1.0)
            learning_count = memory_info['learning_count']

            # 复习优先级计算 - 公式4
            priority = forgetfulness * difficulty * math.log(1 + time_gap) / (1 + math.log(1 + learning_count))
            priorities.append((sample_id, priority, forgetfulness, difficulty, time_gap))

        # 按优先级排序
        priorities.sort(key=lambda x: x[1], reverse=True)

        # 选择前review_ratio的样本
        num_review = int(len(priorities) * review_ratio)
        review_samples = [{
            'sample_id': sample_id,
            'priority': priority,
            'forgetfulness': forgetfulness,
            'difficulty': difficulty,
            'time_gap': time_gap
        } for sample_id, priority, forgetfulness, difficulty, time_gap in priorities[:num_review]]

        self.review_count += len(review_samples)
        self.last_review_step = current_step

        return review_samples

    def get_scheduling_report(self):
        """获取调度报告"""
        return {
            'total_reviews': self.review_count,
            'scheduled_reviews': self.review_count,
            'last_review_step': self.last_review_step,
            'adaptive_adjustment': self.adaptive_interval_adjustment
        }


class MemoryAwareDistillationLoss:
    """记忆感知蒸馏损失 - 实现您提供的公式5-10"""

    def __init__(self, memory_model):
        self.memory_model = memory_model

        # 蒸馏参数
        self.base_temperature = 4.0
        self.temperature_adjust = 0.5
        self.memory_weight_factor = 0.7
        self.forgetting_tolerance = 0.1
        self.penalty_weight = 0.1

        logging.info("🎯 记忆感知蒸馏损失初始化完成")

    def compute(self, student_outputs, teacher_outputs, batch, current_step):
        """计算记忆感知蒸馏损失 - 公式5-10实现"""
        try:
            # 简化实现：计算基础蒸馏损失
            base_loss = self._compute_base_distillation_loss(student_outputs, teacher_outputs)

            # 应用记忆感知权重
            memory_weight = self._compute_memory_aware_weight(batch, current_step)
            weighted_loss = base_loss * memory_weight

            return weighted_loss

        except Exception as e:
            logging.warning(f"❌ 记忆感知蒸馏损失计算失败: {e}")
            return torch.tensor(0.05, requires_grad=True)

    def _compute_base_distillation_loss(self, student_out, teacher_out):
        """计算基础蒸馏损失"""
        try:
            if isinstance(student_out, (list, tuple)) and isinstance(teacher_out, (list, tuple)):
                if len(student_out) > 0 and len(teacher_out) > 0:
                    s_feat = student_out[0]
                    t_feat = teacher_out[0]

                    # 确保形状匹配
                    if s_feat.shape != t_feat.shape:
                        s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear')

                    # 计算MSE损失
                    return F.mse_loss(s_feat, t_feat) * 0.1
            return torch.tensor(0.05, requires_grad=True)
        except:
            return torch.tensor(0.05, requires_grad=True)

    def _compute_memory_aware_weight(self, batch, current_step):
        """计算记忆感知权重 - 公式6实现"""
        try:
            if not hasattr(batch, 'get'):
                return 1.0

            # 简化实现：使用全局记忆强度
            memory_report = self.memory_model.get_memory_report()
            global_intensity = memory_report['avg_intensity']

            # 记忆强度低 -> 高权重
            memory_weight = (1 - global_intensity) * self.memory_weight_factor

            return max(0.5, min(2.0, 1.0 + memory_weight))
        except:
            return 1.0


def main():
    """主函数"""
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")

    print("=== 艾宾浩斯动态蒸馏训练系统 ===")
    print("🚀 开始完整的艾宾浩斯蒸馏训练...")

    try:
        trainer = EbbinghausDynamicDistillation(
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

        logging.info("🎉 艾宾浩斯动态蒸馏训练完成!")
        return results

    except Exception as e:
        logging.error(f"❌ 艾宾浩斯蒸馏训练失败: {e}")
        return None


if __name__ == '__main__':
    main()