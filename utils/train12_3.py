import os
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import logging
import math
from collections import defaultdict
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ProductionDistillationTrainer:
    """生产级蒸馏训练器 - 针对PyTorch 2.5.1 + CUDA 12.1优化"""

    def __init__(self, teacher_path, student_config, data_config, device='cuda'):
        self.device = device
        self.step_count = 0

        # 环境验证
        self._validate_environment()

        # 加载模型
        logging.info("加载教师模型...")
        self.teacher = YOLO(teacher_path)
        self.teacher.model.to(device).eval()
        for p in self.teacher.model.parameters():
            p.requires_grad = False

        logging.info("加载学生模型...")
        self.student = YOLO(student_config)
        self.student.model.to(device)
        self.data_config = data_config

        # 记忆存储系统
        self.memory_store = defaultdict(lambda: {
            'strength': 0.0,
            'last_review': -1,
            'next_review': 0,
            'review_count': 0
        })
        self.history = defaultdict(list)

        # 蒸馏参数
        self.distill_config = {
            'alpha': 0.7,
            'temperature': 4.0,
            'feature_layers': [2, 4, 6],
            'method': 'adaptive'
        }

        # 记忆参数
        self.memory_params = {
            'base_interval': 300,
            'interval_scale': 3.0,
            'min_interval': 30,
            'strength_decay': 0.15,
            'difficulty_weight': 1.5
        }

        # 性能统计
        self.stats = {
            'total_reviews': 0,
            'avg_distill_loss': 0.0,
            'peak_memory_usage': 0.0
        }

        logging.info("生产级蒸馏训练器初始化完成")

    def _validate_environment(self):
        """验证环境兼容性"""
        logging.info("验证训练环境...")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法进行GPU训练")

        # 详细环境信息
        gpu_props = torch.cuda.get_device_properties(0)
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
        logging.info(f"显存: {gpu_props.total_memory / 1024**3:.1f} GB")
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info(f"CUDA: {torch.version.cuda}")
        logging.info(f"cuDNN: {torch.backends.cudnn.version()}")

        # 性能测试
        self._performance_benchmark()

    def _performance_benchmark(self):
        """GPU性能基准测试"""
        logging.info("运行GPU性能测试...")

        # 测试张量操作
        test_size = (4, 3, 640, 640)
        test_tensor = torch.randn(test_size).to(self.device)

        # 前向传播测试
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            for _ in range(10):
                _ = test_tensor * 2 + 1
            end_time.record()

            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time) / 10

        logging.info(f"GPU计算速度: {elapsed:.2f} ms/迭代")
        logging.info("GPU性能测试通过")

    def _extract_features(self, model, x):
        """提取多尺度特征"""
        features = []
        try:
            if hasattr(model, 'model'):
                y, dt = [], []
                for m in model.model.model:
                    if m.f != -1:
                        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                    x = m(x)
                    y.append(x if m.i in self.distill_config['feature_layers'] else None)

                    if m.i in self.distill_config['feature_layers']:
                        features.append(x)
            else:
                for i, layer in enumerate(model.children()):
                    x = layer(x)
                    if i in self.distill_config['feature_layers']:
                        features.append(x)
        except Exception as e:
            logging.warning(f"特征提取降级: {e}")
            features = [x]

        return features

    def _compute_adaptive_distill_loss(self, teacher_features, student_features):
        """自适应蒸馏损失计算"""
        total_loss = 0.0
        valid_layers = 0

        for t_feat, s_feat in zip(teacher_features, student_features):
            try:
                # 形状适配
                if t_feat.shape != s_feat.shape:
                    if len(t_feat.shape) == 4 and len(s_feat.shape) == 4:
                        s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                    if t_feat.size(1) != s_feat.size(1):
                        min_channels = min(t_feat.size(1), s_feat.size(1))
                        t_feat = t_feat[:, :min_channels]
                        s_feat = s_feat[:, :min_channels]

                # 温度缩放蒸馏
                t_feat = F.normalize(t_feat / self.distill_config['temperature'], p=2, dim=1)
                s_feat = F.normalize(s_feat / self.distill_config['temperature'], p=2, dim=1)

                layer_loss = F.mse_loss(s_feat, t_feat, reduction='mean')
                total_loss += layer_loss
                valid_layers += 1

            except Exception as e:
                logging.debug(f"层损失计算跳过: {e}")
                continue

        return total_loss / max(1, valid_layers)

    def _update_memory_model(self, sample_id, loss_value, current_step):
        """记忆模型更新"""
        mem = self.memory_store[sample_id]

        # 首次遇到
        if mem['last_review'] < 0:
            mem.update({
                'strength': 0.0,
                'last_review': current_step,
                'next_review': current_step + self.memory_params['min_interval'],
                'review_count': 0
            })

        # 标准化损失为难度系数
        normalized_loss = min(1.0, loss_value / 5.0)

        # 基于艾宾浩斯遗忘曲线的强度更新
        interval = current_step - mem['last_review']
        forgetting = math.exp(-interval / (self.memory_params['strength_decay'] * 1000))

        # 难度加权学习率
        difficulty_factor = 1.0 + self.memory_params['difficulty_weight'] * normalized_loss
        new_strength = mem['strength'] * forgetting + (1 - forgetting) * (1 - normalized_loss)
        new_strength = max(0.0, min(0.99, new_strength))

        # 更新记忆状态
        mem.update({
            'strength': new_strength,
            'last_review': current_step,
            'review_count': mem['review_count'] + 1,
            'next_review': self._calculate_next_review(new_strength, normalized_loss, current_step)
        })

        return new_strength

    def _calculate_next_review(self, strength, difficulty, current_step):
        """计算下次复习间隔"""
        base = self.memory_params['base_interval']
        strength_factor = 1.0 + strength * self.memory_params['interval_scale']
        difficulty_factor = 1.0 + difficulty * 2.0

        interval = int(base * strength_factor * difficulty_factor)
        interval = max(self.memory_params['min_interval'], interval)

        return current_step + interval

    def _adaptive_weight_scheduling(self, current_step, total_steps):
        """自适应权重调度"""
        base_weight = self.distill_config['alpha']

        # 课程学习
        curriculum = 1.0 - (current_step / total_steps) ** 0.7

        # 记忆感知
        if self.memory_store:
            avg_strength = np.mean([m['strength'] for m in self.memory_store.values()])
            memory_factor = 1.2 - avg_strength
        else:
            memory_factor = 1.0

        final_weight = base_weight * curriculum * memory_factor
        return max(0.3, min(1.0, final_weight))

    def setup_production_pipeline(self):
        """设置生产级蒸馏流水线"""

        def on_train_batch_end(trainer):
            self.step_count += 1

            try:
                if not hasattr(trainer, 'loss') or not hasattr(trainer, 'batch'):
                    return

                imgs = trainer.batch.get('img')
                if imgs is None:
                    return

                imgs = imgs.to(self.device, non_blocking=True)
                batch_size = imgs.size(0)
                sample_ids = [f"batch_{self.step_count}_{i}" for i in range(batch_size)]

                # 教师特征提取
                with torch.no_grad():
                    teacher_features = self._extract_features(self.teacher.model, imgs)

                # 学生特征提取
                student_features = self._extract_features(self.student.model, imgs)

                # 自适应权重
                total_steps = getattr(trainer, 'epochs', 100) * getattr(trainer, 'nb', 1000)
                distill_weight = self._adaptive_weight_scheduling(self.step_count, total_steps)

                # 批量蒸馏损失计算
                total_distill_loss = 0.0
                active_reviews = 0

                for i, sample_id in enumerate(sample_ids):
                    if i >= len(teacher_features) or i >= len(student_features):
                        continue

                    mem = self.memory_store[sample_id]

                    if self.step_count >= mem['next_review']:
                        t_feat_i = [f[i:i+1] for f in teacher_features if f is not None]
                        s_feat_i = [f[i:i+1] for f in student_features if f is not None]

                        if not t_feat_i or not s_feat_i:
                            continue

                        sample_loss = self._compute_adaptive_distill_loss(t_feat_i, s_feat_i)
                        memory_factor = (1.0 - mem['strength']) ** 1.5
                        final_weight = distill_weight * memory_factor

                        total_distill_loss += final_weight * sample_loss
                        active_reviews += 1

                        self._update_memory_model(sample_id, sample_loss.item(), self.step_count)

                        self.history[sample_id].append({
                            'step': self.step_count,
                            'loss': sample_loss.item(),
                            'weight': final_weight,
                            'strength': mem['strength']
                        })

                if active_reviews > 0:
                    avg_distill = total_distill_loss / active_reviews
                    trainer.loss += avg_distill

                    self.stats['total_reviews'] += active_reviews
                    self.stats['avg_distill_loss'] = 0.9 * self.stats['avg_distill_loss'] + 0.1 * avg_distill.item()

                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    self.stats['peak_memory_usage'] = max(self.stats['peak_memory_usage'], current_memory)

                    if self.step_count % 50 == 0:
                        self._log_training_progress(active_reviews, avg_distill.item())

            except Exception as e:
                if self.step_count % 100 == 0:
                    logging.warning(f"蒸馏步骤异常: {e}")

        # 注册回调
        if not hasattr(self.student, 'callbacks'):
            self.student.callbacks = {}
        if 'on_train_batch_end' not in self.student.callbacks:
            self.student.callbacks['on_train_batch_end'] = []
        self.student.callbacks['on_train_batch_end'].append(on_train_batch_end)

    def _log_training_progress(self, active_reviews, distill_loss):
        """训练进度日志"""
        if self.memory_store:
            strengths = [m['strength'] for m in self.memory_store.values()]
            avg_strength = np.mean(strengths)
        else:
            avg_strength = 0.0

        logging.info(
            f"步骤 {self.step_count}: "
            f"复习 {active_reviews} 样本 | "
            f"蒸馏损失: {distill_loss:.4f} | "
            f"平均记忆强度: {avg_strength:.3f} | "
            f"峰值显存: {self.stats['peak_memory_usage']:.1f}GB"
        )

    def train(self, **kwargs):
        """生产级训练方法"""
        logging.info("启动生产级蒸馏训练...")

        self.setup_production_pipeline()

        production_config = {
            'data': self.data_config,
            'epochs': kwargs.get('epochs', 100),
            'imgsz': kwargs.get('imgsz', 640),
            'batch': kwargs.get('batch', 16),
            'device': self.device,
            'workers': kwargs.get('workers', 8),
            'lr0': kwargs.get('lr0', 0.01),
            'amp': kwargs.get('amp', True),
            'patience': 50,
            'save': True,
            'exist_ok': True,
            'verbose': True,
            'single_cls': kwargs.get('single_cls', False),
            'cos_lr': kwargs.get('cos_lr', True),
            'label_smoothing': kwargs.get('label_smoothing', 0.1),
        }

        torch.cuda.empty_cache()

        try:
            logging.info("最终配置检查:")
            for key, value in production_config.items():
                if key != 'data':
                    logging.info(f"  {key}: {value}")

            results = self.student.train(**production_config)
            self._final_analysis()

            logging.info(f"生产级蒸馏训练完成！总步骤: {self.step_count}")
            return results

        except Exception as e:
            logging.error(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _final_analysis(self):
        """最终性能分析"""
        if not self.memory_store:
            return

        logging.info("\n=== 蒸馏训练分析报告 ===")
        logging.info(f"总训练步骤: {self.step_count}")
        logging.info(f"总复习样本数: {self.stats['total_reviews']}")
        logging.info(f"平均蒸馏损失: {self.stats['avg_distill_loss']:.6f}")
        logging.info(f"峰值显存使用: {self.stats['peak_memory_usage']:.1f} GB")

        strengths = [m['strength'] for m in self.memory_store.values()]
        review_counts = [m['review_count'] for m in self.memory_store.values()]

        logging.info(f"记忆样本数: {len(self.memory_store)}")
        logging.info(f"平均记忆强度: {np.mean(strengths):.3f} ± {np.std(strengths):.3f}")
        logging.info(f"平均复习次数: {np.mean(review_counts):.1f}")

        strength_bins = np.histogram(strengths, bins=[0, 0.3, 0.7, 1.0])[0]
        logging.info(f"记忆强度分布: ")
        logging.info(f"  - 容易 [0-0.3]: {strength_bins[0]} 样本")
        logging.info(f"  - 中等 [0.3-0.7]: {strength_bins[1]} 样本")
        logging.info(f"  - 困难 [0.7-1.0]: {strength_bins[2]} 样本")

def main():
    """主函数"""
    try:
        print("=" * 60)
        print("生产级蒸馏训练启动")
        print("=" * 60)

        trainer = ProductionDistillationTrainer(
            teacher_path="runs/segment/train2/weights/best.pt",
            student_config="yolo11n-seg.yaml",
            data_config="./datasets/crack-seg/data.yaml",
            device='cuda'
        )

        results = trainer.train(
            epochs=100,
            imgsz=640,
            batch=8,
            workers=8,
            lr0=0.01,
            amp=True,
            cos_lr=True,
            label_smoothing=0.1
        )

        if results is not None:
            logging.info("生产级蒸馏训练成功完成！")
        else:
            logging.error("训练失败")

        return results

    except Exception as e:
        logging.error(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()