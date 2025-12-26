# import os
# from ultralytics import YOLO
# import torch
# import torch.nn.functional as F
# import logging
#
# # 设置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
#
# class SimpleDistillationTrainer:
#     """简化版蒸馏训练器 - 更可靠"""
#
#     def __init__(self, teacher_path, student_config, data_config, device='cuda'):
#         self.device = device
#
#         # 加载模型
#         self.teacher = YOLO(teacher_path)
#         self.teacher.model.to(device).eval()
#         for p in self.teacher.model.parameters():
#             p.requires_grad = False
#
#         self.student = YOLO(student_config)
#         self.student.model.to(device)
#
#         self.data_config = data_config
#         self.step_count = 0
#
#         logging.info(" 简化蒸馏训练器初始化完成")
#
#     # def setup_distillation_callback(self):
#     #     """设置蒸馏回调"""
#     #
#     #     def on_train_batch_end(trainer):
#     #         self.step_count += 1
#     #
#     #         try:
#     #             # 基本检查
#     #             if not hasattr(trainer, 'loss') or trainer.loss is None:
#     #                 return
#     #             if not hasattr(trainer, 'batch') or trainer.batch is None:
#     #                 return
#     #
#     #             # 记录原始损失
#     #             original_loss = trainer.loss.item()
#     #
#     #             # 应用蒸馏损失
#     #             distill_loss = torch.tensor(0.05, device=self.device, requires_grad=True)
#     #             trainer.loss = trainer.loss + 0.3 * distill_loss
#     #
#     #             # 记录日志
#     #             if self.step_count % 100 == 0:
#     #                 new_loss = trainer.loss.item()
#     #                 logging.info(f" 步骤 {self.step_count}: 蒸馏应用成功")
#     #                 logging.info(f" 损失变化: {original_loss:.4f} -> {new_loss:.4f}")
#     #
#     #         except Exception as e:
#     #             if self.step_count % 200 == 0:
#     #                 logging.warning(f"蒸馏回调错误: {e}")
#     #
#     #     # 注册回调
#     #     if not hasattr(self.student, 'callbacks'):
#     #         self.student.callbacks = {}
#     #     if 'on_train_batch_end' not in self.student.callbacks:
#     #         self.student.callbacks['on_train_batch_end'] = []
#     #     self.student.callbacks['on_train_batch_end'].append(on_train_batch_end)
#
#     def setup_distillation_callback(self):
#         """设置蒸馏回调"""
#
#         # 艾宾浩斯蒸馏参数
#         w0 = 0.3  # 初始权重
#         tau = 2000  # 衰减速度（可调）
#
#         def on_train_batch_end(trainer):
#             self.step_count += 1
#
#             try:
#                 if not hasattr(trainer, 'loss') or trainer.loss is None:
#                     return
#                 if not hasattr(trainer, 'batch') or trainer.batch is None:
#                     return
#
#                 # 原始损失
#                 original_loss = trainer.loss.item()
#
#                 # ---------------------------
#                 #  计算艾宾浩斯蒸馏权重
#                 # ---------------------------
#                 ebbinghaus_weight = w0 * torch.exp(
#                     torch.tensor(- self.step_count / tau, device=self.device)
#                 )
#
#                 # ---------------------------
#                 #  计算 teacher-student logits 蒸馏损失
#                 # ---------------------------
#                 # teacher 前向
#                 with torch.no_grad():
#                     t_out = self.teacher.model(trainer.batch['img'].to(self.device))
#
#                 # student 前向（YOLO 框架中 trainer.loss 已经算过，这里重新 forward 是必须的）
#                 s_out = self.student.model(trainer.batch['img'].to(self.device))
#
#                 # 简化：只对最后一层输出进行蒸馏（最常用）
#                 # 你也可以做 KL/Feature distillation
#                 distill_loss = F.mse_loss(s_out[0], t_out[0])
#
#                 # ---------------------------
#                 #  组合总损失
#                 # ---------------------------
#                 trainer.loss = trainer.loss + ebbinghaus_weight * distill_loss
#
#                 # 打印日志
#                 if self.step_count % 100 == 0:
#                     logging.info(f" Step {self.step_count}: Ebbinghaus Weight = {ebbinghaus_weight.item():.6f}")
#                     logging.info(f" Distill Loss = {distill_loss.item():.4f}")
#                     logging.info(f"Total Loss: {original_loss:.4f} -> {trainer.loss.item():.4f}")
#
#             except Exception as e:
#                 if self.step_count % 200 == 0:
#                     logging.warning(f"蒸馏回调错误: {e}")
#
#         # 注册回调
#         if not hasattr(self.student, 'callbacks'):
#             self.student.callbacks = {}
#         if 'on_train_batch_end' not in self.student.callbacks:
#             self.student.callbacks['on_train_batch_end'] = []
#         self.student.callbacks['on_train_batch_end'].append(on_train_batch_end)
#
#     def train(self,  ** kwargs):
#         """训练方法"""
#         logging.info(" 开始简化蒸馏训练...")
#
#         # 设置蒸馏回调
#         self.setup_distillation_callback()
#
#         # 训练配置
#         config = {
#             'data': self.data_config,
#             'epochs': kwargs.get('epochs', 100),
#             'imgsz': kwargs.get('imgsz', 640),
#             'batch': kwargs.get('batch', 8),
#             'device': self.device,
#             'workers': kwargs.get('workers', 0),
#             'lr0': kwargs.get('lr0', 0.001),
#             'amp': kwargs.get('amp', False),
#             'verbose': True,
#         }
#
#         results = self.student.train(**config)
#         logging.info(f" 简化蒸馏训练完成，总步骤: {self.step_count}")
#         return results
#
#
# def main():
#     """主函数"""
#     try:
#         trainer = SimpleDistillationTrainer(
#             teacher_path="runs/segment/train2/weights/best.pt",
#             student_config="yolo11n-seg.yaml",
#             data_config="./datasets/crack-seg/data.yaml",
#             device='cuda' if torch.cuda.is_available() else 'cpu'
#         )
#
#         results = trainer.train(
#             epochs=100,
#             imgsz=640,
#             batch=8,
#             workers=0,
#             lr0=0.001,
#             amp=False
#         )
#
#         logging.info(" 蒸馏训练完成!")
#         return results
#
#     except Exception as e:
#         logging.error(f" 训练失败: {e}")
#         return None
#
#
# if __name__ == '__main__':
#     main()





import os
from ultralytics import YOLO
import torch
# torch.backends.cudnn.enabled = False
import torch.nn.functional as F
import logging
import math
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SimpleDistillationTrainer:
    """简化版蒸馏训练器 - 增加记忆建模 + 动态复习调度 + 记忆感知蒸馏"""

    def __init__(self, teacher_path, student_config, data_config, device='cuda'):
        self.device = device

        # 加载模型
        self.teacher = YOLO(teacher_path)
        self.teacher.model.to(device).eval()
        for p in self.teacher.model.parameters():
            p.requires_grad = False

        self.student = YOLO(student_config)
        self.student.model.to(device)

        self.data_config = data_config
        self.step_count = 0

        # ---------- Memory modules state ----------
        # memory_store: sample_id -> dict(strength: float [0,1], last_review: int, next_review: int)
        self.memory_store = defaultdict(lambda: {'strength': 0.0, 'last_review': -1, 'next_review': 0})
        # per-sample history (optional): could store last loss etc
        self.history = defaultdict(list)

        # ---------- Hyperparams for Ebbinghaus+memory ----------
        self.ebbinghaus_w0 = 0.3   # initial base distill weight
        self.ebbinghaus_tau = 2000  # global exponential decay tau

        # memory update params
        self.memory_alpha = 0.3    # 更新记忆强度的速率 (EMA)
        self.max_strength = 0.999
        self.min_strength = 0.0

        # dynamic review scheduling params
        self.base_interval = 500    # 基本复习间隔（步骤）
        self.interval_scale = 4.0   # strength 越大，间隔越长 (next_interval = base_interval * (1 + strength*interval_scale))
        self.minimum_interval = 50  # 最小间隔

        # threshold: 当 step >= next_review 时，会对该样本进行“复习/蒸馏”
        # memory-aware weighting: memory_factor = (1 - strength) ** power
        self.memory_power = 1.0

        logging.info(" 简化蒸馏训练器初始化完成")

    # ----------------- 辅助函数 -----------------
    def extract_sample_ids(self, batch):
        """
        从 trainer.batch 尝试提取样本唯一 id 列表（长度 == batch_size）
        支持多个可能出现的字段名；若均未找到，则退回生成伪 id（step_index）
        返回: list of sample_id (str)
        """
        # 常见 ultralytics batch 字段尝试（根据你框架的实际字段做调整）
        if isinstance(batch, dict):
            for k in ('img_files', 'file_name', 'filenames', 'files', 'image_path', 'paths'):
                if k in batch and batch[k] is not None:
                    # 期待是 list-like
                    try:
                        return [str(x) for x in batch[k]]
                    except Exception:
                        pass

            # 有的训练器把 images 放在 batch['imgs'] / batch['img']
            # 我们无法从 tensor 本身获得文件名；回退到 index-based ids
        # fallback: create stable ids based on current global step and local batch index
        batch_size = None
        if isinstance(batch, dict) and 'img' in batch:
            try:
                batch_size = batch['img'].shape[0]
            except Exception:
                batch_size = None
        # 如果还能拿到 trainer.batch[0] style, 也可检查
        if batch_size is None:
            # 最后手段：设为 1 并返回 a single pseudo id
            return [f"pseudo_{self.step_count}_0"]
        else:
            return [f"pseudo_{self.step_count}_{i}" for i in range(batch_size)]

    def compute_next_review_interval(self, strength):
        """
        基于当前记忆强度计算下一次复习的间隔（steps）。
        strength ∈ [0,1]，strength 越高，间隔越长。
        """
        interval = int(self.base_interval * (1.0 + strength * self.interval_scale))
        return max(self.minimum_interval, interval)

    def update_memory_strength(self, sample_id, review_loss_norm):
        """
        使用简单 EMA 更新记忆强度：
          新_strength = (1 - alpha) * old + alpha * (1 - normalized_loss)
        normalized_loss 期望 ∈ [0,1], 越小表示掌握越好。
        """
        old = self.memory_store[sample_id]['strength']
        new = (1.0 - self.memory_alpha) * old + self.memory_alpha * (1.0 - review_loss_norm)
        new = max(self.min_strength, min(self.max_strength, new))
        self.memory_store[sample_id]['strength'] = new
        return new

    def normalize_loss_for_memory(self, loss_value):
        """
        将 distill_loss 或 review 損失 标准化到 [0,1] 用来更新记忆强度。
        这里采用简单的平滑归一（可以换成更复杂的 percentile 或 running-stat）
        """
        # simple soft normalization: sigmoid-like mapping scaled by a factor
        # 先 clamp loss to a reasonable range
        val = float(loss_value)
        val = max(0.0, val)
        # 参数 c 控制尺度，越大越平缓。可调。
        c = 1.0
        norm = 1.0 - (1.0 / (1.0 + val / c))  # maps 0->0, large->1
        # invert so that small loss -> small norm ; we want normalized_loss in [0,1] with small meaning mastered
        return min(1.0, max(0.0, norm))

    # ----------------- 主回调（带记忆的艾宾浩斯蒸馏） -----------------
    def setup_distillation_callback(self):
        """设置蒸馏回调（含记忆建模 + 动态复习调度 + 记忆感知损失）"""

        def on_train_batch_end(trainer):
            self.step_count += 1

            try:
                # 基本检查
                if not hasattr(trainer, 'loss') or trainer.loss is None:
                    return
                if not hasattr(trainer, 'batch') or trainer.batch is None:
                    return

                # 记录原始损失
                original_loss = trainer.loss.item()

                # ------------- 计算 Ebbinghaus 全局权重 -------------
                ebbinghaus_weight = self.ebbinghaus_w0 * math.exp(- self.step_count / max(1.0, self.ebbinghaus_tau))
                ebbinghaus_weight = float(ebbinghaus_weight)  # convert to python float; will wrap as tensor later

                # ------------- 提取样本 ids（长度 == batch size） -------------
                sample_ids = self.extract_sample_ids(trainer.batch)

                # ------------- teacher / student 前向（按 batch） -------------
                # 这里对 batch_image 做一次 teacher forward (no_grad) 并 student forward（用于计算 distillation targets）
                imgs = trainer.batch.get('img') or trainer.batch.get('imgs')  # 可能键名不同
                if imgs is None:
                    # 无法进行基于图像的蒸馏 — 仅记录并返回
                    if self.step_count % 200 == 0:
                        logging.warning("batch 中未找到 'img' 或 'imgs'，无法执行图像级蒸馏。")
                    return

                imgs = imgs.to(self.device)

                with torch.no_grad():
                    t_out = self.teacher.model(imgs)

                s_out = self.student.model(imgs)

                # ------------- 计算 per-sample distillation loss & memory-aware weighting -------------
                # 我们假设 t_out 和 s_out 的第0项为 feature/logit 张量，且 batch 对应顺序一致
                # 你可根据具体 teacher/student 输出结构修改索引
                # 如果输出不是 list/tuple，尝试直接使用张量
                def get_primary_output(x):
                    if isinstance(x, (list, tuple)):
                        return x[0]
                    return x

                t_primary = get_primary_output(t_out)
                s_primary = get_primary_output(s_out)

                # 若 primary 输出 shape 与 batch size 不一致，回退并做整体 mse
                try:
                    # distill_per_sample: list len == batch_size
                    bs = s_primary.shape[0]
                except Exception:
                    bs = None

                total_distill = torch.tensor(0.0, device=self.device)
                active_reviews = 0

                # iterate per sample and decide whether to "review"
                for i, sid in enumerate(sample_ids):
                    # sanity: ensure i within bs
                    if bs is None or i >= bs:
                        break

                    mem = self.memory_store[sid]
                    # 如果首次遇到该样本，mem fields 有默认值
                    if mem['last_review'] < 0:
                        mem['strength'] = 0.0
                        mem['last_review'] = 0
                        mem['next_review'] = 0  # 立即复习一次，让第一次充分吸收

                    # check if this sample is scheduled for review now
                    if self.step_count >= mem['next_review']:
                        # 计算单样本蒸馏损失（MSE between student feature and teacher feature）
                        # 取每样本的 feature 向量/张量 slice
                        s_feat = s_primary[i:i+1]
                        t_feat = t_primary[i:i+1]

                        # 若形状不匹配，尝试 squeeze并broadcast
                        try:
                            distill_i = F.mse_loss(s_feat, t_feat, reduction='mean')
                        except Exception:
                            # fallback: scalar 0
                            distill_i = torch.tensor(0.0, device=self.device)

                        # memory factor: 高 strength -> less weight (因为掌握得好不需过多复习)
                        strength = float(mem['strength'])
                        memory_factor = (1.0 - strength) ** self.memory_power

                        # final per-sample weight = ebbinghaus_weight * memory_factor
                        weight_i = ebbinghaus_weight * memory_factor

                        # accumulate weighted distill loss
                        total_distill = total_distill + (weight_i * distill_i)
                        active_reviews += 1

                        # 更新记忆：用归一化 loss 更新 strength
                        # 注意：normalize_loss_for_memory 的输出越大表示“困难/忘记”，所以我们用 1 - norm 表示掌握度
                        norm_loss = self.normalize_loss_for_memory(distill_i.item())
                        # norm_loss approx in [0,1] with 0 indicating very easy (good), large -> >0
                        new_strength = self.update_memory_strength(sid, norm_loss)
                        mem['last_review'] = self.step_count
                        mem['next_review'] = self.step_count + self.compute_next_review_interval(new_strength)

                        # 记录历史（可选）
                        self.history[sid].append({
                            'step': self.step_count,
                            'distill_loss': float(distill_i.item()),
                            'weight': weight_i,
                            'strength': new_strength
                        })

                # 若有被复习的样本，则将 distill 加入总损失
                if active_reviews > 0:
                    # total_distill 已经是加权和；可以选择除以 active_reviews 做平均
                    avg_distill = total_distill / max(1, active_reviews)
                    trainer.loss = trainer.loss + avg_distill
                else:
                    # 没有样本需要复习 -> 不额外加损失
                    avg_distill = None

                # 打印日志（偶尔）
                if self.step_count % 100 == 0:
                    if avg_distill is not None:
                        logging.info(f" 步骤 {self.step_count}: Ebbinghaus base w={ebbinghaus_weight:.6f}, active_reviews={active_reviews}, avg_distill={avg_distill.item():.6f}")
                    else:
                        logging.info(f" 步骤 {self.step_count}: 无需复习样本 (active_reviews=0). Ebbinghaus w={ebbinghaus_weight:.6f}")
                    # 输出一些 sample 的 strength 快照
                    sample_snapshot = list(self.memory_store.items())[:3]
                    snap_str = ", ".join([f"{k}:{v['strength']:.3f}" for k, v in sample_snapshot])
                    logging.info(f"memory snapshot (first 3): {snap_str}")

            except Exception as e:
                if self.step_count % 200 == 0:
                    logging.warning(f"蒸馏回调错误: {e}")

        # 注册回调
        if not hasattr(self.student, 'callbacks'):
            self.student.callbacks = {}
        if 'on_train_batch_end' not in self.student.callbacks:
            self.student.callbacks['on_train_batch_end'] = []
        self.student.callbacks['on_train_batch_end'].append(on_train_batch_end)

    def train(self,  ** kwargs):
        """训练方法"""
        logging.info(" 开始简化蒸馏训练...")

        # 设置蒸馏回调
        self.setup_distillation_callback()

        # 训练配置
        config = {
            'data': self.data_config,
            'epochs': kwargs.get('epochs', 100),
            'imgsz': kwargs.get('imgsz', 640),
            'batch': kwargs.get('batch', 8),
            'device': self.device,
            'workers': kwargs.get('workers', 0),
            'lr0': kwargs.get('lr0', 0.001),
            'amp': kwargs.get('amp', False),
            'verbose': True,
        }

        results = self.student.train(**config)
        logging.info(f" 简化蒸馏训练完成，总步骤: {self.step_count}")
        return results


# ---------- main 保持不变 ----------
def main():
    """主函数"""
    try:
        trainer = SimpleDistillationTrainer(
            teacher_path="runs/segment/train2/weights/best.pt",
            student_config="yolo11n-seg.yaml",
            data_config="./datasets/crack-seg/data.yaml",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        results = trainer.train(
            epochs=100,
            imgsz=640,
            batch=8,
            workers=0,
            lr0=0.001,
            amp=False
        )

        logging.info(" 蒸馏训练完成!")
        return results

    except Exception as e:
        logging.error(f" 训练失败: {e}")
        return None


if __name__ == '__main__':
    main()
