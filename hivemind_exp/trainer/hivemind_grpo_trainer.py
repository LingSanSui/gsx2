import gc
import hashlib
import logging
import time
import traceback
from typing import Any

import datasets
import torch
from hivemind.dht import DHT
from hivemind.utils import get_dht_time
from trl import GRPOConfig, GRPOTrainer

from hivemind_exp.debug_utils import print_system_info
from hivemind_exp.dht_utils import (
    ROUND_STAGE_NUMBER_KEY,
    get_dht_value,
    get_round_and_stage,
    leaderboard_key,
    node_outputs_key,
    rewards_key,
)
from hivemind_exp.hivemind_utils import HivemindNode, StageData
from hivemind_exp.name_utils import get_name_from_peer_id


MAX_TRAIN_FAILS = 5
CADENCE_OF_UPDATE_STEPS = 4


class HivemindGRPOTrainer:
    """
    GRPOTrainer的子类，通过将中间结果发布到连接的Hivemind DHT来实现多阶段GRPO训练。
    该类负责协调多个节点之间的训练过程，包括协调者节点和跟随者节点的不同行为。
    """

    class PublishingGRPOTrainer(GRPOTrainer):
        """
        内部GRPOTrainer子类，负责将训练结果发布到DHT网络。
        该类扩展了标准GRPOTrainer，添加了发布奖励、输出和排行榜到分布式哈希表的功能。
        """
        def __init__(
            self,
            node: HivemindNode,  # Hivemind节点实例
            dht: DHT,            # 分布式哈希表实例
            tokenizer,           # 分词器
            logger,              # 日志记录器
            **kwargs,            # 其他参数传递给父类
        ):
            """
            初始化PublishingGRPOTrainer
            
            参数:
                node: Hivemind节点实例，包含节点身份和状态信息
                dht: 分布式哈希表实例，用于存储和检索分布式训练数据
                tokenizer: 用于处理文本的分词器
                logger: 日志记录器
                **kwargs: 传递给父类GRPOTrainer的其他参数
            """
            self.node = node
            self.dht = dht
            self.logger = logger
            self.stage_rewards = 300.0  # 阶段奖励累计值
            super().__init__(processing_class=tokenizer, **kwargs)

        def publish_leaderboard(self):
            """
            发布当前轮次和阶段的排行榜到DHT
            根据所有节点的奖励值创建排序后的排行榜并存储到分布式哈希表中
            """
            r, s = self.node.round_num, self.node.stage_num
            curr_rewards: dict[str, Any] | None = get_dht_value(
                self.dht, key=rewards_key(r, s), latest=True
            )
            if curr_rewards:
                # 创建(节点键, 奖励值)对的排序列表
                leaderboard = list(
                    sorted(
                        curr_rewards.items(), key=lambda t: (t[1], t[0]), reverse=True
                    )
                )
                self.dht.store(
                    key=leaderboard_key(r, s),
                    value=leaderboard,
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )
            else:
                self.logger.info(f"无法获取轮次 {r} 阶段 {s - 1} 的奖励值")

        def compute_loss(self, model, inputs, *args, **kwargs):
            """
            计算模型损失并定期将节点输出和奖励发布到DHT
            
            
            参数:
                model: 要训练的模型
                inputs: 模型输入数据
                *args, **kwargs: 传递给父类compute_loss方法的其他参数
                
            返回:
                计算得到的损失值
            """
            loss = super().compute_loss(model, inputs, *args, **kwargs)
            # 奖励函数必须保存node.outputs和node.rewards!
            # 这里的代码负责在适当的时间将数据发布到DHT
            # 每N步发布一次数据到DHT
            self.logger.info(
                f"  ✅✅✅✅✅✅------✅✅✅✅✅ "
            )
            if self.state.global_step % CADENCE_OF_UPDATE_STEPS == 0:
                question = self.node.outputs["question"]
                q_hash = hashlib.md5(question.encode()).hexdigest()

                value = (time.time(), self.node.outputs)
                self.logger.info(
                    f"  --->>   key值为             {node_outputs_key(self.node)}"
                )
                self.logger.info(
                    f"  --->>   subkey值为          {q_hash}"
                )
                self.logger.info(
                    f"  --->>   value值为           {value}"
                )
                self.logger.info(
                    f"  --->>   expiration_time值为 {get_dht_time() + self.node.out_expiration}"
                )
                self.dht.store(
                    key=node_outputs_key(self.node),
                    subkey=q_hash,
                    value=value,
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )
                self.node.put_stage_outputs(
                    self.node.round_num, self.node.stage_num, q_hash, value
                )

                # 累加最新的奖励值
                self.stage_rewards += sum(self.node.rewards)
                
                self.logger.info(
                    f"  --->>   key值为             {rewards_key(self.node.round_num, self.node.stage_num)}"
                )
                self.logger.info(
                    f"  --->>   subkey值为          {self.node.key}"
                )
                self.logger.info(
                    f"  --->>   value值为            {self.stage_rewards}"
                )
                self.logger.info(
                    f"  --->>   expiration_time值为 {get_dht_time() + self.node.out_expiration}"
                )
                self.logger.info(
                    f"  ✅✅✅✅✅✅------✅✅✅✅✅ "
                )
                self.dht.store(
                    key=rewards_key(self.node.round_num, self.node.stage_num),
                    subkey=self.node.key,
                    value=self.stage_rewards,
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )
            if self.node.is_coordinator:
                self.publish_leaderboard()

            return loss

    def __init__(
        self,
        node: HivemindNode,      # Hivemind节点实例
        dht: DHT,                # 分布式哈希表实例
        stage_data: StageData,   # 训练阶段数据
        config: GRPOConfig,      # GRPO配置
        model,                   # 要训练的模型
        tokenizer,               # 分词器
        log_tag=None,            # 日志标签
        **kwargs,                # 其他参数
    ):
        """
        初始化HivemindGRPOTrainer
        
        
        参数:
            node: Hivemind节点实例，定义节点身份和角色（协调者或跟随者）
            dht: 分布式哈希表实例，用于节点间通信和数据共享
            stage_data: 包含训练阶段信息的StageData实例
            config: GRPO训练配置
            model: 要训练的模型
            tokenizer: 用于处理文本的分词器
            log_tag: 可选的日志标签，默认使用节点键
            **kwargs: 其他参数
        """
        # 单个协调者负责递增轮次和阶段编号
        # TODO(lou): 允许选择不同的协调者？
        self.node = node
        self.dht = dht

        self.stage_data = stage_data

        self.config = config
        self.config.dataloader_num_workers=0  # 默认值: 8+
        assert self.config.output_dir
        self.config.output_dir += f"-{get_name_from_peer_id(self.node.key, True)}"  # TODO: 在更合适的位置添加动物名称到保存路径
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not log_tag:
            log_tag = self.node.key

        self.logger = logging.getLogger(f"{__name__}:{log_tag}")

    def wait_for(self, result_fn=lambda: None, interval=10, timeout=30):
        """
        等待函数返回非None结果或超时
        
        
        参数:
            result_fn: 要执行的函数，应返回结果或None
            interval: 重试间隔时间（秒）
            timeout: 最大等待时间（秒）
            
        返回:
            函数的结果，如果超时可能为None
        """
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            result = result_fn()
            if result is None:
                time.sleep(interval)
            else:
                break

        return result

    def train_stages(self, round_num, start_stage, is_coordinator):
        """
        训练指定轮次的多个阶段
        
        
        参数:
            round_num: 当前训练轮次
            start_stage: 开始训练的阶段索引
            is_coordinator: 是否为协调者节点
        """
        # TODO: 需要添加检查点加载功能
        self.node.round_num = round_num
        for i, stage in enumerate(self.stage_data.stages[start_stage:]):
            stage_num = start_stage + i
            self.node.stage_num = stage_num

            if is_coordinator:
                self.dht.store(
                    key=ROUND_STAGE_NUMBER_KEY,
                    value=(self.node.round_num, stage_num),
                    expiration_time=get_dht_time() + self.node.out_expiration,
                )

            self.logger.info(f"📈 训练轮次: {round_num} 阶段: {stage_num}")
            train_dataset, test_dataset = stage.datasets_fn(round_num, stage_num)
            kwargs = {
                "model": self.model,
                "args": self.config,
                "reward_funcs": stage.reward_funcs,
                "train_dataset": train_dataset,
                "eval_dataset": test_dataset,
            }
            trainer = HivemindGRPOTrainer.PublishingGRPOTrainer(
                self.node, self.dht, self.tokenizer, self.logger, **kwargs
            )
            self.train_and_save(trainer, train_dataset)
            self.logger.info(
                f"📉 完成训练轮次: {round_num} 阶段: {stage_num}"
            )

        # 如果需要，推送模型到HF hub
        # TODO: 添加额外的逻辑检查是否提供了访问令牌和HF用户名
        if self.config.push_to_hub_token is not None:
            self.logger.info("正在推送模型到Hugging Face Hub...")
            try:
                trainer.push_to_hub(
                    tags=[
                        "rl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {get_name_from_peer_id(self.node.key)}",
                    ]
                )
                time.sleep(1)
            except Exception:
                self.logger.info(
                    "推送模型到Hugging Face Hub失败。当您完成训练后，请尝试按照以下说明手动推送：https://huggingface.co/docs/hub/en/models-uploading"
                )

        self.cleanup()

    def cleanup(self):
        """
        清理各种缓存，释放内存资源
        包括垃圾回收、GPU缓存清理和节点阶段缓存清理
        """
        # 清理各种阶段缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available():  # type: ignore
            torch.mps.empty_cache()  # type: ignore
        try:
            if torch.xpu.is_available():  # type: ignore
                torch.xpu.empty_cache()  # type: ignore
        except AttributeError:
            pass

        self.node.clear_stage_cache()

    def train_and_save(self, trainer, train_dataset):
        """
        执行训练并保存模型和指标
        
        
        参数:
            trainer: 训练器实例
            train_dataset: 训练数据集
        """
        for num_fails in range(MAX_TRAIN_FAILS):
            try:
                train_result = trainer.train()
                break
            except (BlockingIOError, EOFError) as e:
                self.logger.warning(f"DHT IPC错误: {e}. 重新开始训练...")
                self.cleanup()  # 清理GPU/缓存
                time.sleep(5)
                continue

        # 记录并保存指标
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        self.logger.info("正在保存模型")
        trainer.model.config.use_cache = True
        trainer.save_model(self.config.output_dir)
        self.logger.info(f"模型已保存到 {self.config.output_dir}")
        assert self.config.distributed_state
        self.config.distributed_state.wait_for_everyone()  # 等待所有进程加载完成

        self.tokenizer.save_pretrained(self.config.output_dir)
        self.logger.info(f"分词器已保存到 {self.config.output_dir}")

    def get_round_and_stage(self):
        """
        从DHT获取当前轮次和阶段
        
        返回:
            当前轮次和阶段的元组 (round_num, stage_num)
        """
        return get_round_and_stage(self.dht)

    def coordinator_train(self):
        """
        协调者节点的训练方法
        负责启动新的训练轮次并更新DHT中的轮次和阶段信息
        """
        round_num = 0
        start_time = time.monotonic()
        while (
            round_num < self.stage_data.max_rounds
            and time.monotonic() - start_time < self.stage_data.train_timeout
        ):
            self.logger.info(f"🤖 开始新轮次: {round_num}")

            _ = self.dht.get_visible_maddrs(latest=True)
            self.train_stages(round_num, 0, is_coordinator=True)

            round_num += 1
            if round_num == self.stage_data.max_rounds:
                return

        self.logger.info("训练超时！")

    def follower_train(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=30.0
    ):
        """
        跟随者节点的训练方法
        
        定期检查DHT中的轮次和阶段信息，并加入当前活跃的训练轮次
        
        参数:
            check_interval: 检查DHT的初始间隔时间（秒）
            log_timeout: 日志记录超时时间（秒）
            max_check_interval: 最大检查间隔时间（秒）
        """
        done_rounds = set()
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = (
            check_interval  # 对已完成轮次使用指数退避策略
        )
        while time.monotonic() - start_time < self.stage_data.train_timeout:
            curr_time = time.monotonic()
            _ = self.dht.get_visible_maddrs(latest=True)

            # 获取当前轮次和阶段
            try:
                round_num, stage = self.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    self.logger.debug(
                        f"无法获取轮次和阶段信息: {e}. 将在 {check_interval}秒后重试。"
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num not in done_rounds:
                self.logger.info(
                    f"🐝 加入轮次: {round_num} 从阶段: {stage} 开始"
                )
                try:
                    self.train_stages(round_num, stage, is_coordinator=False)
                except datasets.exceptions.DatasetGenerationError:
                    if stage > 0:
                        self.logger.info("尝试从阶段0重新开始训练！")

                        # 从阶段0重新开始
                        self.train_stages(round_num, 0, is_coordinator=False)
                    else:
                        raise

                done_rounds.add(round_num)
                check_backoff = check_interval  # 成功轮次后重置退避
            else:
                if check_backoff != 30:
                    self.logger.info(
                        f":{self.node.key}:已完成训练轮次: {round_num}。将在 {check_backoff}秒 后重新检查是否有新任务，日志暂停刷新，不是卡住，耐心等待。"
                    )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.stage_data.max_rounds - 1:
                return

        self.logger.info("训练超时！")

    def _train(self):
        """训练入口方法，根据节点角色选择适当的训练方法"""
        if self.node.is_coordinator:
            self.coordinator_train()
        else:
            self.follower_train()

    def train(self):
        """训练方法，捕获并处理训练过程中的异常"""
        try:
            self._train()

        except Exception:
            self.logger.error("训练过程中遇到错误！")
            print_system_info()
            traceback.print_exc()
            raise
