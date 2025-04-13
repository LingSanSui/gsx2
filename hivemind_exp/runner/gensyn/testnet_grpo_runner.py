import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import hivemind
from datasets import Dataset
from trl import GRPOConfig, ModelConfig

from hivemind_exp.chain_utils import (
    SwarmCoordinator,
)
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner
from hivemind_exp.trainer.gensyn.testnet_grpo_trainer import TestnetGRPOTrainer

# 创建日志记录器
logger = logging.getLogger(__name__)


@dataclass
class TestnetGRPOArguments:
    # 互斥参数
    wallet_private_key: str | None = None  # EOA钱包私钥
    modal_org_id: str | None = None  # Modal组织ID

class TestnetGRPORunner(GRPORunner):
    """测试网GRPO运行器，用于在测试网上运行GRPO训练"""
    def __init__(self, coordinator: SwarmCoordinator) -> None:
        """初始化测试网GRPO运行器
        
        Args:
            coordinator: 蜂群协调器实例
        """
        self.coordinator = coordinator

    def get_initial_peers(self) -> list[str]:
        """获取初始对等节点列表
        
        Returns:
            初始对等节点列表
        """
        return self.coordinator.get_bootnodes()

    def register_peer(self, peer_id):
        """注册对等节点
        
        Args:
            peer_id: 对等节点ID
        """
        logger.info(f"正在注册自身，对等节点ID: {peer_id}")
        self.coordinator.register_peer(peer_id)

    def setup_dht(self, grpo_args):
        """设置分布式哈希表
        
        Args:
            grpo_args: GRPO参数
            
        Returns:
            初始化的DHT实例
        """
        initial_peers = grpo_args.initial_peers

        dht = hivemind.DHT(start=True, **self._dht_kwargs(grpo_args))
        logger.info(f"🐝 正在加入蜂群，初始对等节点 = {initial_peers}")

        peer_id = str(dht.peer_id)
        self.name = self._get_animal_name(peer_id)
        self.register_peer(peer_id)
        return dht

    def run(
        self,
        model_args: ModelConfig,
        grpo_args: GRPOArguments,
        training_args: GRPOConfig,
        initial_datasets_fn: Callable[[], Tuple[Dataset, Dataset]],
    ):
        """运行测试网GRPO训练过程
        
        Args:
            model_args: 模型配置参数
            grpo_args: GRPO参数
            training_args: 训练配置参数
            initial_datasets_fn: 获取初始数据集的函数
        """
        initial_peers = grpo_args.initial_peers
        if not initial_peers:
            initial_peers = self.get_initial_peers()
            logger.info(f"从链上获取初始对等节点: {initial_peers}")
        elif initial_peers == ["BOOT"]:
            initial_peers = []
            logger.info("作为引导节点继续！")

        grpo_args.initial_peers = initial_peers
        super().run(
            model_args,
            grpo_args,
            training_args,
            initial_datasets_fn,
            partial(
                TestnetGRPOTrainer,
                coordinator=self.coordinator
            ),
        )
