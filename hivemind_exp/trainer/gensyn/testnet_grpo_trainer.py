from typing import Sequence

from hivemind_exp.chain_utils import SwarmCoordinator
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer


class TestnetGRPOTrainer(HivemindGRPOTrainer):
    """测试网GRPO训练器，用于在测试网上进行GRPO训练"""
    def __init__(self, coordinator: SwarmCoordinator, **kwargs) -> None:
        """初始化测试网GRPO训练器
        
        Args:
            coordinator: 蜂群协调器实例
            **kwargs: 其他参数
        """
        self.coordinator = coordinator
        super().__init__(**kwargs)

    def submit_winners(self, round_num: int, winners: Sequence[str]):
        """提交轮次获胜者
        
        Args:
            round_num: 轮次编号
            winners: 获胜者列表
        """
        self.logger.info(f"🏆 正在提交轮次 {round_num} 的获胜者: {winners}")
        self.coordinator.submit_winners(round_num, winners[:1])

    def get_round_and_stage(self):
        """获取当前轮次和阶段
        
        Returns:
            当前轮次和阶段的元组
        """
        return self.coordinator.get_round_and_stage()

    def train_stages(self, round_num, start_stage, is_coordinator):
        """训练多个阶段并提交获胜者
        
        Args:
            round_num: 轮次编号
            start_stage: 开始阶段
            is_coordinator: 是否为协调器
        """
        super().train_stages(round_num, start_stage, is_coordinator)
        self.submit_winners(round_num, self.stage_data.round_winner_fn())

    def _train(self):
        """训练入口方法，使用跟随者训练模式"""
        self.follower_train()
