from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

# 这个模块定义了Hivemind节点和相关数据结构，用于分布式强化学习训练
# 它提供了节点间通信和数据共享的基础设施，支持多阶段训练过程


@dataclass
class HivemindNode:
    """Hivemind节点类，表示分布式训练中的一个参与节点
    
    这个类封装了分布式训练网络中单个节点的所有状态和功能。
    每个节点可以是普通参与者或协调器，负责处理特定轮次和阶段的训练数据。
    
    属性:
        model_name: 模型名称，标识节点使用的模型类型
        key: 节点唯一标识符，设置为DHT PeerID
        is_coordinator: 是否为协调器节点，协调器负责管理训练进度
        outputs: 最后一次训练步骤的问答输出
        round_cache: 轮次和阶段的缓存，格式为(r, s): Q: (timestamp, outputs)
        rewards: 最后一次训练的奖励输出
        round_num: 当前轮次编号（由协调器递增）
        stage_num: 当前阶段编号（由协调器递增）
        out_expiration: 输出过期时间（小时）
    """
    # 节点元数据
    model_name: str
    key: str  # 设置为DHT PeerID

    is_coordinator: bool = False

    # 最后一次训练步骤的问答输出
    outputs: dict[Any, Any] = field(default_factory=dict)
    # 轮次和阶段的缓存，格式为(r, s): Q: (timestamp, outputs)
    round_cache: dict[tuple[int, int], dict[str, tuple[float, dict]]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    # 最后一次训练的奖励输出
    rewards: Sequence[float | int] = field(default_factory=list)

    # 由协调器递增的值
    round_num: int = 0
    stage_num: int = 0

    out_expiration: int = 60 * 60 * 4  # 4小时，以秒为单位

    @staticmethod
    def coordinator(*args, **kwargs):
        """创建一个协调器节点
        
        这是一个便捷方法，用于创建具有协调器角色的节点实例。
        协调器节点负责管理训练进度和同步其他节点。
        
        Args:
            *args: 传递给HivemindNode构造函数的位置参数
            **kwargs: 传递给HivemindNode构造函数的关键字参数
            
        Returns:
            设置is_coordinator=True的HivemindNode实例
        """
        return HivemindNode(*args, **kwargs, is_coordinator=True)

    def get_stage_outputs(self, r, s) -> dict[str, tuple[float, dict]] | None:
        """获取特定轮次和阶段的输出
        
        从节点的缓存中检索特定轮次和阶段的所有输出数据。
        
        Args:
            r: 轮次编号
            s: 阶段编号
            
        Returns:
            该轮次和阶段的缓存输出字典，如果不存在则返回None
        """
        key = (r, s)
        if key in self.round_cache:
            return self.round_cache[key]

    def put_stage_outputs(self, r, s, question, value: tuple[float, dict]):
        """存储特定轮次和阶段的输出
        
        将训练输出数据存储到节点的轮次缓存中，用于后续处理或共享。
        
        Args:
            r: 轮次编号
            s: 阶段编号
            question: 问题标识符
            value: 要存储的值(时间戳, 输出)元组
        """
        self.round_cache[(r, s)][question] = value

    def clear_stage_cache(self):
        """清除所有阶段缓存
        
        重置节点的轮次缓存，通常在开始新的训练周期时使用。
        """
        self.round_cache.clear()


# 类型定义
# 接收轮次和阶段作为参数
DatasetsFn = Callable[
    [int, int], tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
]  # 数据集函数类型，返回训练和测试数据集的元组

MergeFn = Callable[[list], dict[str, dict]]  # 合并函数类型，用于合并多个输出
LossFn = Callable[[list], dict[str, float]]  # 损失函数类型，计算损失值


@dataclass
class SingleStageData:
    """单阶段数据类，表示训练过程中的一个阶段
    
    封装了训练过程中单个阶段的所有相关数据和函数。
    每个阶段可以有不同的奖励函数和数据集。
    
    属性:
        name: 阶段名称，用于标识和日志记录
        reward_funcs: 奖励函数列表，用于评估模型输出
        datasets_fn: 数据集函数，用于获取训练和测试数据集
    """
    name: str
    reward_funcs: list[Callable]
    datasets_fn: DatasetsFn  # 用于获取训练/测试数据集


@dataclass
class StageData:
    """阶段数据类，表示整个训练过程的多个阶段
    
    管理整个训练过程中的所有阶段，包括阶段序列和全局配置。
    定义了训练的整体结构和时间限制。
    
    属性:
        stages: 单阶段数据序列，按顺序执行
        round_winner_fn: 轮次获胜者选择函数，用于确定每轮的最佳模型
        max_rounds: 最大轮次数，默认为100
        train_timeout: 训练超时时间（秒），默认为4天
        round_timeout: 轮次超时时间（秒），默认为4小时
    """
    stages: Sequence[SingleStageData]
    round_winner_fn: Callable

    max_rounds: int = 100
    train_timeout: int = 60 * 60 * 24 * 4  # 4天，以秒为单位
    round_timeout: int = 60 * 60 * 4  # 4小时，以秒为单位

    def __len__(self):
        """返回阶段数量
        
        Returns:
            训练过程中的总阶段数
        """
        return len(self.stages)

