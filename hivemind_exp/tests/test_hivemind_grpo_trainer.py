import itertools
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hivemind
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from hivemind_exp.dht_utils import (
    HivemindNode,
    leaderboard_key,
    outputs_key,
    rewards_key,
)
from hivemind_exp.hivemind_utils import SingleStageData, StageData
from hivemind_exp.tests.fake_data import CK, QUESTION, QUESTION_HASH, RSK, SAMPLES
from hivemind_exp.trainer.hivemind_grpo_trainer import (
    HivemindGRPOTrainer,
    get_dht_value,
)

# 本测试文件用于测试HivemindGRPOTrainer的功能，包括单节点和多节点场景下的训练过程、
# 数据同步、奖励计算和排行榜生成等功能。


def dummy_reward_func(node: HivemindNode, prompts, completions, **kwargs) -> list[int]:
    """测试用的奖励函数
    
    为测试提供一个简单的奖励计算逻辑：协调者节点获得2分，其他节点获得1分。
    同时设置节点的outputs属性，模拟实际训练中的输出。
    
    参数:
        node: Hivemind节点实例
        prompts: 提示信息列表
        completions: 模型生成的完成内容
        **kwargs: 其他参数
        
    返回:
        奖励值列表
    """
    # 设置节点输出为问题内容
    node.outputs = {"question": prompts[0][-1]["content"]}
    # 根据节点角色分配奖励
    if node.is_coordinator:
        rewards = [2]  # 协调者获得2分
    else:
        rewards = [1]  # 其他节点获得1分

    # 保存奖励到节点
    node.rewards = rewards
    return rewards


# 测试用的小型模型名称
TEST_MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def get_model_config(tmp_path, max_steps):
    """获取模型和训练配置
    
    加载测试用的小型模型和GRPO训练配置。
    
    参数:
        tmp_path: 临时路径，用于保存模型输出
        max_steps: 最大训练步数
        
    返回:
        模型和配置的元组
    """
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)
    # 创建GRPO训练配置
    config = GRPOConfig(
        output_dir=tmp_path,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        max_steps=max_steps,
    )
    return model, config


def create_dht_and_trainer(tmp_path, node, stage_data, max_steps=1, initial_peers=[]):
    """创建DHT和训练器实例
    
    初始化分布式哈希表和HivemindGRPOTrainer实例，用于测试。
    
    参数:
        tmp_path: 临时路径，用于保存模型输出
        node: Hivemind节点实例
        stage_data: 训练阶段数据
        max_steps: 最大训练步数，默认为1
        initial_peers: 初始对等节点列表，默认为空
        
    返回:
        DHT和训练器实例的元组
    """
    # 创建分布式哈希表
    dht = hivemind.DHT(start=True, initial_peers=initial_peers, cache_nearest=2)
    # 获取模型和配置
    model, config = get_model_config(tmp_path, max_steps=max_steps)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    # 创建训练器
    trainer = HivemindGRPOTrainer(
        dht=dht,
        node=node,
        model=model,
        tokenizer=tokenizer,
        config=config,
        stage_data=stage_data,
    )
    return dht, trainer




###############
# 单节点测试 #
###############

def test_single_node_crash(tmp_path):
    """测试单节点崩溃情况
    
    验证当datasets_fn函数抛出异常时，训练器能否正确处理错误。
    """
    # 创建协调者节点
    node = HivemindNode.coordinator("test", CK)

    # 空奖励函数
    def reward_func(**kwargs):
        return []

    # 会抛出异常的数据集函数
    def error_fn (r, s):
        raise ValueError("error")

    # 创建训练器
    _, trainer = create_dht_and_trainer(
        tmp_path,
        node,
        StageData(
            max_rounds=1,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn= error_fn,
                ),
            ],
        ),
    )
    # 验证训练过程中是否正确抛出异常
    with pytest.raises(ValueError, match='error'):
        trainer.train()


def test_single_node_single_stage(tmp_path):
    """测试单节点单阶段训练
    
    验证单个节点在单阶段训练中是否能正常完成训练过程。
    """
    # 创建协调者节点
    node = HivemindNode.coordinator("test", CK)

    # 创建奖励函数
    def reward_func(**kwargs):
        return dummy_reward_func(node, **kwargs)

    # 创建DHT和训练器
    dht, trainer = create_dht_and_trainer(
        tmp_path,
        node,
        StageData(
            max_rounds=1,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
            ],
        ),
    )
    # 执行训练
    trainer.train()


def test_single_node_multi_stage(tmp_path):
    """测试单节点多阶段训练
    
    冒烟测试：不实际合并数据，只标记完成情况。
    验证单个节点在多阶段训练中是否能正确处理阶段转换。
    """
    # 用于记录阶段完成情况的字典
    completions = {}

    # 第一阶段数据集函数，标记阶段完成
    def datasets_one(r, s):
        completions["merged_0"] = True
        return SAMPLES, SAMPLES

    # 创建协调者节点
    node = HivemindNode.coordinator("test", CK)

    # 创建奖励函数
    def reward_func(**kwargs):
        return dummy_reward_func(node, **kwargs)

    # 创建DHT和训练器，包含两个训练阶段
    dht, trainer = create_dht_and_trainer(
        tmp_path,
        node,
        StageData(
            max_rounds=1,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
                SingleStageData(
                    name="1",
                    reward_funcs=[reward_func],
                    datasets_fn=datasets_one,  # type: ignore
                ),
            ],
        ),
    )
    # 执行训练
    trainer.train()

    # 验证第一阶段是否已完成
    assert completions == {"merged_0": True}


##############
# 多节点测试 #
##############

# 以下测试将实际检查DHT输出、奖励和排行榜数据。

# TODO: 修复以下测试的不稳定性问题。

def test_multi_node_single_stage(tmp_path):
    """测试多节点单阶段训练
    
    验证多个节点在单阶段训练中能否正确同步数据、计算奖励和生成排行榜。
    测试包括协调者节点和普通节点的交互。
    """
    # 设置测试参数
    max_rounds = 1
    max_steps = 2

    def create_stage_data(node):
        """为节点创建阶段数据
        
        根据节点创建包含奖励函数的阶段数据。
        
        参数:
            node: Hivemind节点实例
            
        返回:
            配置好的StageData实例
        """
        def reward_func(**kwargs):
            return dummy_reward_func(node, **kwargs)

        return StageData(
            max_rounds=max_rounds,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
            ],
        )

    # 创建协调者节点和普通节点
    node0 = HivemindNode.coordinator("test", CK)
    node1 = HivemindNode("test", "0")

    # 为协调者节点创建DHT和训练器
    dht0, trainer0 = create_dht_and_trainer(
        Path(tmp_path) / "0", node0, create_stage_data(node0), max_steps
    )
    # 为普通节点创建DHT和训练器，连接到协调者节点的DHT
    dht1, trainer1 = create_dht_and_trainer(
        Path(tmp_path) / "1",
        node1,
        create_stage_data(node1),
        max_steps,
        dht0.get_visible_maddrs(),
    )
    # 使用线程池并行执行两个节点的训练
    with ThreadPoolExecutor() as executor:
        for trainer in (trainer0, trainer1):
            executor.submit(trainer.train)

    # 验证当前轮次和阶段
    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (max_rounds - 1, 0)

    # 验证每个轮次和阶段的输出、奖励和排行榜
    for r, s in itertools.product([0], [0]):
        # 验证节点输出
        outputs = get_dht_value(dht0, key=outputs_key(node0.key, r, s), latest=True)
        assert outputs
        assert outputs[QUESTION_HASH][1] == {"question": QUESTION}

        # 验证奖励计算
        rewards = get_dht_value(dht0, key=rewards_key(r, s), latest=True)
        assert rewards
        assert len(rewards) == 2
        assert math.isclose(rewards[CK], 2.0 * max_steps)  # 协调者获得2分/步
        assert math.isclose(rewards[node1.key], max_steps)  # 普通节点获得1分/步

        # 验证排行榜生成
        leaderboard = get_dht_value(dht0, key=leaderboard_key(r, s), latest=True)
        assert leaderboard
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == CK  # 协调者应排在第一位
        assert math.isclose(leaderboard[0][1], 2.0 * max_steps)


def test_multi_node_multi_stage(tmp_path):
    """测试多节点多阶段训练
    
    冒烟测试：不实际合并数据，只标记完成情况。
    验证多个节点在多阶段训练中能否正确处理阶段转换、同步数据和计算奖励。
    """
    # 用于记录阶段完成情况的计数器字典
    completions = defaultdict(int)
    max_rounds = 2
    max_steps = 2

    # 第一阶段数据集函数，记录阶段完成次数
    def datasets_one(r, s):
        completions["merged_0"] += 1
        return SAMPLES, SAMPLES

    # 第二阶段数据集函数，记录阶段完成次数
    def datasets_two(r, s):
        completions["merged_1"] += 1
        return SAMPLES, SAMPLES

    def create_stage_data(node):
        """为节点创建多阶段训练数据
        
        创建包含三个阶段的训练数据配置。
        
        参数:
            node: Hivemind节点实例
            
        返回:
            配置好的StageData实例
        """
        def reward_func(**kwargs):
            return dummy_reward_func(node, **kwargs)

        return StageData(
            max_rounds=max_rounds,
            round_winner_fn=lambda:[CK],
            stages=[
                SingleStageData(
                    name="0",
                    reward_funcs=[reward_func],
                    datasets_fn=lambda r, s: (SAMPLES, SAMPLES),  # type: ignore
                ),
                SingleStageData(
                    name="1",
                    reward_funcs=[reward_func],
                    datasets_fn=datasets_one,  # type: ignore
                ),
                SingleStageData(
                    name="2",
                    reward_funcs=[reward_func],
                    datasets_fn=datasets_two,  # type: ignore
                ),
            ],
        )

    # 创建协调者节点和普通节点
    node0 = HivemindNode.coordinator("test", CK)
    node1 = HivemindNode("test", "0")

    # 为协调者节点创建DHT和训练器
    dht0, trainer0 = create_dht_and_trainer(
        Path(tmp_path) / "0", node0, create_stage_data(node0), max_steps
    )
    # 为普通节点创建DHT和训练器，连接到协调者节点的DHT
    dht1, trainer1 = create_dht_and_trainer(
        Path(tmp_path) / "1",
        node1,
        create_stage_data(node1),
        max_steps,
        dht0.get_visible_maddrs(),
    )
    # 使用线程池并行执行两个节点的训练
    with ThreadPoolExecutor() as executor:
        for trainer in (trainer0, trainer1):
            executor.submit(trainer.train)

    # 验证最终轮次和阶段
    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (max_rounds - 1, 2)  # 应该完成到最后一轮的第三阶段

    # 验证每个阶段的合并次数
    # 每个阶段应该被每个节点合并max_rounds次，总共2*max_rounds次
    assert completions == {
        "merged_0": max_rounds * 2,  # 第一阶段合并次数
        "merged_1": max_rounds * 2,  # 第二阶段合并次数
    }

    # 验证每个轮次和阶段的输出、奖励和排行榜
    for r, s in itertools.product(range(1), range(3)):
        # 验证节点输出
        outputs = get_dht_value(dht0, key=outputs_key(node0.key, r, s), latest=False)
        assert outputs
        assert outputs[QUESTION_HASH][1] == {"question": QUESTION}

        # 验证奖励计算
        rewards = get_dht_value(dht0, key=rewards_key(r, s), latest=False)
        assert rewards
        assert len(rewards) == 2
        assert math.isclose(rewards[CK], 2.0 * max_steps)  # 协调者获得2分/步
        assert math.isclose(rewards[node1.key], max_steps)  # 普通节点获得1分/步

        # 验证排行榜生成
        leaderboard = get_dht_value(dht0, key=leaderboard_key(r, s), latest=False)
        assert leaderboard
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == CK  # 协调者应排在第一位
        assert math.isclose(leaderboard[0][1], 2.0 * max_steps)
