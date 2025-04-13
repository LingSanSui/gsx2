import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import hivemind
import pytest
from datasets import Dataset
from hivemind.dht import DHT
from hivemind.utils import get_dht_time
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from hivemind_exp.dht_utils import ROUND_STAGE_NUMBER_KEY, outputs_key
from hivemind_exp.gsm8k.stage_utils import (
    HivemindNode,
    get_stage2_samples,
    get_stage3_samples,
    gsm8k_stage_data,
    merge_stage1_question,
    merge_stage2_question,
    merged_prev_stage_datasets,
    rewards_key,
)
from hivemind_exp.hivemind_utils import SingleStageData
from hivemind_exp.tests.fake_data import (
    CK,
    QUESTION,
    RSK,
    SAMPLES,
    STAGE_2_MERGED,
    STAGE_2_OUTPUTS,
    samples_with_key,
)
from hivemind_exp.trainer.hivemind_grpo_trainer import (
    HivemindGRPOTrainer,
    get_dht_value,
)

# 测试使用的模型名称
TEST_MODEL_NAME = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def get_model_config(tmp_path):
    """获取测试用的模型和配置
    
    Args:
        tmp_path: 临时路径，用于存储模型输出
        
    Returns:
        model: 加载的模型实例
        config: GRPO配置实例
    """
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME)
    config = GRPOConfig(
        output_dir=tmp_path,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        max_steps=1,
    )
    return model, config


def wrap_datasets_fn(stage: SingleStageData, check):
    """包装数据集函数，添加检查逻辑
    
    Args:
        stage: 单阶段数据实例
        check: 检查函数，用于验证数据集
    """
    orig = stage.datasets_fn

    def wrapped(r, s):
        value = orig(r, s)
        check(value[0])
        return value

    stage.datasets_fn = wrapped


def check_dataset(prefix: str, min_count: int, dataset: Dataset):
    """检查数据集中特定前缀的特征数量
    
    Args:
        prefix: 特征前缀
        min_count: 最小特征数量
        dataset: 要检查的数据集
    """
    c = 0
    for feature in dataset.features:
        if feature.startswith(prefix):
            c += 1
    assert c >= min_count


def create_dht_and_trainer(tmp_path, node, min_peers=1, initial_peers=[]):
    """创建DHT和训练器实例
    
    Args:
        tmp_path: 临时路径
        node: Hivemind节点
        min_peers: 最小对等节点数量
        initial_peers: 初始对等节点列表
        
    Returns:
        dht: 分布式哈希表实例
        trainer: Hivemind GRPO训练器实例
    """
    dht = hivemind.DHT(start=True, initial_peers=initial_peers, cache_nearest=min_peers)
    model, config = get_model_config(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

    # 始终检查阶段合并

    def check_merged_stage1_dataset(dataset: Dataset):
        # print(f"Merged stage 1 for: {node.key}", dataset)
        check_dataset("agent_answers", min_peers, dataset)

    def check_merged_stage2_dataset(dataset: Dataset):
        # print(f"Merged stage 2 for: {node.key}", dataset)
        check_dataset("agent_opinion", min_peers, dataset)

    stage_data = gsm8k_stage_data(dht, node, SAMPLES, SAMPLES, check_interval=1)
    stage_data.max_rounds = 1
    stage_data.stages[0].datasets_fn = lambda r, s: (SAMPLES, SAMPLES)  # type: ignore
    wrap_datasets_fn(stage_data.stages[1], check_merged_stage1_dataset)
    wrap_datasets_fn(stage_data.stages[2], check_merged_stage2_dataset)

    trainer = HivemindGRPOTrainer(
        dht=dht,
        node=node,
        model=model,
        tokenizer=tokenizer,
        config=config,
        stage_data=stage_data,
    )
    return dht, trainer


def store_dummy_rewards(dht: DHT, keys, r, s):
    """在DHT中存储虚拟奖励
    
    Args:
        dht: 分布式哈希表实例
        keys: 节点键列表
        r: 轮次编号
        s: 阶段编号
    """
    for key in keys:
        dht.store(
            key=rewards_key(r, s),
            subkey=key,
            value=[99],
            expiration_time=get_dht_time() + 60,
        )


class StorageMode(Enum):
    """存储模式枚举
    
    定义了三种存储模式：
    - DHT: 仅存储在分布式哈希表中
    - NODE: 仅存储在本地节点中
    - BOTH: 同时存储在DHT和本地节点中
    """
    DHT = 1
    NODE = 2
    BOTH = 3


def store_stage_outputs(
    dht: DHT, node: HivemindNode, r, s, value: dict, storage_mode=StorageMode.BOTH
):
    """存储阶段输出
    
    根据存储模式，将阶段输出存储在DHT和/或本地节点中
    
    Args:
        dht: 分布式哈希表实例
        node: Hivemind节点
        r: 轮次编号
        s: 阶段编号
        value: 要存储的值
        storage_mode: 存储模式，默认为BOTH
    """
    if storage_mode in (StorageMode.DHT, StorageMode.BOTH):
        dht.store(
            key=outputs_key(node.key, r, s),
            subkey=QUESTION,
            value=(0, value),
            expiration_time=get_dht_time() + 120,
        )
    if storage_mode in (StorageMode.NODE, StorageMode.BOTH):
        node.put_stage_outputs(r, s, QUESTION, (0, value))


# 阶段2样本和合并意见的测试数据
STAGE_2_SAMPLES = [
    STAGE_2_OUTPUTS[CK],
    STAGE_2_OUTPUTS["0"],
]

STAGE_2_MERGED_OPINIONS = STAGE_2_MERGED["agent_opinion"]


@pytest.mark.parametrize(
    "merge_fn,sample_fn,stage,samples,group_field,get_expected_fn",
    [
        (
            merge_stage1_question,
            get_stage2_samples,
            0,
            SAMPLES,
            "agent_answers",
            lambda: ("The meaning of life is to sleep.", "The meaning of life is 42."),
        ),
        (
            merge_stage2_question,
            get_stage3_samples,
            1,
            STAGE_2_SAMPLES,
            "agent_opinion",
            lambda: (STAGE_2_MERGED_OPINIONS["0"], STAGE_2_MERGED_OPINIONS[CK]),
        ),
    ],
)
def test_merged_prev_stage_datasets(
    merge_fn, sample_fn, stage, samples, group_field, get_expected_fn
):
    """测试合并前一阶段数据集的功能
    
    测试从DHT和本地节点获取并合并前一阶段数据的功能，验证不同存储模式下的数据访问和合并逻辑。
    
    Args:
        merge_fn: 合并函数
        sample_fn: 样本生成函数
        stage: 阶段编号
        samples: 样本数据
        group_field: 分组字段
        get_expected_fn: 获取预期结果的函数
    """
    dht = hivemind.DHT(start=True)
    coord = HivemindNode.coordinator("test", CK)
    node = HivemindNode("test", "0")

    def merge_coord():
        return merged_prev_stage_datasets(dht, coord, 0, stage + 1, merge_fn, sample_fn)

    def merge_node():
        return merged_prev_stage_datasets(dht, node, 0, stage + 1, merge_fn, sample_fn)

    ## 未存储任何数据！
    with pytest.raises(Exception):
        _ = merge_coord()

    # 训练循环同时保存到两处
    coord_samples = samples_with_key(CK, samples, group_field)
    store_stage_outputs(dht, coord, 0, stage, coord_samples[0], StorageMode.DHT)
    store_stage_outputs(
        dht, coord, 0, stage, coord_samples[1], StorageMode.NODE
    )  # 优先级更高

    node_samples = samples_with_key(node.key, samples, group_field)
    store_stage_outputs(
        dht, node, 0, stage, node_samples[0], StorageMode.NODE
    )  # 仅本地

    ## DHT上奖励不可见！
    coord_expected, node_expected = get_expected_fn()
    cf, nf = merge_coord()[0][0], merge_node()[0][0]

    # 本地
    assert cf[f"{group_field}_{CK}"] == coord_expected
    assert f"{group_field}_{node.key}" not in cf

    # 本地
    assert f"{group_field}_{CK}" not in nf
    assert nf[f"{group_field}_{node.key}"] == node_expected

    ## 检查奖励可见时的合并输出！
    store_dummy_rewards(dht, [coord.key, node.key], 0, stage)
    cf, nf = merge_coord()[0][0], merge_node()[0][0]

    # 本地
    assert cf[f"{group_field}_{CK}"] == coord_expected
    assert f"{group_field}_{node.key}" not in cf

    # 本地 + DHT
    assert nf[f"{group_field}_{CK}"] == node_expected
    assert nf[f"{group_field}_{node.key}"] == node_expected


def test_gsm8k_stage_data(tmp_path):
    """测试GSM8K阶段数据功能
    
    测试GSM8K数据集的阶段数据结构和训练流程，验证多节点环境下的数据处理和阶段转换。
    创建多个节点并运行训练，检查各阶段的输出和奖励。
    
    Args:
        tmp_path: 临时路径
    """
    coord = HivemindNode.coordinator("test", CK)
    nodes = [HivemindNode("test", str(i)) for i in range(3)]

    dht_trainers = [create_dht_and_trainer(Path(tmp_path) / "C", coord, min_peers=1)]
    dht0 = dht_trainers[0][0]
    for i, node in enumerate(nodes):
        dht_trainers.append(
            create_dht_and_trainer(
                Path(tmp_path) / str(i),
                node,
                min_peers=2,
                initial_peers=dht0.get_visible_maddrs(),
            )
        )

    for dht, _ in dht_trainers:
        _ = dht.get_visible_maddrs(latest=True)

    with ThreadPoolExecutor() as executor:
        for dht, trainer in dht_trainers:
            executor.submit(trainer.train)

    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (0, 2)  # 1轮，3个阶段

    def check_outputs(outputs: dict[str, tuple] | None, output_checks={}):
        assert outputs
        qo = outputs[QUESTION][1]
        assert qo["question"] == QUESTION
        assert qo["answer"] == "42"
        for k, check in output_checks.items():
            assert k in qo
            assert check(qo[k])

    for r, s in itertools.product(range(1), range(3)):
        match s:
            case 0:
                checks = {}
            case 1:
                # 合并前只有一个
                checks = {"agent_opinion": lambda c: len(c) == 1}
            case 2:
                checks = {"final_agent_decision": lambda c: len(c) == 1}
            case _:
                checks = {}

        for i in range(len(nodes)):
            check_outputs(
                get_dht_value(dht0, key=outputs_key(nodes[i].key, r, s), latest=True),
                checks,
            )

        rewards = get_dht_value(dht0, key=rewards_key(r, s), latest=True)
        assert rewards
        assert rewards.keys() == set([CK] + [node.key for node in nodes])


def test_gsm8k_follower_no_outputs(tmp_path):
    """测试GSM8K跟随者节点无输出情况
    
    测试当跟随者节点没有本地和DHT输出时的行为。在阶段0重新启动确保每个阶段有多个示例。
    
    Args:
        tmp_path: 临时路径
    """
    node = HivemindNode.coordinator("test", CK)
    dht, trainer = create_dht_and_trainer(Path(tmp_path) / "0", node)

    dht.store(
        key=ROUND_STAGE_NUMBER_KEY,
        value=(0, 1),
        expiration_time=get_dht_time() + node.out_expiration,
    )
    # 节点可能看不到本地和DHT输出。在阶段0重新启动确保每个阶段有多个示例。
    trainer.follower_train(0.1)

    # 检查阶段0输出
    outputs = get_dht_value(dht, key=outputs_key(CK, 0, 0), latest=True)
    assert outputs is not None

def test_gsm8k_delayed_join(tmp_path):
    """测试GSM8K延迟加入功能
    
    测试节点延迟加入训练过程的情况，验证节点能够在训练过程中途加入并正常参与。
    
    Args:
        tmp_path: 临时路径
    """
    node0 = HivemindNode.coordinator("test", CK)
    node1 = HivemindNode("test", "0")

    dht0, trainer0 = create_dht_and_trainer(Path(tmp_path) / "0", node0)
    dht1, trainer1 = create_dht_and_trainer(
        Path(tmp_path) / "1",
        node1,
        initial_peers=dht0.get_visible_maddrs(),
    )
    _ = dht0.get_visible_maddrs(latest=True)
    _ = dht1.get_visible_maddrs(latest=True)

    def delayed_train():
        while trainer0.node.stage_num == 0:
            time.sleep(0.5)

        trainer1.train()

    with ThreadPoolExecutor() as executor:
        executor.submit(trainer0.train)
        executor.submit(delayed_train)

    rs = get_dht_value(dht0, key=RSK, latest=True)
    assert rs == (0, 2)  # 1轮，3个阶段

    for r, s in itertools.product(range(1), range(3)):
        outputs0 = get_dht_value(dht1, key=outputs_key(node0.key, r, s), latest=True)
        assert outputs0

        if s > 0:
            outputs1 = get_dht_value(
                dht0, key=outputs_key(node1.key, r, s), latest=True
            )
            assert outputs1
