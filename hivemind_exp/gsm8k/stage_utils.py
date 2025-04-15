import logging
import time
from collections import defaultdict
from typing import Sequence

import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
import hivemind_exp.gsm8k.stage2_rewards as stage2_rewards
import hivemind_exp.gsm8k.stage3_rewards as stage3_rewards
from hivemind_exp.dht_utils import (
    DHT,
    HivemindNode,
    get_dht_value,
    get_outputs,
    rewards_key,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage2_samples, get_stage3_samples
from hivemind_exp.gsm8k.stage_merger import (
    Any,
    merge_stage1_question,
    merge_stage2_question,
)
from hivemind_exp.hivemind_utils import SingleStageData, StageData


# 这个模块包含GSM8K数据集各阶段之间的工具函数，用于数据合并和阶段转换

def merged_prev_stage_datasets(
    dht: DHT,
    node: HivemindNode,
    r: int,
    s: int,
    merge_fn,
    samples_fn,
    dht_sample_limit = 200,
    check_interval: float = 5,
    wait_timeout: float = 10,
    log_tag=None,
):
    """合并前一阶段的数据集，从本地和DHT获取样本
    
    Args:
        dht: 分布式哈希表实例
        node: Hivemind节点实例
        r: 当前轮次编号
        s: 当前阶段编号
        merge_fn: 用于合并样本的函数
        samples_fn: 用于生成样本的函数
        dht_sample_limit: DHT样本数量限制，默认为200
        check_interval: 检查间隔时间，默认为5秒
        wait_timeout: 等待超时时间，默认为10秒
        log_tag: 日志标签，默认为节点键值
        
    Returns:
        合并后的数据集
    """
    if not log_tag:
        log_tag = node.key

    logger = logging.getLogger(f"{__name__}:{log_tag}")

    merged_qs = []

    # 从本地和DHT检索并合并上一阶段的样本
    def get_prev_rewards():
        return get_dht_value(
            dht, key=rewards_key(r, s - 1), beam_size=100
        )

    prev_rewards: dict[str, Any] | None = get_prev_rewards()
    start_time = time.monotonic()
    while not prev_rewards and time.monotonic() - start_time < wait_timeout:
        logger.info(
            f"无法检索轮次 {r} 阶段 {s - 1} 的奖励；将在 {check_interval}秒后重试 "
        )
        time.sleep(check_interval)
        prev_rewards = get_prev_rewards()

    # 首先添加当前节点的本地样本
    prev_items: dict[str, list] = defaultdict(list)
    try:
        prev_node_outputs = get_outputs(dht, node.key, r, s - 1, node.get_stage_outputs)
        for item in prev_node_outputs.items():
            prev_items[node.key].append(item)
    except ValueError:
        # 在轮次开始后加入
        logger.info(f"无法检索轮次 {r} 阶段 {s - 1} 的本地输出")

    # 仅当奖励可用时添加其他节点的样本
    if prev_rewards:
        node_keys = prev_rewards.keys()
        dht_sample_count = 0
        for node_key in node_keys:
            if dht_sample_count > dht_sample_limit:
                break

            if node_key == node.key:
                continue
            try:
                prev_node_outputs = get_outputs(dht, node_key, r, s - 1)
                for item in prev_node_outputs.items():
                    prev_items[node_key].append(item)

                    dht_sample_count += 1
                    if dht_sample_count > dht_sample_limit:
                        break

            except ValueError:
                # 跳过此节点当前轮次和阶段的答案
                logger.debug(
                    f"发现节点 {node_key} 发布的奖励，但没有输出！"
                )

    # 按问题哈希值分组样本
    q_to_keyed_items: dict[str, dict[str, Any]] = defaultdict(dict)
    for node_key, items in prev_items.items():
        for item in items:
            q_hash, (_, outputs) = item
            q_to_keyed_items[q_hash][node_key] = outputs

    # 合并样本列表
    for outputs in q_to_keyed_items.values():
        merged = merge_fn(outputs)
        merged_qs.append(merged)

    return samples_fn(merged_qs)


def gsm8k_stage_data(
    dht: DHT,
    node: HivemindNode,
    initial_train_dataset,
    initial_test_dataset,
    check_interval: float = 5,
    log_tag=None,
):
    """创建GSM8K数据集的阶段数据结构
    
    Args:
        dht: 分布式哈希表实例
        node: Hivemind节点实例
        initial_train_dataset: 初始训练数据集
        initial_test_dataset: 初始测试数据集
        check_interval: 检查间隔时间，默认为5秒
        log_tag: 日志标签
        
    Returns:
        包含所有阶段数据的StageData实例
    """
    def cumulative_reward_0(**kwargs):
        return stage1_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_1(**kwargs):
        return stage2_rewards.hivemind_cumulative_reward(node, **kwargs)

    def cumulative_reward_2(**kwargs):
        return stage3_rewards.hivemind_cumulative_reward(node, **kwargs)

    def stage2_datasets_fn(r, s):
        return merged_prev_stage_datasets(
            dht,
            node,
            r,
            s,
            merge_stage1_question,
            get_stage2_samples,
            check_interval=check_interval,
            log_tag=log_tag,
        )

    def stage3_datasets_fn(r, s):
        return merged_prev_stage_datasets(
            dht,
            node,
            r,
            s,
            merge_stage2_question,
            get_stage3_samples,
            check_interval=check_interval,
            log_tag=log_tag,
        )

    def round_winners(limit=10) -> Sequence[str]:
        """确定轮次获胜者
        
        Args:
            limit: 返回的获胜者数量限制，默认为10
            
        Returns:
            获胜节点ID列表
        """
        final_stage_outputs, _ = merged_prev_stage_datasets(
            dht,
            node,
            node.round_num,
            3,
            lambda x: x,
            lambda v: (v, v),
            check_interval=check_interval,
            log_tag=log_tag,
        )
        rewards = defaultdict(float)
        for outputs in final_stage_outputs:
            for node_key, output in outputs.items():
                prompts = [
                    [
                        {"role": "system", "content": output["question"]},
                        {"role": "system", "content": output["stage3_prompt"]},
                    ],
                ]
                final_answer = next(iter(output["final_agent_decision"].items()))[1]
                completions = [[{"role": "assistant", "content": final_answer}]]
                cumulative_reward_2(prompts=prompts, completions=completions, **output)
                rewards[node_key] += sum(node.rewards)

        rewards = sorted(list(rewards.items()), key=lambda x: x[1], reverse=True)
        return [n for n, _ in rewards][:limit]

    return StageData(
        round_winner_fn=round_winners,
        stages=[
            SingleStageData(
                name="0",
                reward_funcs=[
                    stage1_rewards.xmlcount_reward_func,
                    stage1_rewards.soft_format_reward_func,
                    stage1_rewards.strict_format_reward_func,
                    stage1_rewards.int_reward_func,
                    stage1_rewards.correctness_reward_func,
                    cumulative_reward_0,
                ],
                datasets_fn=lambda r, s: (initial_train_dataset, initial_test_dataset),  # type: ignore
            ),
            SingleStageData(
                name="1",
                reward_funcs=[
                    stage2_rewards.proper_id_reward_func,
                    stage2_rewards.correctness_reward_func,
                    stage2_rewards.strict_format_reward_func,
                    stage2_rewards.soft_format_reward_func,
                    stage2_rewards.xmlcount_reward_func,
                    cumulative_reward_1,
                ],
                datasets_fn=stage2_datasets_fn,
            ),
            SingleStageData(
                name="2",
                reward_funcs=[
                    stage3_rewards.consensus_reward_func,
                    stage3_rewards.concensus_correctness_reward_func,
                    stage3_rewards.question_recreation_reward_func,
                    stage3_rewards.final_correctness_reward_func,
                    stage3_rewards.strict_format_reward_func,
                    stage3_rewards.soft_format_reward_func,
                    stage3_rewards.xmlcount_reward_func,
                    cumulative_reward_2,
                ],
                datasets_fn=stage3_datasets_fn,
            ),
        ],
    )
