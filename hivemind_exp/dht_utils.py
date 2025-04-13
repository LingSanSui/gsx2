import hashlib
from functools import lru_cache
from typing import Any

from hivemind.dht import DHT
from hivemind.utils import ValueWithExpiration

from hivemind_exp.hivemind_utils import HivemindNode

# 这个模块提供了与分布式哈希表(DHT)交互的工具函数，用于在分布式训练中共享和检索数据

ROUND_STAGE_NUMBER_KEY = "rl_swarm_rs"  # 无子键。由协调器发布。

# 轮次和阶段（例如 0_0）会被附加到键后面。
LEADERBOARD_KEY_PREFIX = (
    "rl_swarm_leaderboard"  # 子键 = 指标。由协调器发布。
)
REWARDS_KEY = "rl_swarm_rewards"  # 子键 = 指标。所有节点都可以发布。

# 节点键、轮次和阶段（例如 abcde_0_0）会被附加到键后面。
OUTPUTS_KEY_PREFIX = "rl_swarm_outputs"  # 子键 = 示例哈希。所有节点都可以发布。


def leaderboard_key(round_num, stage) -> str:
    """生成特定轮次和阶段的排行榜键
    
    Args:
        round_num: 轮次编号
        stage: 阶段编号
        
    Returns:
        格式化的排行榜键字符串
    """
    return f"{LEADERBOARD_KEY_PREFIX}_{round_num}_{stage}"


def rewards_key(round_num, stage) -> str:
    """生成特定轮次和阶段的奖励键
    
    Args:
        round_num: 轮次编号
        stage: 阶段编号
        
    Returns:
        格式化的奖励键字符串
    """
    return f"{REWARDS_KEY}_{round_num}_{stage}"


def outputs_key(node_key: str, round_num, stage) -> str:
    """生成特定节点、轮次和阶段的输出键
    
    Args:
        node_key: 节点唯一标识符
        round_num: 轮次编号
        stage: 阶段编号
        
    Returns:
        格式化的输出键字符串
    """
    return f"{OUTPUTS_KEY_PREFIX}_{node_key}_{round_num}_{stage}"


def node_outputs_key(node: HivemindNode) -> str:
    """从HivemindNode对象生成输出键
    
    Args:
        node: HivemindNode实例
        
    Returns:
        基于节点信息的输出键字符串
    """
    return outputs_key(node.key, node.round_num, node.stage_num)


def hash_keys(outputs):
    """处理输出字典的键，确保所有键都是哈希值
    
    处理训练器的旧版本，这些版本没有对问题键进行哈希处理。
    
    Args:
        outputs: 输出字典，键为问题，值为输出
        
    Returns:
        处理后的字典，所有键都是32字符的哈希值
    """
    result = {}
    for k, v in outputs.items():
        if len(k) != 32:  # 不完美，但足够使用
            k = hashlib.md5(k.encode()).hexdigest()
        result[k] = v

    return result


@lru_cache
def get_outputs(
    dht: DHT, node_key: str, r, s, get_cached_fn=None
) -> dict[str, tuple[float, dict]]:  # Q: (timestamp, outputs)
    """获取特定节点、轮次和阶段的输出
    
    首先尝试使用提供的缓存函数，如果失败则从DHT获取。
    
    Args:
        dht: 分布式哈希表实例
        node_key: 节点唯一标识符
        r: 轮次编号
        s: 阶段编号
        get_cached_fn: 可选的缓存获取函数
        
    Returns:
        问题到(时间戳, 输出)元组的字典
        
    Raises:
        ValueError: 如果无法检索输出
    """
    # 首先尝试提供的缓存函数
    if get_cached_fn:
        if outputs := get_cached_fn(r, s):
            return hash_keys(outputs)

    # 接下来从DHT尝试获取，包括对等节点的输出
    if outputs := get_dht_value(dht, key=outputs_key(node_key, r, s), latest=False):
        return hash_keys(outputs)

    raise ValueError(
        f"无法检索节点 {node_key} 在轮次 {r} 阶段 {s} 的输出"
    )


def get_round_and_stage(
    dht: DHT,
) -> tuple[int, int]:
    """从DHT获取当前轮次和阶段
    
    Args:
        dht: 分布式哈希表实例
        
    Returns:
        当前轮次和阶段的元组
        
    Raises:
        ValueError: 如果无法找到当前轮次和阶段
    """
    value = get_dht_value(dht, key=ROUND_STAGE_NUMBER_KEY, latest=True)
    if not value:
        raise ValueError("无法找到当前轮次和阶段")

    round_num, stage = value
    return round_num, stage


def get_dht_value(dht: DHT, **kwargs) -> Any | None:
    """从DHT获取值并处理包装器
    
    Args:
        dht: 分布式哈希表实例
        **kwargs: 传递给dht.get的参数
        
    Returns:
        解包后的值，如果不存在则返回None
    """
    wrapper = dht.get(**kwargs)
    if not wrapper:
        return None

    assert isinstance(wrapper, ValueWithExpiration)
    value = wrapper.value
    if isinstance(value, dict):
        # 存在子键；解包ValueWithExpiration
        return {k: v.value for k, v in value.items()}
    return value
