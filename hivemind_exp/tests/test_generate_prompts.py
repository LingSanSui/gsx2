import copy

import pytest

from hivemind_exp.gsm8k.generate_prompts import *
from hivemind_exp.tests.fake_data import *


def test_get_stage2_samples():
    """测试生成第二阶段样本的功能
    
    验证get_stage2_samples函数能否从第一阶段合并数据中正确生成第二阶段的样本。
    打印结果用于手动检查。
    """
    print(get_stage2_samples([STAGE_1_MERGED]))


def test_get_stage2_samples_missing_agents():
    """测试处理缺失代理回答的情况
    
    验证get_stage2_samples函数在某些代理回答缺失的情况下是否仍能正常工作。
    创建两个样本，分别缺少不同代理的回答，测试函数的健壮性。
    """
    # 创建两个样本副本，分别删除不同代理的回答
    s1 = copy.deepcopy(STAGE_1_MERGED)
    s2 = copy.deepcopy(s1)
    del s1["agent_answers"]["0"]
    del s2["agent_answers"]["1"]
    get_stage2_samples([s1, s2])


def test_get_stage3_samples():
    """测试生成第三阶段样本的功能
    
    验证get_stage3_samples函数能否从第二阶段合并数据中正确生成第三阶段的样本。
    打印结果用于手动检查。
    """
    print(get_stage3_samples([STAGE_2_MERGED]))


def test_get_stage3_samples_missing_agents():
    """测试处理缺失代理意见的情况
    
    验证get_stage3_samples函数在某些代理意见缺失的情况下是否仍能正常工作。
    创建两个样本，分别缺少不同代理的意见，测试函数的健壮性。
    """
    # 创建两个样本副本，分别删除不同代理的意见
    s1 = copy.deepcopy(STAGE_2_MERGED)
    s2 = copy.deepcopy(s1)
    del s1["agent_opinion"][CK]
    del s2["agent_opinion"]["0"]
    get_stage3_samples([s1, s2])
