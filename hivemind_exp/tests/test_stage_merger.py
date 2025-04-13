import pytest

from hivemind_exp.gsm8k.stage_merger import *
from hivemind_exp.tests.fake_data import *


def test_merge_stage1():
    """测试第一阶段数据合并功能
    
    验证merge_stage1_question函数能否正确合并来自不同代理的第一阶段输出数据，
    并与预期的合并结果进行比较。
    """
    merged = merge_stage1_question(STAGE_1_OUTPUTS)
    assert merged == STAGE_1_MERGED


def test_merge_stage2():
    """测试第二阶段数据合并功能
    
    验证merge_stage2_question函数能否正确合并来自不同代理的第二阶段输出数据，
    并与预期的合并结果进行比较。
    """
    merged = merge_stage2_question(STAGE_2_OUTPUTS)
    assert merged == STAGE_2_MERGED
