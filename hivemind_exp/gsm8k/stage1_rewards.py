import os
import random
import re

import numpy as np

from hivemind_exp.hivemind_utils import HivemindNode

# 这个模块包含GSM8K数据集第一阶段的奖励函数和相关工具函数
# 第一阶段主要是让模型直接解答数学问题并生成格式化的答案


def extract_xml_answer(text: str) -> str:
    """从文本中提取<answer>标签内的答案
    
    Args:
        text: 包含XML格式答案的文本
        
    Returns:
        提取并去除空白的答案文本
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def count_xml(text) -> float:
    """计算文本中XML标签的格式正确性得分
    
    Args:
        text: 需要评估的文本
        
    Returns:
        基于XML标签格式正确性的浮点数得分
    """
    count = 0.0
    # 检查<think>标签的开始格式是否正确
    if text.count("<think>\n") == 1:
        count += 0.125
    # 检查</think>标签的结束格式是否正确
    if text.count("\n</think>\n") == 1:
        count += 0.125
    # 检查<answer>标签的开始格式是否正确
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        # 对答案后的多余文本进行轻微惩罚
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    # 检查</answer>标签的结束格式是否正确
    if text.count("\n</answer>") == 1:
        count += 0.125
        # 对答案后的多余文本进行轻微惩罚
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# 奖励函数
def correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    """评估模型回答的正确性并给予奖励
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        answer: 正确答案列表
        weighting: 奖励权重，默认为2.0
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 获取问题内容
    q = prompts[0][-1]["content"]
    # 从回答中提取XML格式的答案部分
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  
        os.makedirs(
            f"model_output_samples/gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "correctness_samples.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"Question:\n{q}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{responses[0]}\n\nExtracted:\n{extracted_responses[0]}"
            f.write(out_line)
    
    # 如果提取的答案与正确答案完全匹配，则给予加权奖励，否则为0
    return [
        1.0 * weighting if r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


def int_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    """检查答案是否为整数并给予奖励
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为0.5
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 从回答中提取XML格式的答案部分
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # 如果提取的答案是数字，则给予加权奖励，否则为0
    return [1.0 * weighting if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    """检查回答是否严格遵循特定格式的奖励函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为0.5
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 定义严格的XML格式模式，要求精确的换行和标签位置
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 检查每个回答是否匹配严格的格式模式
    matches = [re.match(pattern, r) for r in responses]
    # 如果匹配严格格式，则给予加权奖励，否则为0
    return [1.0 * weighting if match else 0.0 for match in matches]


def soft_format_reward_func(completions, weighting=0.5, **kwargs) -> list[float]:
    """检查回答是否遵循宽松格式的奖励函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为0.5
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 定义宽松的XML格式模式，允许更灵活的空白和换行
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 检查每个回答是否匹配宽松的格式模式
    matches = [re.match(pattern, r) for r in responses]
    # 如果匹配宽松格式，则给予加权奖励，否则为0
    return [1.0 * weighting if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, weighting=1.0, **kwargs) -> list[float]:
    """基于XML标签格式正确性给予奖励的函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为1.0
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 提取所有完成的回答内容
    contents = [completion[0]["content"] for completion in completions]
    # 对每个回答内容计算XML格式得分并应用权重
    return [count_xml(c) * weighting for c in contents]

def top_k_cumulative_reward(
    prompts,
    completions,
    answer,
    logging=False,
    **kwargs,
) -> list[float]:
    """
    累积所有奖励为一个总奖励，用于提示生成的top_k选择器
    
    将各个奖励函数的结果累加起来，形成一个综合评分，用于在多个候选答案中选择最佳答案。
    这个函数结合了正确性、整数检查、格式检查和XML标签质量等多个方面的评分。
    
    Args:
        prompts: 提示列表，包含问题内容
        completions: 模型完成的回答列表，每个元素是一个回答
        answer: 正确答案列表，用于评估回答的正确性
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数，传递给各个奖励函数
        
    Returns:
        每个回答的累积奖励值列表，值越高表示质量越好
    """
    # 计算正确性奖励
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    # 计算整数奖励
    int_reward = int_reward_func(completions)
    # 计算严格格式奖励
    strict_format_reward = strict_format_reward_func(completions)
    # 计算宽松格式奖励
    soft_format_reward = soft_format_reward_func(completions)
    # 计算XML计数奖励
    xmlcount_reward = xmlcount_reward_func(completions)
    
    # 将所有奖励相加得到总奖励
    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]
    return total_reward


def hivemind_cumulative_reward(
    node: HivemindNode,
    prompts,
    completions,
    answer,
    logging=False,
    output_signal_selector="max",
    **kwargs,
) -> list[float]:
    """
    累积所有奖励函数的总和并将JSON保存到节点输出
    
    这个函数是第一阶段训练中与Hivemind节点交互的核心函数。它计算各种奖励函数的值，
    将它们累加为总奖励，然后根据输出信号选择器选择最佳回答，并将结果保存到Hivemind节点中。
    这使得其他节点可以访问和使用这些结果进行分布式训练。
    
    Args:
        node: Hivemind节点实例，用于存储输出和奖励
        prompts: 提示列表，包含问题内容
        completions: 模型完成的回答列表，每个元素是一个回答
        answer: 正确答案列表，用于评估回答的正确性
        logging: 是否记录日志，默认为False
        output_signal_selector: 输出信号选择器，默认为"max"，选择奖励最高的回答
        **kwargs: 其他参数，传递给各个奖励函数
        
    Returns:
        每个回答的累积奖励值列表（实际返回全0列表，真实奖励保存在node.rewards中）
    """
    # 计算各种奖励函数的值
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    # 计算整数奖励
    int_reward = int_reward_func(completions)
    # 计算严格格式奖励
    strict_format_reward = strict_format_reward_func(completions)
    # 计算宽松格式奖励
    soft_format_reward = soft_format_reward_func(completions)
    # 计算XML计数奖励
    xmlcount_reward = xmlcount_reward_func(completions)
    
    # 计算总奖励值
    total_reward = [
        sum(tup)
        for tup in zip(
            correctness_reward,
            int_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    # 获取提示内容
    prompt = prompts[0][-1]["content"]
    if output_signal_selector == "max":
        # 生成输出数据，选择奖励最高的回答
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": prompt,
            "answer": answer[0],
            "agent_answers": {node.key: responses[maximal_reward_idx]},
        }

    # 如果指定了输出信号选择器，则保存输出数据和奖励值到节点
    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = total_reward

    # 返回全0列表，实际奖励已保存在node.rewards中
    return [0.0 for _ in total_reward]
