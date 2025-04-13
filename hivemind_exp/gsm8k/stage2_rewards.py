import os
import random
import re

import numpy as np

import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
from hivemind_exp.hivemind_utils import HivemindNode

# 这个模块包含GSM8K数据集第二阶段的奖励函数和相关工具函数
# 第二阶段主要是让模型评估和比较第一阶段中不同模型的答案


def extract_xml_identity(text: str) -> str:
    """从文本中提取<identify>标签内的身份标识
    
    Args:
        text: 包含XML格式身份标识的文本
        
    Returns:
        提取并去除空白的身份标识文本
    """
    id = text.split("<identify>")[-1]
    id = id.split("</identify>")[0]
    return id.strip()


def extract_xml_ids(text: str) -> str:
    """从文本中提取所有<student>标签内的学生ID
    
    Args:
        text: 包含多个学生ID的文本
        
    Returns:
        提取的学生ID列表
    """
    ids = []
    ids_raw = text.split("<student>")[1:]
    for id in ids_raw:
        ids += [id.split("</student>")[0].strip()]
    return ids


def extract_original_question(text: str) -> str:
    """从文本中提取原始问题
    
    Args:
        text: 包含原始问题的文本
        
    Returns:
        提取的原始问题文本
    """
    q = text.split("  \n\nThe following answers to this question were suggested:")[0]
    q = q.split("The question we were given is: ")[-1]
    return q


def extract_answers(text: str) -> str:
    """从文本中提取所有学生的答案
    
    Args:
        text: 包含多个学生答案的文本
        
    Returns:
        学生ID到答案的映射字典
    """
    answers = {}
    raw = text.split("<student>")[1:]
    for a in raw:
        id = a.split("</student>")[0].strip()
        ans = a.split("</student> said \n")[-1].strip()
        answers[id] = ans
    return answers


def count_xml(text) -> float:
    """计算文本中XML标签的格式正确性得分
    
    Args:
        text: 需要评估的文本
        
    Returns:
        基于XML标签格式正确性的浮点数得分
    """
    count = 0.0
    # 检查<compare>标签的开始格式是否正确
    if text.count("<compare>\n") == 1:
        count += 0.125
    # 检查</compare>标签的结束格式是否正确
    if text.count("\n</compare>\n") == 1:
        count += 0.125
    # 检查<explain>标签的开始格式是否正确
    if text.count("<explain>\n") == 1:
        count += 0.125
    # 检查</explain>标签的结束格式是否正确
    if text.count("\n</explain>\n") == 1:
        count += 0.125
    # 检查<identify>标签的开始格式是否正确
    if text.count("\n<identify>\n") == 1:
        count += 0.125
        # 对标识后的多余文本进行轻微惩罚
        count -= len(text.split("\n</identify>\n")[-1]) * 0.001
    # 检查</identify>标签的结束格式是否正确
    if text.count("\n</identify>") == 1:
        count += 0.125
        # 对标识后的多余文本进行轻微惩罚
        count -= (len(text.split("\n</identify>")[-1]) - 1) * 0.001
    return count


# Reward functions
def proper_id_reward_func(
    prompts, completions, answer, weighting=2.0, logging=True, **kwargs
) -> list[float]:
    """评估模型是否选择了有效的学生ID并给予奖励
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        answer: 正确答案列表
        weighting: 奖励权重，默认为2.0
        logging: 是否记录日志，默认为True
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 获取提示内容
    p = prompts[0][-1]["content"]
    # 从提示中提取所有有效的学生ID
    agent_ids = extract_xml_ids(p)
    # 从回答中提取模型选择的学生ID
    extracted_responses = [extract_xml_identity(r) for r in responses]
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "id_extact_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nValid IDs:\n{agent_ids}\n\nExtracted:\n{extracted_responses[0]}\n\nGot reward? {extracted_responses[0] in agent_ids}"
            f.write(out_line)
    # 如果提取的学生ID在有效ID列表中，则给予加权奖励，否则为0
    return [1.0 * weighting if r in agent_ids else 0.0 for r in extracted_responses]


def correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=True, **kwargs
) -> list[float]:
    """评估模型选择的答案的正确性并给予奖励
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        answer: 正确答案列表
        weighting: 奖励权重，默认为2.0
        logging: 是否记录日志，默认为True
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 获取提示内容
    p = prompts[0][-1]["content"]
    # 从提示中提取所有学生的答案
    agent_answers = extract_answers(p)
    # 从回答中提取模型选择的学生ID
    extracted_responses = [extract_xml_identity(r) for r in responses]
    chosen_rewards = []
    for r in extracted_responses:
        cur_reward = 0
        # 如果模型选择了某个学生的答案
        if r in agent_answers:
            # 如果学生答案正确，加1.0分
            if stage1_rewards.extract_xml_answer(agent_answers[r]) == answer[0]:
                cur_reward += 1.0
            # 如果学生答案是数字，加0.5分
            if stage1_rewards.extract_xml_answer(agent_answers[r]).isdigit():
                cur_reward += 0.5
            # 如果学生答案符合严格格式，加0.5分
            pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 0.5
            # 如果学生答案符合宽松格式，加0.5分
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 0.5
            # 加上学生答案的XML格式得分
            cur_reward += stage1_rewards.count_xml(agent_answers[r])
        # 如果模型认为所有答案都错误
        elif r in [
            "None",
            "No one",
            "All answers are wrong",
            "All answers were wrong",
            "All are wrong",
            "All were wrong",
            "None are correct",
            "None were correct",
            "No one is correct",
        ]:
            # 提取所有学生答案
            agent_as = [
                stage1_rewards.extract_xml_answer(agent_answers[id])
                for id in agent_answers
            ]
            # 检查所有学生答案是否都错误
            check_submissions = [
                True if r == a else False for r, a in zip(agent_as, answer)
            ]
            # 如果所有学生答案确实都错误，加10分
            if all(check_submissions):
                cur_reward += 10
        chosen_rewards += [cur_reward]
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        if extracted_responses[0] in agent_answers:
            os.makedirs(
                f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                exist_ok=True,
            )
            log_file = os.path.join(
                "model_output_samples",
                f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                "correctness_samps.txt",
            )
            with open(log_file, "a") as f:
                f.write("-" * 20)
                out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nChosen answer ID:\n{extracted_responses[0]}\n\nExtracted:\n{agent_answers[extracted_responses[0]]}\n\nReward for choice: {chosen_rewards[0]}"
                f.write(out_line)
    # 返回加权后的奖励值列表
    return [r * weighting for r in chosen_rewards]


def strict_format_reward_func(
    completions, weighting=0.5, logging=True, **kwargs
) -> list[float]:
    """检查回答是否严格遵循特定格式的奖励函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为0.5
        logging: 是否记录日志，默认为True
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 定义严格的XML格式模式，要求精确的换行和标签位置
    pattern = r"^<compare>\n.*?\n</compare>\n<explain>\n.*?\n</explain>\n<identify>\n.*?\n</identify>\n$"
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 检查每个回答是否匹配严格的格式模式
    matches = [re.match(pattern, r) for r in responses]
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s2_strict_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    # 如果匹配严格格式，则给予加权奖励，否则为0
    return [1.0 * weighting if match else 0.0 for match in matches]


def soft_format_reward_func(
    completions, weighting=0.5, logging=True, **kwargs
) -> list[float]:
    """检查回答是否遵循宽松格式的奖励函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为0.5
        logging: 是否记录日志，默认为True
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 定义宽松的XML格式模式，允许更灵活的空白和换行
    pattern = (
        r"<compare>.*?</compare>\s*<explain>.*?</explain>\s*<identify>.*?</identify>"
    )
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 检查每个回答是否匹配宽松的格式模式
    matches = [re.match(pattern, r) for r in responses]
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s2_soft_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    # 如果匹配宽松格式，则给予加权奖励，否则为0
    return [1.0 * weighting if match else 0.0 for match in matches]


def xmlcount_reward_func(
    completions, weighting=1.0, logging=True, **kwargs
) -> list[float]:
    """基于XML标签格式正确性给予奖励的函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为1.0
        logging: 是否记录日志，默认为True
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 提取所有完成的回答内容
    contents = [completion[0]["content"] for completion in completions]
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "strict_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = (
                f"\nResponse:\n{contents[0]}\n\nCount reward: {count_xml(contents[0])}"
            )
            f.write(out_line)
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
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        answer: 正确答案列表
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的累积奖励值列表
    """
    # 计算各种奖励函数的值
    proper_id_reward = proper_id_reward_func(
        prompts, completions, answer, logging=logging
    )
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    strict_format_reward = strict_format_reward_func(completions, logging=logging)
    soft_format_reward = soft_format_reward_func(completions, logging=logging)
    xmlcount_reward = xmlcount_reward_func(completions, logging=logging)
    
    # 计算总奖励值
    total_reward = [
        sum(tup)
        for tup in zip(
            proper_id_reward,
            correctness_reward,
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
    累积所有奖励为一个总奖励并将JSON结果保存到节点输出
    
    这个函数是第二阶段训练中与Hivemind节点交互的核心函数。它计算各种奖励函数的值，
    将它们累加为总奖励，然后根据输出信号选择器选择最佳回答，并将结果保存到Hivemind节点中。
    在第二阶段中，模型需要评估和比较第一阶段中不同模型的答案，选择最佳答案或指出所有答案都不正确。
    
    Args:
        node: HivemindNode节点对象，用于存储输出和奖励
        prompts: 提示列表，包含原始问题和多个学生的答案
        completions: 模型完成的回答列表，包含模型对各个答案的评估和选择
        answer: 正确答案列表，用于评估模型选择的正确性
        logging: 是否记录日志，默认为False
        output_signal_selector: 输出信号选择器，默认为"max"，选择奖励最高的回答
        **kwargs: 其他参数，传递给各个奖励函数
        
    Returns:
        返回全零的奖励列表，实际奖励已保存到node.rewards中
    """
    # 计算ID正确性奖励
    proper_id_reward = proper_id_reward_func(
        prompts, completions, answer, logging=logging
    )
    # 计算答案正确性奖励
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    # 计算严格格式奖励
    strict_format_reward = strict_format_reward_func(completions, logging=logging)
    # 计算宽松格式奖励
    soft_format_reward = soft_format_reward_func(completions, logging=logging)
    # 计算XML标签格式奖励
    xmlcount_reward = xmlcount_reward_func(completions, logging=logging)
    # 累积所有奖励
    total_reward = [
        sum(tup)
        for tup in zip(
            proper_id_reward,
            correctness_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    # 从提示中提取原始问题
    question = extract_original_question(prompts[0][-1]["content"])
    if output_signal_selector == "max":
        # 生成输出数据，选择奖励最高的回答
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": question,  # 原始问题
            "answer": answer[0],  # 正确答案
            "stage2_prompt": prompts[0][-1]["content"],  # 第二阶段的提示
            "agent_opinion": {node.key: responses[maximal_reward_idx]},  # 节点的意见（奖励最高的回答）
        }

    # 如果输出信号选择器不为None，则保存输出数据和奖励到节点
    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = total_reward

    # 返回全零的奖励列表，实际奖励已保存到node.rewards中
    return [0.0 for _ in total_reward]
