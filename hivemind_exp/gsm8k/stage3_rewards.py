import os
import random
import re
from difflib import SequenceMatcher

import numpy as np

import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
from hivemind_exp.hivemind_utils import HivemindNode

# 这个模块包含GSM8K数据集第三阶段的奖励函数和相关工具函数
# 第三阶段主要是让模型总结前两个阶段的结果并给出最终答案


def extract_xml_identity(text: str) -> str:
    """从文本中提取<majority>标签内的多数派身份标识
    
    Args:
        text: 包含XML格式多数派身份标识的文本
        
    Returns:
        提取并去除空白的多数派身份标识文本
    """
    id = text.split("<majority>")[-1]
    id = id.split("</majority>")[0]
    return id.strip()


def extract_xml_final_answer(text: str) -> str:
    """从文本中提取<answer>标签内的最终答案
    
    Args:
        text: 包含XML格式最终答案的文本
        
    Returns:
        提取并去除空白的最终答案文本
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_xml_question(text: str) -> str:
    """从文本中提取<question>标签内的问题
    
    Args:
        text: 包含XML格式问题的文本
        
    Returns:
        提取并去除空白的问题文本
    """
    question = text.split("<question>")[-1]
    question = question.split("</question>")[0]
    return question.strip()


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


# TODO: Rethink how we add this reward in general setting with delayed rewards. Agents might learn to reward hack by "spamming" identify tags of their choice...
def extract_xml_choices(text: str) -> str:
    """从文本中提取所有<identify>标签内的选择ID
    
    Args:
        text: 包含多个选择ID的文本
        
    Returns:
        提取的选择ID列表
    """
    ids = []
    ids_raw = text.split("<identify>")[1:]
    for id in ids_raw:
        ids += [id.split("</identify>")[0].strip()]
    return ids


def extract_original_question(text: str) -> str:
    """从文本中提取原始问题
    
    Args:
        text: 包含原始问题的文本
        
    Returns:
        提取并去除空白的原始问题文本
    """
    q = text.split("  \n\nThe following answers to this question were suggested:")[0]
    q = q.split("The question we were given is: ")[-1]
    return q.strip()


def extract_answers(text: str) -> str:
    """从文本中提取所有学生的答案
    
    Args:
        text: 包含多个学生答案的文本
        
    Returns:
        学生ID到答案的映射字典
    """
    answers = {}
    raw = text.split(
        "  \nAfter comparing these answers, the following feedback was given about which answer is best: \n"
    )[0].split("<student>")[1:]
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
    # 检查<summarize_feedback>标签的开始格式是否正确
    if text.count("<summarize_feedback>\n") == 1:
        count += 0.125
    # 检查</summarize_feedback>标签的结束格式是否正确
    if text.count("\n</summarize_feedback>\n") == 1:
        count += 0.125
    # 检查<majority>标签的开始格式是否正确
    if text.count("<majority>\n") == 1:
        count += 0.125
    # 检查</majority>标签的结束格式是否正确
    if text.count("\n</majority>\n") == 1:
        count += 0.125
    # 检查<question>标签的开始格式是否正确
    if text.count("<question>\n") == 1:
        count += 0.125
    # 检查</question>标签的结束格式是否正确
    if text.count("\n</question>\n") == 1:
        count += 0.125
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


def swarm_majority(choices):
    """计算选择中的多数派选项
    
    Args:
        choices: 选择列表
        
    Returns:
        获得最多票数的选项列表
    """
    votes = {}
    max_votes = 0
    # 统计每个选项的票数
    for c in choices:
        if c in votes:
            votes[c] += 1
        else:
            votes[c] = 1
        # 更新最大票数
        if votes[c] > max_votes:
            max_votes = votes[c]
    majority = []
    # 找出所有具有最大票数的选项
    for c in votes:
        if votes[c] >= max_votes:
            majority += [c]
    return majority


# 奖励函数
def consensus_reward_func(
    prompts, completions, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    """评估模型是否选择了多数派认为正确的答案并给予奖励
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为2.0
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表
    """
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 获取提示内容
    p = prompts[0][-1]["content"]
    # 从提示中提取评论者的选择
    critic_choices = extract_xml_choices(p)
    # 计算多数派选择
    majority_choices = swarm_majority(critic_choices)
    # 从回答中提取模型选择的身份标识
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
            "consensus_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nCritic Choice Distribution:\n{critic_choices}\n\nExtracted:\n{extracted_responses[0]}\n\nGot reward? {extracted_responses[0] in majority_choices}"
            f.write(out_line)
    # 如果提取的身份标识在多数派选择中，则给予加权奖励，否则为0
    return [
        1.0 * weighting if r in majority_choices else 0.0 for r in extracted_responses
    ]


def question_recreation_reward_func(
    prompts, completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    """评估模型重新创建原始问题的准确性并给予奖励
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为1.0
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表，基于重建问题与原始问题的相似度
    """
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 获取提示内容
    p = prompts[0][-1]["content"]
    # 从提示中提取原始问题
    q = extract_original_question(p)
    # 从回答中提取模型重建的问题
    recreated_qs = [extract_xml_question(r) for r in responses]
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "question_recreation_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nOriginal Question:\n{q}\n\nExtracted recreation:\n{recreated_qs[0]}\n\nGot reward? {SequenceMatcher(None, recreated_qs[0], q).ratio()}"
            f.write(out_line)
    # 返回基于问题相似度的加权奖励值列表
    return [SequenceMatcher(None, r, q).ratio() * weighting for r in recreated_qs]


def concensus_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    """评估模型选择的答案的正确性并给予奖励
    
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
    # 获取提示内容
    p = prompts[0][-1]["content"]
    # 从提示中提取所有学生的答案
    agent_answers = extract_answers(p)
    # 从回答中提取模型选择的身份标识
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


def final_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    """评估模型最终答案的正确性并给予奖励
    
    Args:
        prompts: 提示列表
        completions: 模型完成的回答列表
        answer: 正确答案列表
        weighting: 奖励权重，默认为2.0
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表，基于最终答案是否正确
    """
    # 提取所有完成的回答内容
    responses = [completion[0]["content"] for completion in completions]
    # 获取提示内容
    p = prompts[0][-1]["content"]
    # 从回答中提取模型的最终答案
    extracted_responses = [extract_xml_final_answer(r) for r in responses]
    # 有1%的概率将样本写入文件进行记录
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "final_answer_correctness_samples.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"Prompt:\n{p}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{responses[0]}\n\nExtracted:\n{extracted_responses[0]}"
            f.write(out_line)
    # 如果提取的最终答案与正确答案完全匹配，则给予加权奖励，否则为0
    return [
        1.0 * weighting if r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


def strict_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    """检查回答是否严格遵循特定格式的奖励函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为0.5
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表，基于是否匹配严格格式
    """
    # 定义严格的XML格式模式，要求精确的换行和标签位置
    pattern = r"^<summarize_feedback>\n.*?\n</summarize_feedback>\n<majority>\n.*?\n</majority>\n<question>\n.*?\n</question>\n<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
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
            "s3_strict_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    # 如果匹配严格格式，则给予加权奖励，否则为0
    return [1.0 * weighting if match else 0.0 for match in matches]


def soft_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    """检查回答是否遵循宽松格式的奖励函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为0.5
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表，基于是否匹配宽松格式
    """
    # 定义宽松的XML格式模式，允许更灵活的空白和换行
    pattern = r"<summarize_feedback>.*?</summarize_feedback>\s*<majority>.*?</majority>\s*<question>.*?</question>\s*<think>.*?</think>\s*<answer>.*?</answer>"
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
            "s3_soft_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    return [1.0 * weighting if match else 0.0 for match in matches]


def xmlcount_reward_func(
    completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    """基于XML标签格式正确性给予奖励的函数
    
    Args:
        completions: 模型完成的回答列表
        weighting: 奖励权重，默认为1.0
        logging: 是否记录日志，默认为False
        **kwargs: 其他参数
        
    Returns:
        每个回答的奖励值列表，基于XML标签格式正确性
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
            "count_xml_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = (
                f"\nResponse:\n{contents[0]}\n\nCount reward: {count_xml(contents[0])}"
            )
            f.write(out_line)
    return [count_xml(c) * weighting for c in contents]


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
    
    这个函数是第三阶段训练中与Hivemind节点交互的核心函数。它计算各种奖励函数的值，
    将它们累加为总奖励，然后根据输出信号选择器选择最佳回答，并将结果保存到Hivemind节点中。
    在第三阶段中，模型需要总结前两个阶段的结果，确定多数派意见，重新创建原始问题，
    并给出最终答案。这是整个训练过程的最后一个阶段，产生最终的问题解决方案。
    
    Args:
        node: HivemindNode节点对象，用于存储输出和奖励
        prompts: 提示列表，包含前两个阶段的结果和评估
        completions: 模型完成的回答列表，包含模型的总结、多数派选择和最终答案
        answer: 正确答案列表，用于评估模型最终答案的正确性
        logging: 是否记录日志，默认为False
        output_signal_selector: 输出信号选择器，默认为"max"，选择奖励最高的回答
        **kwargs: 其他参数，传递给各个奖励函数
        
    Returns:
        返回全零的奖励列表，实际奖励已保存到node.rewards中
    """
    consensus_reward = consensus_reward_func(prompts, completions, logging=logging)
    concensus_correctness = concensus_correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    question_recreation_reward = question_recreation_reward_func(
        prompts, completions, logging=logging
    )
    final_correctness = final_correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    strict_format_reward = strict_format_reward_func(completions, logging=logging)
    soft_format_reward = soft_format_reward_func(completions, logging=logging)
    xmlcount_reward = xmlcount_reward_func(completions, logging=logging)
    total_reward = [
        sum(tup)
        for tup in zip(
            consensus_reward,
            concensus_correctness,
            question_recreation_reward,
            final_correctness,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    prompt = prompts[0][-1]["content"]
    question = extract_original_question(prompt)
    if output_signal_selector == "max":
        # Generate output line
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": question,
            "answer": answer[0],
            "stage3_prompt": prompt,
            "final_agent_decision": {node.key: responses[maximal_reward_idx]},
        }

    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = total_reward

    return [0.0 for _ in total_reward]
