from typing import Any

# 这个模块包含GSM8K数据集各阶段之间的合并函数，用于整合不同节点的输出

def merge_stage1_question(outputs: dict[str, dict[str, Any]]):
    """合并第一阶段的问题和答案
    
    Args:
        outputs: 包含不同节点输出的字典
        
    Returns:
        合并后的数据字典，包含问题、答案和所有代理的答案
    """
    # TODO: Currently question+answer keeps getting replaced at every file. This is wasteful and can be optimized
    # TODO: If an agents' answers more than once (or >1 answer from the same agent id hash), then current implementation will only keep the last seen in the loop. Should allow for multiple answers?
    merged = {"question": None, "answer": None, "agent_answers": {}}
    for o in outputs.values():
        merged["question"] = o["question"]
        merged["answer"] = o["answer"]
        merged["agent_answers"].update(o["agent_answers"])
    # 填充默认值。TODO: 决定这是否是一个好的选择
    for agent in outputs:
        if agent not in merged["agent_answers"]:
            merged["agent_answers"].update({agent: "No answer received..."})
    return merged


def merge_stage2_question(outputs: dict[str, dict[str, Any]]):
    """合并第二阶段的问题和答案
    
    Args:
        outputs: 包含不同节点输出的字典
        
    Returns:
        合并后的数据字典，包含问题、答案、第二阶段提示和代理意见
    """
    # TODO: Currently question+answer keeps getting replaced at every file. This is wasteful and can be optimized
    # TODO: If an agents' answers more than once (or >1 answer from the same agent id hash), then current implementation will only keep the last seen in the loop. Should allow for multiple answers?
    merged = {
        "question": None,
        "answer": None,
        "stage2_prompt": None,
        "agent_opinion": {},
    }
    for o in outputs.values():
        for col in ["question", "answer", "stage2_prompt"]:
            if col in o:
                merged[col] = o[col]
        if "agent_opinion" in o:
            merged["agent_opinion"].update(o["agent_opinion"])
    # 填充默认值。TODO: 决定这是否是一个好的选择
    for agent in outputs:
        if agent not in merged["agent_opinion"]:
            merged["agent_opinion"].update({agent: "No feedback received..."})
    return merged
