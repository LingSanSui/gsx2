import logging

import colorlog
from trl import GRPOConfig, ModelConfig, TrlParser

from hivemind_exp.chain_utils import (
    ModalSwarmCoordinator,
    WalletSwarmCoordinator,
    setup_web3,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples
from hivemind_exp.runner.gensyn.testnet_grpo_runner import (
    TestnetGRPOArguments,
    TestnetGRPORunner,
)
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner

# 这个模块提供了在单个GPU上训练GSM8K数据集的功能


def main():
    """主函数，设置日志记录，解析命令行参数，并运行训练循环
    
    该函数完成以下任务：
    1. 设置彩色日志记录
    2. 解析命令行参数和配置
    3. 根据参数选择适当的运行器
    4. 使用第一阶段样本运行训练循环
    """
    # 设置日志记录
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(green)s%(levelname)s:%(name)s:%(message)s")
    )
    root_logger.addHandler(handler)

    # 解析命令行参数和配置
    parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))  # type: ignore
    model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()

    # 运行主训练循环
    # 根据参数选择适当的运行器：Modal组织ID、钱包私钥或默认运行器
    if org_id := testnet_args.modal_org_id:
        runner = TestnetGRPORunner(ModalSwarmCoordinator(org_id, web3=setup_web3()))
    elif priv_key := testnet_args.wallet_private_key:
        runner = TestnetGRPORunner(WalletSwarmCoordinator(priv_key, web3=setup_web3()))
    else:
        runner = GRPORunner()

    # 使用第一阶段样本运行训练循环
    runner.run(model_args, grpo_args, training_args, get_stage1_samples)


if __name__ == "__main__":
    main()
