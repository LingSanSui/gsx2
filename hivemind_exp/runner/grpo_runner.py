import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Tuple

import hivemind
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, ModelConfig

from hivemind_exp.gsm8k.stage_utils import gsm8k_stage_data
from hivemind_exp.hivemind_utils import HivemindNode
from hivemind_exp.name_utils import get_name_from_peer_id
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer

# 创建日志记录器
logger = logging.getLogger(__name__)

@dataclass
class GRPOArguments:
    """
    GRPO训练参数数据类，包含Hivemind网络配置、模型参数和Hugging Face Hub参数
    """
    # Hivemind参数
    initial_peers: list[str] = field(default_factory=list)  # 初始对等节点列表
    public_maddr: str | None = None  # 公共多地址
    host_maddr: str | None = None  # 主机多地址
    identity_path: str | None = None  # 身份路径
    max_rounds: int = 100  # 最大训练轮数

    # 模型参数
    dataset_id_or_path: str = "openai/gsm8k"  # 数据集ID或路径
    dataset_splits: str = "train"  # 数据集分割
    tokenizer_name_or_path: str | None = None  # 分词器名称或路径
    number_of_data_samples: int = 50000  # 数据样本数量
    public_maddr: str | None = None  # 公共多地址

    # Hugging Face Hub参数
    hf_token: str | None = None  # HF令牌


class GRPORunner:
    """
    GRPO运行器类，负责设置和运行GRPO训练过程
    
    该类负责初始化分布式训练环境、加载模型和分词器、设置DHT网络，并启动训练过程。
    它协调多个节点之间的通信，并管理训练过程中的各个阶段。
    """
    def get_model(self, args: GRPOConfig, model_name: str):
        """
        获取预训练的因果语言模型
        
        根据提供的模型名称或路径加载预训练的因果语言模型，并应用配置参数中的初始化选项。
        如果启用了梯度检查点，则会禁用模型缓存以确保兼容性。
        
        Args:
            args: GRPO配置参数，包含模型初始化选项和训练配置
            model_name: 模型名称或Hugging Face Hub上的路径
            
        Returns:
            预训练的因果语言模型实例，已应用指定的初始化选项
        """
        model_init_kwargs = args.model_init_kwargs or {}
        # 如果启用了梯度检查点（不支持缓存），则禁用缓存
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        return AutoModelForCausalLM.from_pretrained(model_name, **model_init_kwargs)

    def get_tokenizer_name(self, model_args: ModelConfig, script_args: GRPOArguments):
        """
        获取分词器名称
        
        根据提供的参数确定要使用的分词器名称或路径。优先使用script_args中指定的分词器，
        如果未指定，则使用model_args中的模型名称作为分词器名称。如果两者都未指定，则抛出错误。
        
        Args:
            model_args: 模型配置参数，包含模型名称或路径
            script_args: GRPO参数，可能包含特定的分词器名称或路径
            
        Returns:
            分词器名称或路径，用于加载适当的分词器
            
        Raises:
            ValueError: 当无法从任何参数中解析出分词器名称时抛出
        """
        if script_args.tokenizer_name_or_path:
            return script_args.tokenizer_name_or_path
        if model_args.model_name_or_path:
            return model_args.model_name_or_path
        raise ValueError("无法解析分词器名称")

    def _dht_kwargs(self, grpo_args):
        """
        构建DHT关键字参数
        
        根据提供的GRPO参数构建用于初始化分布式哈希表(DHT)的关键字参数字典。
        包括初始对等节点、公共地址、主机地址和身份路径等配置。
        
        Args:
            grpo_args: GRPO参数，包含DHT初始化所需的网络配置
            
        Returns:
            DHT初始化的关键字参数字典，包含所有必要的网络配置选项
        """
        kwargs = {}
        initial_peers = grpo_args.initial_peers
        if initial_peers:
            kwargs["initial_peers"] = initial_peers

        if public_maddr := grpo_args.public_maddr:
            kwargs["announce_maddrs"] = [public_maddr]

        if host_maddr := grpo_args.host_maddr:
            kwargs["host_maddrs"] = [host_maddr]

        if identity_path := grpo_args.identity_path:
            kwargs["identity_path"] = identity_path

        return kwargs

    def _get_animal_name(self, peer_id):
        """
        从对等节点ID获取动物名称
        
        根据对等节点ID生成一个唯一的动物名称，用于在日志和输出中标识节点。
        这使得在分布式环境中更容易识别和区分不同的节点。
        
        Args:
            peer_id: 对等节点ID，用作生成动物名称的种子
            
        Returns:
            基于对等节点ID生成的唯一动物名称
        """
        animal_name = get_name_from_peer_id(peer_id)
        logger.info(f"🐱 你好 🐈 [{animal_name}] 🦮 [{peer_id}]!")
        return animal_name

    def setup_dht(self, grpo_args):
        """
        设置分布式哈希表(DHT)
        
        初始化DHT实例并配置网络连接。如果提供了初始对等节点，则加入现有的蜂群网络；
        否则，创建一个新的蜂群网络并成为协调者节点。同时为当前节点生成一个唯一的动物名称标识。
        
        Args:
            grpo_args: GRPO参数，包含DHT初始化所需的网络配置
            
        Returns:
            初始化的DHT实例，已连接到蜂群网络
        """
        initial_peers = grpo_args.initial_peers
        dht = hivemind.DHT(start=True, **self._dht_kwargs(grpo_args))
        if initial_peers:
            logger.info(f"🐝 正在加入蜂群，初始对等节点 = {initial_peers}")
        else:
            first_visible = str(dht.get_visible_maddrs()[0])
            logger.info(f"🤖 正在启动蜂群，地址为 {first_visible}")

        self.name = self._get_animal_name(str(dht.peer_id))
        return dht

    def run(
        self,
        model_args: ModelConfig,
        grpo_args: GRPOArguments,
        training_args: GRPOConfig,
        initial_datasets_fn: Callable[[], Tuple[Dataset, Dataset]],
        trainer_factory_fn: Callable = HivemindGRPOTrainer,
    ):
        """
        运行GRPO训练过程
        
        这是主要的执行方法，负责整个训练流程的协调。它执行以下步骤：
        1. 配置训练参数和批量大小
        2. 如果提供了HF令牌，登录Hugging Face Hub
        3. 加载分词器
        4. 通过Hivemind创建分布式哈希表(DHT)
        5. 加载和准备数据集
        6. 实例化模型
        7. 创建Hivemind节点（协调者或跟随者）
        8. 设置训练阶段数据
        9. 创建训练器实例
        10. 启动训练循环
        
        Args:
            model_args: 模型配置参数，包含模型名称和初始化选项
            grpo_args: GRPO参数，包含网络配置和训练设置
            training_args: 训练配置参数，包含学习率、批量大小等训练超参数
            initial_datasets_fn: 获取初始数据集的函数，返回训练集和测试集的元组
            trainer_factory_fn: 创建训练器的工厂函数，默认为HivemindGRPOTrainer
        """
        #########################
        # 记录参数
        #########################
        logger.debug(f"模型参数 {model_args}")
        logger.debug(f"训练/评估参数 {training_args}")

        # 设置批量大小，用于训练和生成
        batch_size = 2
        training_args.per_device_train_batch_size = batch_size
        training_args.num_generations = batch_size

        ############################
        # 如果需要，登录HF hub
        ############################
        # 如果提供了有效的Hugging Face令牌，则登录并配置推送权限
        if (grpo_args.hf_token not in [None, "None"]):
            training_args.push_to_hub_token = grpo_args.hf_token
            login(token=training_args.push_to_hub_token, add_to_git_credential=True)
            logger.info("已成功登录Hugging Face Hub")
        else:
            training_args.push_to_hub_token = None
            logger.info("未提供Hugging Face令牌，将不会推送模型到Hub")

        ################
        # 加载分词器
        ################
        # 根据配置加载适当的分词器，并确保设置了填充令牌
        tokenizer_name = self.get_tokenizer_name(model_args, grpo_args)
        logger.info(f"正在加载分词器: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("分词器没有填充令牌，已将EOS令牌设置为填充令牌")

        #########################
        # 通过Hivemind创建DHT
        #########################
        # 初始化分布式哈希表，建立节点间通信网络
        logger.info("正在初始化Hivemind分布式哈希表...")
        dht = self.setup_dht(grpo_args)

        #####################################
        # 加载数据集，准备和格式化
        #####################################
        # 调用提供的函数加载初始训练和测试数据集
        logger.info("正在加载和准备数据集...")
        train_dataset, test_dataset = initial_datasets_fn()
        logger.info(f"已加载训练数据集({len(train_dataset)}个样本)和测试数据集({len(test_dataset)}个样本)")

        #########################
        # 实例化模型
        #########################
        # 加载预训练模型用于GRPO训练
        model_name_or_path = model_args.model_name_or_path
        assert model_name_or_path, "必须提供模型名称或路径"
        logger.info(f"正在加载模型: {model_name_or_path}")
        model = self.get_model(training_args, model_name_or_path)

        # 根据是否有初始对等节点决定创建协调者节点还是跟随者节点
        initial_peers = grpo_args.initial_peers
        if initial_peers:
            logger.info("创建跟随者节点...")
            node = HivemindNode(model_name_or_path, str(dht.peer_id))
        else:
            logger.info("创建协调者节点...")
            node = HivemindNode.coordinator(model_name_or_path, str(dht.peer_id))

        # 设置训练阶段数据并创建训练器实例
        logger.info("正在设置训练阶段数据...")
        stage_data = gsm8k_stage_data(dht, node, train_dataset, test_dataset)
        stage_data.max_rounds = grpo_args.max_rounds
        logger.info(f"最大训练轮数设置为: {stage_data.max_rounds}")
        
        logger.info("正在创建训练器实例...")
        trainer = trainer_factory_fn(
            dht=dht,
            node=node,
            model=model,
            tokenizer=tokenizer,
            config=training_args,
            stage_data=stage_data,
            log_tag=self.name,
        )

        ###############
        # 训练循环
        ###############
        # 启动训练过程，记录开始时间和预期的训练周期数
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(
            f"开始训练 {current_time} 共 {training_args.num_train_epochs} 个周期"
        )
        logger.info(f"节点角色: {'协调者' if node.is_coordinator else '跟随者'}")
        trainer.train()
        logger.info(f"训练完成，结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
