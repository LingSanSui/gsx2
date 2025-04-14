import json
import logging
from abc import ABC

import requests
from eth_account import Account
from web3 import Web3

# 这个模块提供了与区块链交互的工具函数，用于协调分布式训练过程

# Alchemy API URL，用于连接Gensyn测试网
ALCHEMY_URL = "https://gensyn-testnet.g.alchemy.com/public"

# Gensyn主网链ID
MAINNET_CHAIN_ID = 685685

# Swarm协调器合约版本和相关配置
SWARM_COORDINATOR_VERSION = "0.2"
SWARM_COORDINATOR_ABI_JSON = (
    f"hivemind_exp/contracts/SwarmCoordinator_{SWARM_COORDINATOR_VERSION}.json"
)
SWARM_COORDINATOR_CONTRACT = "0x2fC68a233EF9E9509f034DD551FF90A79a0B8F82"

# Modal代理URL，用于API调用
MODAL_PROXY_URL = "http://localhost:3000/api/"

# 配置日志记录器
logger = logging.getLogger(__name__)


class SwarmCoordinator(ABC):
    """Swarm协调器基类，提供与区块链合约交互的基本功能
    
    这个抽象基类定义了与Swarm协调器智能合约交互的接口和共享功能。
    子类需要实现具体的交易发送方法。
    """
    @staticmethod
    def coordinator_contract(web3: Web3):
        """创建协调器合约实例
        
        Args:
            web3: Web3实例，用于与区块链交互
            
        Returns:
            配置好的合约实例
        """
        with open(SWARM_COORDINATOR_ABI_JSON, "r") as f:
            contract_abi = json.load(f)["abi"]

        return web3.eth.contract(address=SWARM_COORDINATOR_CONTRACT, abi=contract_abi)

    def __init__(self, web3: Web3, **kwargs) -> None:
        """初始化Swarm协调器
        
        Args:
            web3: Web3实例，用于与区块链交互
            **kwargs: 传递给父类的额外参数
        """
        self.web3 = web3
        self.contract = SwarmCoordinator.coordinator_contract(web3)
        super().__init__(**kwargs)

    def register_peer(self, peer_id): ...

    def submit_winners(self, round_num, winners): ...

    def get_bootnodes(self):
        """获取引导节点列表
        
        Returns:
            合约中注册的引导节点列表
        """
        return self.contract.functions.getBootnodes().call()

    def get_round_and_stage(self):
        """获取当前轮次和阶段
        
        使用批处理请求同时获取当前轮次和阶段，减少网络调用次数。
        
        Returns:
            当前轮次和阶段的元组
        """
        with self.web3.batch_requests() as batch:
            batch.add(self.contract.functions.currentRound())
            batch.add(self.contract.functions.currentStage())
            round_num, stage_num = batch.execute()

        return round_num, stage_num


class WalletSwarmCoordinator(SwarmCoordinator):
    """使用钱包私钥的Swarm协调器实现
    
    这个类使用以太坊钱包私钥直接与区块链交互，发送交易。
    适用于有完全控制权的环境。
    """
    def __init__(self, private_key: str, **kwargs) -> None:
        """初始化钱包Swarm协调器
        
        Args:
            private_key: 以太坊钱包私钥
            **kwargs: 传递给父类的额外参数
        """
        super().__init__(**kwargs)
        self.account = setup_account(self.web3, private_key)

    def _default_gas(self):
        """获取默认的gas设置
        
        Returns:
            包含gas和gasPrice的字典
        """
        return {
            "gas": 2000000,
            "gasPrice": self.web3.to_wei("1", "gwei"),
        }

    def register_peer(self, peer_id):
        """注册对等节点
        
        Args:
            peer_id: 要注册的对等节点ID
        """
        send_chain_txn(
            self.web3,
            self.account,
            lambda: self.contract.functions.registerPeer(peer_id).build_transaction(
                self._default_gas()
            ),
        )

    def submit_winners(self, round_num, winners):
        """提交轮次获胜者
        
        Args:
            round_num: 轮次编号
            winners: 获胜者列表
        """
        send_chain_txn(
            self.web3,
            self.account,
            lambda: self.contract.functions.submitWinners(
                round_num, winners
            ).build_transaction(self._default_gas()),
        )


class ModalSwarmCoordinator(SwarmCoordinator):
    """使用Modal API的Swarm协调器实现
    
    这个类通过Modal API与区块链交互，适用于无法直接访问区块链的环境。
    使用API代理发送交易请求。
    """
    def __init__(self, org_id: str, **kwargs) -> None:
        """初始化Modal Swarm协调器
        
        Args:
            org_id: 组织ID，用于API认证
            **kwargs: 传递给父类的额外参数
        """
        self.org_id = org_id
        super().__init__(**kwargs)

    def register_peer(self, peer_id):
        """通过API注册对等节点
        
        Args:
            peer_id: 要注册的对等节点ID
        """
        try:
            send_via_api(self.org_id, "register-peer", {"peerId": peer_id})
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 500:
                raise

            logger.info("调用register-peer端点时出现未知错误！继续执行。")
            # TODO: 验证实际合约错误。
            # logger.info(f"对等节点ID [{peer_id}] 已经注册！继续执行。")

    def submit_winners(self, round_num, winners):
        """通过API提交轮次获胜者
        
        Args:
            round_num: 轮次编号
            winners: 获胜者列表
        """
        try:
            args = (
                self.org_id,
                "submit-winner",
                {"roundNumber": round_num, "winners": winners},
            )
            send_via_api(
                *args
            )
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code != 500:
                raise

            logger.info("调用submit-winner端点时出现未知错误！继续执行。")
            # TODO: 验证实际合约错误。
            # logger.info("本轮次的获胜者已提交！继续执行。")


def send_via_api(org_id, method, args):
    """通过API发送请求
    
    Args:
        org_id: 组织ID，用于API认证
        method: API方法名称
        args: 请求参数
        
    Returns:
        API响应的JSON数据
        
    Raises:
        requests.exceptions.HTTPError: 当HTTP请求失败时
    """
    # 构建URL和负载
    url = MODAL_PROXY_URL + method
    payload = {"orgId": org_id} | args

    # 发送POST请求
    response = requests.post(url, json=payload)
    response.raise_for_status()  # 对HTTP错误抛出异常
    return response.json()


def setup_web3() -> Web3:
    """设置Web3连接到Gensyn测试网
    
    Returns:
        配置好的Web3实例
        
    Raises:
        Exception: 当无法连接到测试网时
    """
    # 检查测试网连接
    web3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
    if web3.is_connected():
        logger.info("✅ 已连接到Gensyn测试网")
    else:
        raise Exception("无法连接到Gensyn测试网")
    return web3


def setup_account(web3: Web3, private_key) -> Account:
    """设置以太坊账户
    
    Args:
        web3: Web3实例
        private_key: 账户私钥
        
    Returns:
        配置好的Account实例
    """
    # 检查钱包余额
    account = web3.eth.account.from_key(private_key)
    balance = web3.eth.get_balance(account.address)
    eth_balance = web3.from_wei(balance, "ether")
    logger.info(f"💰 钱包余额: {eth_balance} ETH")
    return account


def send_chain_txn(
    web3: Web3, account: Account, txn_factory, chain_id=MAINNET_CHAIN_ID
):
    """发送区块链交易
    
    Args:
        web3: Web3实例
        account: 账户实例
        txn_factory: 交易构建工厂函数
        chain_id: 链ID，默认为MAINNET_CHAIN_ID
        
    Returns:
        None
    """
    checksummed = Web3.to_checksum_address(account.address)
    txn = txn_factory() | {
        "chainId": chain_id,
        "nonce": web3.eth.get_transaction_count(checksummed),
    }

    # 签名交易
    signed_txn = web3.eth.account.sign_transaction(txn, private_key=account.key)

    # 发送交易
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    logger.info(f"已发送交易，哈希值: {web3.to_hex(tx_hash)}")
