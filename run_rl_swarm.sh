#!/bin/bash

# 通用参数设置
# 设置根目录为当前工作目录
ROOT=$PWD

# 导出环境变量，这些变量将在脚本中使用
export PUB_MULTI_ADDRS       # 公共多地址
export PEER_MULTI_ADDRS      # 对等节点多地址
export HOST_MULTI_ADDRS      # 主机多地址
export IDENTITY_PATH         # 身份路径
export CONNECT_TO_TESTNET    # 是否连接到测试网
export ORG_ID                # 组织ID
export HF_HUB_DOWNLOAD_TIMEOUT=120  # HuggingFace下载超时时间（2分钟）

# 检查是否提供了公共多地址，否则设置为默认值
# 公共多地址用于让其他节点连接到你的节点
DEFAULT_PUB_MULTI_ADDRS=""
PUB_MULTI_ADDRS=${PUB_MULTI_ADDRS:-$DEFAULT_PUB_MULTI_ADDRS}

# 检查是否提供了对等节点多地址，否则设置为默认值
# 这是Gensyn协调器节点的地址，你的节点将连接到这个节点
DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ" # gensyn协调器节点
PEER_MULTI_ADDRS=${PEER_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

# 检查是否提供了主机多地址，否则设置为默认值
# 这是你的节点将监听的地址和端口
DEFAULT_HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_HOST_MULTI_ADDRS}

# RSA私钥的路径。如果此路径不存在，将创建一个新的密钥对。
# 如果你想要一个新的PeerID，请删除此文件。
DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

# 设置终端颜色
GREEN_TEXT="\033[32m"  # 绿色文本
RESET_TEXT="\033[0m"   # 重置文本颜色

# 定义一个函数用于输出绿色文本
echo_green() {
    echo -e "$GREEN_TEXT$1$RESET_TEXT"
}

# 询问用户是否要连接到测试网
# while true; do
#     echo -en $GREEN_TEXT
#     read -p ">> 您想连接到测试网吗？[Y/n] " yn
#     echo -en $RESET_TEXT
#     yn=${yn:-Y}  # 如果用户直接按Enter，默认为"Y"
#     case $yn in
#         [Yy]*)  CONNECT_TO_TESTNET=True && break ;;
#         [Nn]*)  CONNECT_TO_TESTNET=False && break ;;
#         *)  echo ">>> 请回答yes或no。" ;;
#     esac
# done
CONNECT_TO_TESTNET=True

# 如果用户选择连接到测试网
if [ "$CONNECT_TO_TESTNET" = "True" ]; then
    # 运行modal_login服务器
    echo "请登录以创建以太坊服务器钱包"
    cd modal-login
    # 检查yarn命令是否存在；如果不存在，安装Yarn
    source ~/.bashrc
    if ! command -v yarn > /dev/null 2>&1; then
        # 检测Ubuntu（包括WSL Ubuntu）并相应地安装Yarn
        if grep -qi "ubuntu" /etc/os-release 2>/dev/null || uname -r | grep -qi "microsoft"; then
            echo "检测到Ubuntu或WSL Ubuntu。通过apt安装Yarn..."
            curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
            echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
            sudo apt update && sudo apt install -y yarn
        else
            echo "Yarn未安装。正在安装Yarn..."
            curl -o- -L https://yarnpkg.com/install.sh | sh
            echo 'export PATH="$HOME/.yarn/bin:$HOME/.config/yarn/global/node_modules/.bin:$PATH"' >> ~/.bashrc
            source ~/.bashrc
        fi
    fi
    # 安装依赖并在后台启动开发服务器
    yarn install
    yarn dev > /dev/null 2>&1 & # 在后台运行并抑制输出

    SERVER_PID=$!  # 存储进程ID
    sleep 5
    open http://localhost:3000  # 打开浏览器访问登录页面
    cd ..

    # 等待用户数据文件创建
    echo_green ">> 等待modal userData.json文件创建..."
    while [ ! -f "modal-login/temp-data/userData.json" ]; do
        sleep 5  # 每5秒检查一次
    done
    echo "找到userData.json。继续进行..."

    # 从用户数据文件中提取组织ID
    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo "您的ORG_ID设置为: $ORG_ID"

    # 等待API密钥被客户端激活
    echo "等待API密钥激活..."
    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        if [[ "$STATUS" == "activated" ]]; then
            echo "API密钥已激活！继续进行..."
            break
        else
            echo "等待API密钥激活..."
            sleep 5
        fi
    done

    # 定义清理服务器进程的函数
    cleanup() {
        echo_green ">> 关闭服务器..."
        kill $SERVER_PID
        #rm -r modal-login/temp-data/*.json
        exit 0
    }

    # 设置捕获Ctrl+C并调用cleanup函数
    trap cleanup INT
fi

# 定义安装pip依赖的函数
pip_install() {
    pip install --disable-pip-version-check -q -r "$1"
}

# 安装必要的依赖
echo_green ">> 获取依赖..."
pip_install "$ROOT"/requirements-hivemind.txt  # 安装Hivemind依赖
pip_install "$ROOT"/requirements.txt         # 安装其他依赖

# 根据系统环境选择合适的配置文件
if ! command -v nvidia-smi &> /dev/null; then
    # 如果没有NVIDIA GPU
    CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
elif [ -n "$CPU_ONLY" ]; then
    # 或者用户指定只使用CPU
    CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
else
    # 检测到NVIDIA GPU
    pip_install "$ROOT"/requirements_gpu.txt  # 安装GPU相关依赖
    CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
fi

echo_green ">> 完成！"
HUGGINGFACE_ACCESS_TOKEN="None"
# 处理HuggingFace令牌
# if [ -n "${HF_TOKEN}" ]; then # 检查HF_TOKEN是否已设置，如果已设置则使用它，否则提示用户选择
#     HUGGINGFACE_ACCESS_TOKEN=${HF_TOKEN}
# else
#     echo -en $GREEN_TEXT
#     read -p ">> 您想将在RL集群中训练的模型推送到Hugging Face Hub吗？[y/N] " yn
#     echo -en $RESET_TEXT
#     yn=${yn:-N} # 如果用户直接按Enter，默认为"N"
#     case $yn in
#         [Yy]*) read -p "输入您的Hugging Face访问令牌: " HUGGINGFACE_ACCESS_TOKEN ;;
#         [Nn]*) HUGGINGFACE_ACCESS_TOKEN="None" ;;
#         *) echo ">>> 未给出答案，因此不会将模型推送到Hugging Face Hub" && HUGGINGFACE_ACCESS_TOKEN="None" ;;
#     esac
# fi

echo_green ">> 祝您在集群中好运！"

# 根据是否连接测试网启动训练
if [ -n "$ORG_ID" ]; then
    # 使用组织ID连接到测试网
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --modal_org_id "$ORG_ID" \
        --config "$CONFIG_PATH"
else
    # 不连接测试网，使用本地配置
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --public_maddr "$PUB_MULTI_ADDRS" \
        --initial_peers "$PEER_MULTI_ADDRS" \
        --host_maddr "$HOST_MULTI_ADDRS" \
        --config "$CONFIG_PATH"
fi

wait  # 保持脚本运行直到按Ctrl+C
