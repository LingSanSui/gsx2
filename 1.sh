#!/bin/bash

# 定义常量
SCRIPT_NAME="1.sh"
MAIN_SCRIPT="run_rl_swarm.sh"
CHECK_DIR="modal-login/temp-data"
REQUIRED_FILES=("userApiKey.json" "userData.json")
LIVE_LOG_FILE="/tmp/swarm_live_${$}.log" # 使用当前PID确保唯一性
RETRY_DELAY=10 # 重启延迟时间（秒）

# 定义清理函数
cleanup() {
    echo "正在清理..."
    rm -f "$LIVE_LOG_FILE"
    if [ -n "$MAIN_PID" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
        echo "正在终止 $MAIN_SCRIPT 进程 (PID: $MAIN_PID)..."
        # 使用 SIGTERM 尝试优雅关闭，给它一点时间
        kill "$MAIN_PID"
        sleep 2
        # 如果还活着，强制 kill (SIGKILL)
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            echo "进程未退出，强制终止 (SIGKILL)..."
            kill -9 "$MAIN_PID"
        fi
        wait "$MAIN_PID" 2>/dev/null
    fi
    echo "清理完成。"
}

# 设置退出时的清理
trap cleanup EXIT

# 检查本脚本是否已经在运行
check_self_running() {
    # 获取当前进程的PID
    local current_pid=$$
    # 使用更精确的模式匹配，排除当前进程
    local pids=$(pgrep -f "[^0-9]$SCRIPT_NAME" | grep -v "$current_pid")
    
    if [ -n "$pids" ]; then
        local count=$(echo "$pids" | wc -l)
        echo "检测到 $SCRIPT_NAME 已有 $count 个实例在运行，退出..."
        exit 1
    fi
}

# 检查主脚本是否已经在运行，如果是则终止它
check_main_script_running() {
    local pid=$(pgrep -f "$MAIN_SCRIPT")
    
    if [ -n "$pid" ]; then
        echo "检测到 $MAIN_SCRIPT 已在运行 (PID: $pid)，正在终止..."
        kill "$pid"
        sleep 2
        # 如果还活着，强制 kill
        if kill -0 "$pid" 2>/dev/null; then
            echo "进程未退出，强制终止 (SIGKILL)..."
            kill -9 "$pid"
        fi
        wait "$pid" 2>/dev/null
        echo "$MAIN_SCRIPT 已终止。"
    fi
}

# 检查并强制更新所需文件
check_files() {
    echo "开始检查并强制更新所需文件..."
    # 确保目标目录存在
    mkdir -p "$CHECK_DIR"

    for file in "${REQUIRED_FILES[@]}"; do
        local source_file="./$file"       # 源文件路径 (当前目录)
        local target_file="$CHECK_DIR/$file" # 目标文件路径

        # 1. 首先检查源文件是否存在
        if [ ! -f "$source_file" ]; then
            echo "错误: 源文件 '$source_file' 在当前目录未找到！脚本无法继续。"
            exit 1 # 如果源文件都没有，无法进行复制，必须退出
        fi

        # 2. 检查目标文件是否存在，如果存在则删除
        if [ -f "$target_file" ]; then
            echo "目标文件 '$target_file' 已存在，正在删除..."
            rm -f "$target_file" # 使用 -f 避免文件不存在时的错误提示
            if [ $? -ne 0 ]; then
                 echo "警告: 删除 '$target_file' 时遇到问题，但仍将尝试复制。"
            fi
        fi

        # 3. 执行复制操作
        echo "正在从 '$source_file' 复制到 '$target_file'..."
        cp "$source_file" "$target_file"
        # 检查复制是否成功
        if [ $? -ne 0 ]; then
            echo "错误: 复制文件 '$source_file' 到 '$target_file' 失败！脚本无法继续。"
            exit 1 # 复制失败是严重问题，脚本退出
        fi
        echo "文件 '$target_file' 已成功更新/创建。"

    done
    echo "文件检查与强制更新完成。"
}

# 主程序开始
echo "===== 启动 $MAIN_SCRIPT 监控脚本 ====="

# 检查本脚本是否已经在运行
check_self_running

# 检查主脚本是否已经在运行，如果是则终止它
check_main_script_running

# 主循环
while true; do
    # 检查并更新文件
    #check_files

    echo "-----------------------------------------------------"
    echo "准备启动 $MAIN_SCRIPT 进程..."
    echo "日志将实时显示下方，并同时写入 $LIVE_LOG_FILE"
    echo "脚本将监控输出中的 'Traceback' 或 'but this warning has only been added since PyTorch 2.4 (function operator())' 以触发重启..."
    echo "-----------------------------------------------------"

    # 确保日志文件存在
    touch "$LIVE_LOG_FILE"

    # 1. 后台运行主脚本，输出到 tee (屏幕 + 文件)
    ( . /workspace/rl-swarm/.venv/bin/activate && ./run_rl_swarm.sh ) > >(tee -a "$LIVE_LOG_FILE") 2>&1 &
    MAIN_PID=$!

    # 2. 启动日志监控 - 使用grep监控两个关键词
    tail -f --pid=$MAIN_PID "$LIVE_LOG_FILE" | grep --line-buffered -Ei "Traceback|but this warning has only been added since PyTorch 2\.4 \(function operator\(\)\)"
    GREP_EXIT_STATUS=$?

    # 3. 分析结果
    if kill -0 $MAIN_PID 2>/dev/null; then
        # 主进程仍然存活
        if [ $GREP_EXIT_STATUS -eq 0 ]; then
            # === Case 1: 检测到关键词 ===
            echo "-----------------------------------------------------"
            echo "[!] 在日志中检测到错误关键词!"
            echo "正在终止当前进程 (PID: $MAIN_PID) 以便重启..."
            kill $MAIN_PID
            wait $MAIN_PID 2>/dev/null
            echo "进程已终止。"
        else
            # === Case 2: 监控意外结束，但主进程仍在 ===
            echo "-----------------------------------------------------"
            echo "[Warning] 日志监控意外停止 (grep status: $GREP_EXIT_STATUS)，但主进程 (PID: $MAIN_PID) 仍在运行。"
            echo "为了恢复监控，将重启主进程..."
            kill $MAIN_PID
            wait $MAIN_PID 2>/dev/null
            echo "进程已终止。"
        fi
        # --- 重启逻辑 ---
        echo "将在 ${RETRY_DELAY} 秒后尝试重启..."
        echo "-----------------------------------------------------"
        sleep $RETRY_DELAY
        # 删除旧日志文件以便下次 touch 创建新的
        rm -f "$LIVE_LOG_FILE"
        continue

    else
        # === Case 3: 主进程已死 (且未检测到关键词) ===
        echo "-----------------------------------------------------"
        echo "主进程 (PID: $MAIN_PID) 已意外退出，且日志监控未检测到错误关键词。"
        echo "自动重启脚本将停止。"
        echo "-----------------------------------------------------"
        # 清理日志文件由 trap 完成
        exit 0
    fi
done