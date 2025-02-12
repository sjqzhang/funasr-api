#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查并激活虚拟环境
if [ -d "venv" ]; then
    echo "找到虚拟环境，正在激活..."
    source venv/bin/activate
else
    echo "警告: 未找到虚拟环境 (venv/)"
fi

# 检查并启动应用
if [ -f "app.py" ]; then
    echo "启动 app.py..."
    exec python app.py
elif [ -f "main.py" ]; then
    echo "启动 main.py..."
    exec python main.py
else
    echo "错误: 未找到 app.py 或 main.py"
    exit 1
fi 
