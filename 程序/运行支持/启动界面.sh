#!/bin/sh
# 项目根目录入口调用本脚本，复用本机已安装环境，不自动安装或更新依赖。
set -eu
support_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$support_dir/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
for python_bin in "${CAMERA_CALIBRATION_PYTHON:-}" "${XDG_CACHE_HOME:-$HOME/.cache}/vla-fisheye/venv/bin/python" "$(command -v python3 || true)" /usr/bin/python3 "$HOME/miniconda3/bin/python3"; do
    if [ -n "$python_bin" ] && [ -x "$python_bin" ] && "$python_bin" -B -c 'from PySide6.QtWidgets import QApplication; from PySide6.QtNetwork import QLocalServer; import cv2, numpy; assert hasattr(cv2, "aruco")' >/dev/null 2>&1; then
        exec "$python_bin" -B "$project_dir/程序/运行支持/启动程序.py" "$@"
    fi
done
printf '%s\n' '无法启动：未找到同时包含 PySide6、NumPy 和 ArUco 检测的 OpenCV Python 环境。所需组件见“程序/运行支持/运行依赖.txt”。' "日志位置：$project_dir/日志/运行.log" >&2
exit 1
