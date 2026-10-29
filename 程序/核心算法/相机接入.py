"""只读复用已有 V4L2 设备探测与隔离采集；本模块不提供照片保存接口。"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


# 受保护的旧项目只允许读取。此设置也随 multiprocessing spawn 继承，
# 防止第一次加载旧模块或工作进程加载时生成/更新该目录的字节码缓存。
sys.dont_write_bytecode = True
_SOURCE = Path(__file__).resolve().parents[3] / "鱼眼相机安装调试" / "程序" / "核心算法"
if not (_SOURCE / "相机设备控制.py").is_file() or not (_SOURCE / "相机采集保存.py").is_file():
    raise ImportError(f"找不到工作区已有相机后端：{_SOURCE}。请保留原项目目录。")

# 把本文件作为一个可导入的包入口。子模块具有真实可解析的模块名，
# spawn 子进程反序列化旧采集函数时会先导入本文件；旧代码中的相对导入
# 也自然落到同一只读目录。无需复制旧代码或给受保护目录新增 __init__.py。
__path__ = [str(_SOURCE)]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__
_devices = importlib.import_module(f"{__name__}.相机设备控制")
_capture = importlib.import_module(f"{__name__}.相机采集保存")

CameraError = _devices.CameraError
Device = _devices.Device
Mode = _devices.Mode
discover_devices = _devices.discover_devices
probe_device = _devices.probe_device
CaptureController = _capture.CaptureController

__all__ = ["CameraError", "Device", "Mode", "discover_devices", "probe_device", "CaptureController"]
