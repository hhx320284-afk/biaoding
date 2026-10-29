"""PiPER X 关节反馈与 FK 只读接入；不会使能、运动或改写机械臂设置。

优先调用官方 get_joint_angles()、fk()：关节弧度，法兰位姿米／弧度。
https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/drivers/core/arm_driver_abstract.py
FK 与 get_flange_pose 的输出均为法兰到基座，固定轴 XYZ 欧拉角。
SDK 的 CAN 分包时间戳来自 python-can，为接收时的 Unix epoch 秒。
为排除 SDK 返回的半包零值，本适配器检查当前官方解析器的三组缓存；
只有 SDK 没有 FK 接口时才读取法兰反馈，不掩盖 FK 或关节数据错误。
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import threading
import time

from 手眼算法 import parse_pose


class RobotReadError(RuntimeError):
    """可直接在界面展示的机械臂接入错误。"""


def discover_can_interfaces() -> list[dict]:
    """只读枚举 SocketCAN；接口名不代表左／右臂，不更改系统配置。"""
    interfaces = []
    for interface in sorted(Path("/sys/class/net").iterdir()):
        try:
            if int((interface / "type").read_text().strip()) != 280:
                continue
            is_up = bool(int((interface / "flags").read_text().strip(), 16) & 1)
        except (OSError, ValueError):
            continue  # USB-CAN 枚举途中可能已经拔出。
        serial = ""
        try:
            serial = (interface / "device").resolve().parent.joinpath("serial").read_text().strip()
        except OSError:
            pass  # 非 USB 的 SocketCAN 接口没有 USB 序列号。
        label = f"{interface.name} · {'已启用' if is_up else '未启用'}"
        if serial:
            label += f" · 设备尾号 {serial[-8:]}"
        interfaces.append({"channel": interface.name, "is_up": is_up, "serial": serial, "label": label})
    return interfaces


class PiperReader:
    """仅适用于用户已确认的 AgileX PiPER X 和已启用的 Linux SocketCAN。

    connect() 最多等待 5 秒；read_pose() 返回平移为米的末端到基座矩阵。
    SDK 是可选依赖。不会自动安装 SDK、启用 CAN、设置 TCP 或发送控制指令。
    """

    _JOINT_PACKETS = ("joint_12", "joint_34", "joint_56")
    _FLANGE_PACKETS = ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
    _MAX_AGE = 0.5
    _MAX_PACKET_SPAN = 0.1
    _CONNECT_TIMEOUT = 5.0

    def __init__(self, channel: str = "can0", firmware: str = "default"):
        if (not isinstance(channel, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", channel.strip())):
            raise RobotReadError("请输入有效的 CAN 接口名称，例如 can0。")
        if firmware not in ("default", "v183", "v188", "v189"):
            raise RobotReadError("请选择与 PiPER X 控制器匹配的固件版本。")
        self.channel = channel.strip()
        self.firmware = firmware
        self._arm = None
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._worker = None

    def _check_channel(self) -> None:
        """在进入 SDK 前指出接口缺失／未启用，避免误报为反馈超时。"""
        interface = Path("/sys/class/net") / self.channel
        if not interface.exists():
            available = "、".join(item["channel"] for item in discover_can_interfaces()) or "无"
            raise RobotReadError(
                f"未找到 CAN 接口 {self.channel}，当前可用接口：{available}。"
                "请插好所选机械臂的 USB-CAN，再点击“刷新接口”核对。"
            )
        try:
            interface_type = int((interface / "type").read_text().strip())
            flags = int((interface / "flags").read_text().strip(), 16)
        except (OSError, ValueError) as exc:
            raise RobotReadError(f"无法读取 {self.channel} 的系统状态，请检查 USB-CAN 连接。") from exc
        if interface_type != 280:
            raise RobotReadError(f"{self.channel} 不是 SocketCAN 接口，请重新选择机械臂的 CAN 接口。")
        if not flags & 1:
            raise RobotReadError(
                f"CAN 接口 {self.channel} 尚未启用（DOWN），因此无法接收机械臂反馈。"
                "请先点击“启用所选 CAN”（PiPER X 使用 1 Mbps），再连接机械臂。"
            )

    @classmethod
    def _feedback(cls, arm, packet_names, getter_name, label, packet_ids):
        """复制六维反馈并核对全部分包，不能只使用 SDK 汇总包的时间戳。"""
        parser = getattr(arm, "_parser", None)
        getter = getattr(arm, getter_name, None)
        if parser is None or not callable(getter):
            raise RobotReadError(f"pyAgxArm 版本不兼容，无法核对完整{label}，请使用官方当前版本。")
        for _ in range(3):
            packets = [getattr(parser, name, None) for name in packet_names]
            if any(packet is None for packet in packets):
                raise RobotReadError(f"尚未收到完整{label}（需要 CAN {packet_ids} 三组反馈）。")
            stamps = tuple(float(packet.timestamp) for packet in packets)
            message = getter()
            values = tuple(float(value) for value in message.msg) if message is not None else ()
            current_packets = [getattr(parser, name, None) for name in packet_names]
            if (all(old is new for old, new in zip(packets, current_packets))
                    and stamps == tuple(float(packet.timestamp) for packet in current_packets)):
                break
        else:
            raise RobotReadError(f"{label}正在更新，请待机械臂停稳后重新读取。")
        now_epoch, now_mono = time.time(), time.monotonic()
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise RobotReadError(f"机械臂返回的{label}不完整或包含无效数值。")
        if not all(math.isfinite(stamp) and stamp > 0 for stamp in stamps):
            raise RobotReadError("机械臂反馈缺少有效接收时间戳，不能作为标定样本。")
        oldest, newest = min(stamps), max(stamps)
        age = now_epoch - oldest
        if now_epoch - newest < -0.05:
            raise RobotReadError("CAN 反馈时间戳晚于本机时间，请检查系统时钟。")
        if age > cls._MAX_AGE:
            raise RobotReadError(f"{label}已过期（{age:.2f} 秒），请检查机械臂反馈后重新读取。")
        if newest - oldest > cls._MAX_PACKET_SPAN:
            raise RobotReadError(f"{label}分包相差超过 100 毫秒，请检查 CAN 通信后重新读取。")
        return values, oldest, now_mono - max(0.0, age)

    @classmethod
    def _snapshot(cls, arm) -> dict:
        if arm.has_comm_error():
            raise RobotReadError("CAN 通信已中断，请检查机械臂电源、接线和接口状态。")
        fk = getattr(arm, "fk", None)
        if callable(fk):
            joints, timestamp, monotonic = cls._feedback(
                arm, cls._JOINT_PACKETS, "get_joint_angles", "关节角反馈", "0x2A5、0x2A6、0x2A7",
            )
            try:
                values = tuple(float(value) for value in fk(list(joints)))
            except Exception as exc:
                raise RobotReadError(f"PiPER X 正运动学 FK 计算失败：{exc}") from exc
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                raise RobotReadError("PiPER X FK 返回的法兰位姿不完整或包含无效数值。")
            source, source_label = "fk", "关节反馈 + FK"
        else:
            values, timestamp, monotonic = cls._feedback(
                arm, cls._FLANGE_PACKETS, "get_flange_pose", "法兰位姿", "0x2A2、0x2A3、0x2A4",
            )
            joints = None
            source, source_label = "controller_flange", "控制器法兰反馈（SDK 无 FK 接口）"
        transform = parse_pose(" ".join(str(value) for value in values), "rpy", "m", "rad")
        return {
            "T_base_end": transform.tolist(),
            "joint_angles": list(joints) if joints is not None else None,
            "joint_angle_unit": "rad",
            "pose_source": source,
            "pose_source_label": source_label,
            "monotonic": monotonic,
            "timestamp": timestamp,
            "frame": "flange",
            "model": "piper_x",
        }

    def connect(self) -> None:
        with self._lock:
            if self._arm is not None:
                self.read_pose()
                return
            if self._worker is not None and self._worker.is_alive():
                raise RobotReadError("上一次连接正在结束，请稍后再连接。")
            self._check_channel()
            cancel = self._cancel = threading.Event()
            ready = threading.Event()
            outcome = {}
            deadline = time.monotonic() + self._CONNECT_TIMEOUT

            def open_reader():
                arm = None
                transferred = False
                try:
                    try:
                        from pyAgxArm import AgxArmFactory, create_agx_arm_config
                    except ImportError as exc:
                        raise RobotReadError(
                            "当前运行环境未安装可用的 pyAgxArm SDK。请在本程序 Python 环境安装官方 SDK："
                            'python -m pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"。'
                            "安装完成后重新启动界面，再连接机械臂。"
                        ) from exc
                    config = create_agx_arm_config(
                        robot="piper_x", firmeware_version=self.firmware,
                        interface="socketcan", channel=self.channel,
                    )
                    arm = AgxArmFactory.create_arm(config)
                    if cancel.is_set():
                        return
                    arm.connect()
                    last_error = RobotReadError("尚未收到机械臂完整关节反馈。")
                    while not cancel.is_set() and time.monotonic() < deadline:
                        try:
                            self._snapshot(arm)
                        except RobotReadError as exc:
                            last_error = exc
                            outcome["last_error"] = exc
                            cancel.wait(0.02)
                            continue
                        with self._lock:
                            if not cancel.is_set():
                                self._arm = arm
                                transferred = True
                                outcome["ok"] = True
                        return
                    if not cancel.is_set():
                        raise last_error
                except Exception as exc:
                    outcome["error"] = exc
                finally:
                    if arm is not None and not transferred:
                        try:
                            arm.disconnect(join_timeout=0.1)
                        except Exception:
                            pass
                    ready.set()

            self._worker = threading.Thread(target=open_reader, name="PiPER关节只读连接", daemon=True)
            self._worker.start()
        if not ready.wait(max(0.0, deadline - time.monotonic())):
            with self._lock:
                cancel.set()
                if self._arm is not None:
                    return
            detail = outcome.get("last_error") or outcome.get("error") or "SDK 尚未完成连接"
            raise RobotReadError(
                f"连接 {self.channel} 超时（{self._CONNECT_TIMEOUT:g} 秒）：{detail}"
                "请核对所选机械臂、CAN 接线、供电和固件版本。"
            )
        if not outcome.get("ok"):
            error = outcome.get("error")
            if isinstance(error, RobotReadError):
                raise error
            if error is not None:
                raise RobotReadError(f"无法连接 CAN 接口 {self.channel}：{error}") from error
            raise RobotReadError("机械臂连接已取消。")

    def read_pose(self) -> dict:
        with self._lock:
            if self._arm is None:
                raise RobotReadError("尚未连接 PiPER X。")
            try:
                result = self._snapshot(self._arm)
                result["channel"] = self.channel
                return result
            except RobotReadError:
                raise
            except Exception as exc:
                raise RobotReadError(f"读取法兰位姿失败：{exc}") from exc

    def disconnect(self) -> None:
        self._cancel.set()
        with self._lock:
            arm, self._arm = self._arm, None
        if arm is not None:
            try:
                arm.disconnect(join_timeout=0.2)
            except Exception as exc:
                raise RobotReadError(f"断开机械臂连接失败：{exc}") from exc
