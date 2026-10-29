"""PiPER X 法兰反馈接入；显式运动入口与采集读取分开。

沿用工作区手眼标定的分包核对和关节反馈＋FK 方案；该项目接入文件存在
独立导入错误，因此本模块不导入或修改它。旋转转换复用本项目四点算法。
连接和采集不会使能或运动。仅明确调用 enable_for_motion、
move_flange_linear、stop_motion 或经操作者确认的 recover_teaching 才发送对应指令，
不自动复位或改写 TCP。

SDK 核对版本：agilexrobotics/pyAgxArm @
d90e234ba8e408cfa964b38aa9120b3131fc9f85。
官方 connect 启动接收／FPS 线程；get_joint_angles 与 get_flange_pose
只读取接收缓存，fk 在本机计算。SDK 导入和连接均延迟到 connect。
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

from 四点标定 import rotation_matrix


SDK_REVISION = "d90e234ba8e408cfa964b38aa9120b3131fc9f85"
# 用户明确恢复界面执行；仅解除排障暂停，不绕过设备状态与运动偏差检查。
MOTION_CONTROL_HOLD_REASON = ""


class RobotReadError(RuntimeError):
    """可以直接显示给操作者的读取错误。"""


class _ReceiveThreadSnapshots:
    """在 SDK 接收线程完成解码后复制三包，读线程只取得不可变副本。"""

    _GROUPS = {
        "get_joint_angles": (("joint_12", "joint_34", "joint_56"), (0x2A5, 0x2A6, 0x2A7)),
        "get_flange_pose": (("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"), (0x2A2, 0x2A3, 0x2A4)),
    }

    def __init__(self, arm):
        self._parser = getattr(arm, "_parser", None)
        self._getters = {name: getattr(arm, name, None) for name in self._GROUPS}
        self._lock = threading.Lock()
        self._latest = {}
        self._pending = {name: {} for name in self._GROUPS}
        self._last_seen = {}
        self._errors = {}
        self._sequence = {name: 0 for name in self._GROUPS}
        self._motion_status = None
        self._motor_enabled = {}
        self._motion_error = None
        self._routes = {packet_id: (getter, packet_name)
                        for getter, (names, ids) in self._GROUPS.items()
                        for packet_name, packet_id in zip(names, ids)}

    def receive(self, frame):
        """必须追加在 SDK 解析回调之后；异常留给读线程报告，不中断收包。"""
        identifier = getattr(frame, "arbitration_id", None)
        if identifier == 0x2A1 or identifier in range(0x261, 0x267):
            self._receive_motion(frame)
            return
        route = self._routes.get(getattr(frame, "arbitration_id", None))
        if route is None:
            return
        getter_name, packet_name = route
        with self._lock:
            if getter_name in self._errors:
                return
            try:
                if (getattr(frame, "is_error_frame", False)
                        or getattr(frame, "is_remote_frame", False)
                        or getattr(frame, "is_extended_id", False)
                        or getattr(frame, "dlc", None) != 8 or len(frame.data) != 8):
                    raise ValueError("收到无效的关节／法兰反馈报文，需要完整的 8 字节标准数据帧")
                timestamp = float(frame.timestamp)
                if not math.isfinite(timestamp) or timestamp <= 0:
                    raise ValueError("反馈报文缺少有效接收时间戳")
                previous = self._last_seen.get(packet_name)
                if previous is not None:
                    if timestamp < previous:
                        raise ValueError("反馈报文时间倒退，请重新连接")
                    if timestamp == previous:
                        return  # 同一包的重读不能推进分组或稳定计时。
                names = self._GROUPS[getter_name][0]
                packet = getattr(self._parser, packet_name, None)
                if packet is None or float(packet.timestamp) != timestamp:
                    raise ValueError("SDK 解码回调顺序不兼容，不能取得完整反馈")
                self._last_seen[packet_name] = timestamp
                pending = self._pending[getter_name]
                pending[packet_name] = timestamp
                if len(pending) != len(names):
                    return
                getter = self._getters[getter_name]
                if not callable(getter):
                    raise ValueError("SDK 缺少所需反馈读取接口")
                # 此回调与解析器串行运行；下一帧尚未解码，getter 无并发写入。
                message = getter()
                values = tuple(float(value) for value in message.msg) if message is not None else ()
                stamps = tuple(float(getattr(self._parser, name).timestamp) for name in names)
                if stamps != tuple(pending[name] for name in names):
                    raise ValueError("反馈分组与已接收报文不一致")
                if len(values) != 6 or not all(math.isfinite(value) for value in values):
                    raise ValueError("反馈必须包含 6 个有限数值")
                self._sequence[getter_name] += 1
                self._latest[getter_name] = (values, stamps, self._sequence[getter_name])
                pending.clear()  # 下次发布前，三个分包都必须再次收到新报文。
            except Exception as exc:
                self._errors[getter_name] = str(exc)
                self._pending[getter_name].clear()

    def _receive_motion(self, frame):
        with self._lock:
            try:
                if (frame.is_error_frame or frame.is_remote_frame or frame.is_extended_id
                        or frame.dlc != 8 or len(frame.data) != 8):
                    raise ValueError("运动状态报文格式无效")
                stamp = float(frame.timestamp)
                if not math.isfinite(stamp) or stamp <= 0:
                    raise ValueError("运动状态时间戳无效")
                if frame.arbitration_id == 0x2A1:
                    packet = self._parser.arm_status
                    if float(packet.timestamp) != stamp:
                        raise ValueError("运动状态解码顺序不一致")
                    if self._motion_status is not None and stamp < self._motion_status[0]:
                        raise ValueError("运动状态时间倒退")
                    msg = packet.msg
                    self._motion_status = (stamp, int(msg.ctrl_mode), int(msg.arm_status),
                                           int(msg.motion_status), int(msg.err_code), int(msg.teach_status),
                                           int(msg.mode_feedback))
                else:
                    index = frame.arbitration_id - 0x260
                    packet = getattr(self._parser, f"driver_state_{index}")
                    if float(packet.timestamp) != stamp:
                        raise ValueError("电机状态解码顺序不一致")
                    previous = self._motor_enabled.get(index)
                    if previous is not None and stamp < previous[0]:
                        raise ValueError("电机状态时间倒退")
                    self._motor_enabled[index] = (stamp, bool(packet.msg.foc_status.driver_enable_status))
            except Exception as exc:
                self._motion_error = str(exc)

    def motion_read(self):
        with self._lock:
            if self._motion_error:
                raise RobotReadError(f"运动状态无效：{self._motion_error}。")
            if self._motion_status is None or len(self._motor_enabled) != 6:
                raise RobotReadError("尚未收到完整控制器状态及六关节使能反馈，不能执行运动。")
            stamp, ctrl, status, moving, error, teaching, mode = self._motion_status
            motors = tuple(self._motor_enabled[i] for i in range(1, 7))
            return {"ctrl_mode": ctrl, "arm_status": status, "motion_status": moving,
                    "err_code": error, "teach_status": teaching, "mode_feedback": mode,
                    "enabled": tuple(item[1] for item in motors),
                    "packet_timestamps": (stamp,) + tuple(item[0] for item in motors)}

    def read(self, getter_name, label, packet_ids):
        with self._lock:
            if getter_name in self._errors:
                raise RobotReadError(f"{label}原子快照失败：{self._errors[getter_name]}。")
            snapshot = self._latest.get(getter_name)
            if snapshot is None:
                raise RobotReadError(f"尚未收到完整{label}（需要 CAN {packet_ids} 三组新反馈）。")
            return snapshot


class PiperReader:
    """读取 base_from_flange；平移为米、姿态矩阵为右手正交旋转。

    connect 最多等待 5 秒；read_pose 和 disconnect 可在后台线程调用。
    本模块不自动配置接口或扫描设备，只连接操作者明确填写的接口。
    """

    _JOINT_PACKETS = ("joint_12", "joint_34", "joint_56")
    _FLANGE_PACKETS = ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
    _MAX_AGE = 0.25
    _MAX_PACKET_SPAN = 0.1

    def __init__(self, channel: str = "can1", firmware: str = "default"):
        if (not isinstance(channel, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", channel.strip())):
            raise RobotReadError("请输入有效的 CAN 接口名称，例如右臂 can1。")
        if firmware not in ("default", "v183", "v188", "v189"):
            raise RobotReadError("请选择与 PiPER X 控制器匹配的固件版本。")
        self.channel = channel.strip()
        self.firmware = firmware
        self._arm = None
        self._feedback_state = None
        self._motion_locked = False
        self._motion_active = False
        self._motion_phase = "idle"
        self._prepared_monotonic = None
        self._prepared_snapshot = None
        self._motion_send_attempts = 0
        self._motion_sent_frames = 0
        self._recovery_active = False
        self._preparation_active = False
        self._stop_sequence = 0
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._worker = None
        self._epoch_to_monotonic = time.monotonic() - time.time()
        self._diagnostics = None

    def attach_diagnostics(self, recorder):
        """仅绑定内存诊断对象；不连接或启动记录，不改变设备状态。"""
        self._diagnostics = recorder

    def diagnostic_context(self):
        """只读现有缓存，保存版本和两路位姿；不向设备查询或发送指令。"""
        context = {"model": "piper_x", "channel": self.channel,
                   "firmware_selection": self.firmware, "sdk_revision": SDK_REVISION,
                   "python_version": sys.version, "automatic_motion_hold": MOTION_CONTROL_HOLD_REASON}
        context["source_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("机械臂采集.py", "探针运动.py", "控制诊断.py")}
        with self._lock:
            for name, read in (("fk_snapshot", lambda: self._snapshot(self._arm, self._feedback_state)),
                               ("controller_status", self.motion_status),
                               ("send_state", self.motion_send_state)):
                try:
                    context[name] = read()
                except Exception as exc:
                    context[name + "_error"] = str(exc)
            try:
                values, _, _, stamps, sequence = self._feedback(
                    self._feedback_state, "get_flange_pose", "控制器原生法兰反馈", "0x2A2、0x2A3、0x2A4")
                context["controller_flange"] = {"pose_m_rad": list(values),
                    "packet_timestamps": list(stamps), "sequence": sequence}
            except Exception as exc:
                context["controller_flange_error"] = str(exc)
        return context

    def diagnostic_event(self, label, details=None, *, trigger=False):
        recorder = self._diagnostics
        if recorder is not None:
            try:
                recorder.event(label, details, trigger=trigger)
            except Exception:
                # 记录故障不得阻断已有的急停／恢复发送流程。
                logging.getLogger(__name__).exception("控制诊断事件记录失败")

    def _reject_command(self, label, reason):
        self.diagnostic_event(label, {"reason": reason, "phase": self._motion_phase})
        raise RobotReadError(reason)

    def _check_channel(self):
        interface = Path("/sys/class/net") / self.channel
        if not interface.exists():
            raise RobotReadError(
                f"未找到 CAN 接口 {self.channel}。请连接机械臂的 USB-CAN，"
                f"确认右臂对应的 SocketCAN 接口已按厂商说明配置为 {self.channel}，再重新连接。"
            )
        try:
            interface_type = int((interface / "type").read_text().strip())
            flags = int((interface / "flags").read_text().strip(), 16)
        except (OSError, ValueError) as exc:
            raise RobotReadError(f"无法读取 {self.channel} 的系统状态，请检查 SocketCAN 配置。") from exc
        if interface_type != 280:  # Linux ARPHRD_CAN
            raise RobotReadError(f"{self.channel} 不是 CAN 接口，请填写已配置的 SocketCAN 接口名。")
        if not flags & 1:  # Linux IFF_UP
            raise RobotReadError(
                f"CAN 接口 {self.channel} 尚未启用。请按厂商说明核对波特率并启用接口，再重新连接。"
            )

    def _feedback(self, state, getter_name, label, packet_ids):
        """只读已发布的完整副本，拒绝过期或跨度过大的分组。"""
        values, stamps, sequence = state.read(getter_name, label, packet_ids)
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise RobotReadError(f"机械臂返回的{label}不完整或包含无效数值。")
        if not all(math.isfinite(stamp) and stamp > 0 for stamp in stamps):
            raise RobotReadError("机械臂反馈缺少有效接收时间戳，不能作为标定样本。")
        oldest, newest = min(stamps), max(stamps)
        now_epoch, now_mono = time.time(), time.monotonic()
        # 固定时钟映射保证重复读取同一反馈得到相同 monotonic；不能把重复包算作停稳时长。
        monotonic = oldest + self._epoch_to_monotonic
        if abs((now_mono - now_epoch) - self._epoch_to_monotonic) > 0.05:
            raise RobotReadError("系统时钟发生变化，请断开后重新连接机械臂，再重新确认触点。")
        if now_epoch - newest < -0.05:
            raise RobotReadError("CAN 反馈时间戳晚于本机时间，请检查系统时钟后重新连接。")
        age = now_mono - monotonic
        if age > self._MAX_AGE:
            raise RobotReadError(f"{label}已过期（{age:.2f} 秒），请检查 CAN 接线和机械臂反馈。")
        if newest - oldest > self._MAX_PACKET_SPAN:
            raise RobotReadError(f"{label}分包相差超过 100 毫秒，请检查 CAN 通信后重新读取。")
        return values, oldest, monotonic, stamps, sequence

    def _snapshot(self, arm, state):
        if arm.has_comm_error():
            raise RobotReadError("CAN 通信已中断，请检查机械臂电源、接线和接口状态。")
        fk = getattr(arm, "fk", None)
        if callable(fk):
            joints, timestamp, monotonic, stamps, sequence = self._feedback(
                state, "get_joint_angles", "关节角反馈", "0x2A5、0x2A6、0x2A7",
            )
            try:
                values = tuple(float(value) for value in fk(list(joints)))
            except Exception as exc:
                raise RobotReadError(f"PiPER X 正运动学 FK 计算失败：{exc}") from exc
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                raise RobotReadError("PiPER X FK 返回的法兰位姿不完整或包含无效数值。")
            source, source_label = "fk", "关节反馈 + FK"
        else:
            values, timestamp, monotonic, stamps, sequence = self._feedback(
                state, "get_flange_pose", "法兰位姿", "0x2A2、0x2A3、0x2A4",
            )
            joints = None
            source, source_label = "controller_flange", "控制器法兰反馈（SDK 无 FK 接口）"
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix(np.rad2deg(values[3:]), "rpy_deg")
        transform[:3, 3] = values[:3]
        return {
            "T_base_flange": transform.tolist(),
            "transform_direction": "base_from_flange",
            "translation_unit": "m",
            "joint_angles": list(joints) if joints is not None else None,
            "joint_angle_unit": "rad",
            "pose_source": source,
            "pose_source_label": source_label,
            "monotonic": monotonic,
            "timestamp": timestamp,
            "packet_timestamps": list(stamps),
            "feedback_snapshot": "receive_thread_group",
            "feedback_sequence": sequence,
            "sdk_revision": SDK_REVISION,
            "frame": "flange",
            "model": "piper_x",
            "channel": self.channel,
        }

    def _checked_motion_pose(self):
        """运动前交叉核对控制器原生法兰与关节 FK；不改变标定采集的读取路径。"""
        with self._lock:
            if self._arm is None or self._feedback_state is None:
                raise RobotReadError("右臂尚未连接，不能核对运动起点。")
            snapshot = self._snapshot(self._arm, self._feedback_state)
            if snapshot["pose_source"] != "fk":
                raise RobotReadError("缺少关节 FK，不能交叉核对控制器法兰位置，拒绝运动。")
            values, _, _, stamps, _ = self._feedback(
                self._feedback_state, "get_flange_pose", "控制器原生法兰反馈", "0x2A2、0x2A3、0x2A4")
            all_stamps = (*snapshot["packet_timestamps"], *stamps)
            if max(all_stamps) - min(all_stamps) > self._MAX_PACKET_SPAN:
                raise RobotReadError("关节与控制器法兰反馈相差超过 100 毫秒，拒绝使用不同步的运动起点。")
            controller = np.eye(4)
            controller[:3, :3] = rotation_matrix(np.rad2deg(values[3:]), "rpy_deg")
            controller[:3, 3] = values[:3]
            local = np.asarray(snapshot["T_base_flange"], dtype=float)
            distance = float(np.linalg.norm(controller[:3, 3] - local[:3, 3]))
            rotation = local[:3, :3].T @ controller[:3, :3]
            angle = math.acos(float(np.clip((np.trace(rotation) - 1.) / 2., -1., 1.)))
            if distance > .0001 or angle > math.radians(.05):
                raise RobotReadError(
                    f"控制器法兰与本地 FK 不一致：位置差 {distance * 1000:.3f} mm、"
                    f"姿态差 {math.degrees(angle):.3f}°，拒绝发送运动指令。")
            snapshot["controller_T_base_flange"] = controller.tolist()
            snapshot["controller_packet_timestamps"] = list(stamps)
            return snapshot

    def connect(self):
        with self._lock:
            if self._arm is not None:
                self.read_pose()
                return
            if self._worker is not None and self._worker.is_alive():
                raise RobotReadError("上一次连接正在结束，请稍后再连接。")
            cancel = self._cancel = threading.Event()
            ready = threading.Event()
            outcome = {}
            deadline = time.monotonic() + 5.0
            self._epoch_to_monotonic = time.monotonic() - time.time()

            def open_reader():
                arm = None
                transferred = False
                try:
                    try:
                        from pyAgxArm import AgxArmFactory, create_agx_arm_config
                    except ImportError as exc:
                        raise RobotReadError(
                            "当前 Python 环境缺少可用的 pyAgxArm SDK。请按使用说明安装到 "
                            f"{sys.executable}，安装后重新启动界面。"
                        ) from exc
                    self._check_channel()
                    config = create_agx_arm_config(
                        robot="piper_x", firmeware_version=self.firmware,
                        interface="socketcan", channel=self.channel,
                    )
                    arm = AgxArmFactory.create_arm(config)
                    if cancel.is_set():
                        return
                    register = getattr(getattr(arm, "_ctx", None), "register_parser_packet_fun", None)
                    if not callable(register) or getattr(arm, "_parser", None) is None:
                        raise RobotReadError("pyAgxArm 不支持接收线程完整快照，请安装使用说明中的 SDK 版本。")
                    state = _ReceiveThreadSnapshots(arm)
                    # SDK 在构造 arm 时先注册解析器；追加回调必须早于启动接收线程。
                    register(state.receive)
                    arm.connect()
                    last_error = RobotReadError("尚未收到机械臂完整关节反馈。")
                    while not cancel.is_set() and time.monotonic() < deadline:
                        try:
                            self._snapshot(arm, state)
                        except RobotReadError as exc:
                            last_error = exc
                            cancel.wait(0.02)
                            continue
                        with self._lock:
                            if not cancel.is_set():
                                self._arm = arm
                                self._feedback_state = state
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

            self._worker = threading.Thread(target=open_reader, name="PiPER只读连接", daemon=True)
            self._worker.start()
        if not ready.wait(max(0.0, deadline - time.monotonic())):
            with self._lock:
                cancel.set()
                if self._arm is not None:
                    return
            raise RobotReadError("连接超时（5 秒）：未收到完整新鲜反馈，请检查 CAN、电源和固件选择。")
        if not outcome.get("ok"):
            error = outcome.get("error")
            if isinstance(error, RobotReadError):
                raise error
            if error is not None:
                raise RobotReadError(f"无法连接 CAN 接口 {self.channel}：{error}") from error
            raise RobotReadError("机械臂连接已取消。")

    def read_pose(self):
        with self._lock:
            if self._arm is None:
                raise RobotReadError("尚未连接 PiPER X，请先连接机械臂。")
            try:
                return self._snapshot(self._arm, self._feedback_state)
            except RobotReadError:
                raise
            except Exception as exc:
                raise RobotReadError(f"读取法兰位姿失败：{exc}") from exc

    def disconnect(self):
        self._cancel.set()
        self.cancel_teaching_recovery()
        stop_error = None
        with self._lock:
            self._prepared_monotonic = self._prepared_snapshot = None
            if self._arm is not None and self._motion_active:
                try:
                    self.stop_motion()
                except Exception as exc:
                    stop_error = str(exc)
            arm, self._arm = self._arm, None
            self._feedback_state = None
        try:
            if arm is not None:
                arm.disconnect(join_timeout=0.2)
        except Exception as exc:
            raise RobotReadError(f"断开机械臂连接失败：{exc}") from exc
        finally:
            self.diagnostic_event("机械臂连接断开")
            if self._diagnostics is not None:
                try:
                    self._diagnostics.stop()
                except Exception:
                    logging.getLogger(__name__).exception("停止控制诊断失败，保留机械臂断开结果")
        if stop_error:
            raise RobotReadError(f"CAN 已关闭，但电子急停发送失败：{stop_error}")

    def motion_status(self):
        """只读完整运动状态；未知、过期或通信错误均拒绝执行。"""
        with self._lock:
            if self._arm is None or self._feedback_state is None:
                raise RobotReadError("右臂尚未连接。")
            if self._arm.has_comm_error():
                raise RobotReadError("右臂 CAN 通信异常。")
            status = self._feedback_state.motion_read()
            now = time.time()
            ages = [now - stamp for stamp in status["packet_timestamps"]]
            if min(ages) < -0.05 or max(ages) > 0.3:
                raise RobotReadError("控制器或电机状态超过 0.3 秒未更新，不能执行运动。")
            status["locked"] = self._motion_locked
            status["preparing"] = self._preparation_active
            return status

    def motion_send_state(self):
        """只读本连接的发送阶段；pending 表示指令可能在途，不代表机械臂确实移动。

        send_attempt_count 在每次使能／运动帧实际交给 CAN 发送之前递增，包含失败。
        执行器可对比入场计数，区分零发送检查失败与需要停止的可能在途指令。
        运动准备的模式帧也计数；急停和示教恢复不计数。
        到达、运动准备保持原位的确认或急停发送成功后撤销 pending。
        """
        with self._lock:
            prepared = (self._motion_phase == "prepared" and self._arm is not None
                        and not self._motion_locked and not self._preparation_active and not self._motion_active)
            return {"phase": self._motion_phase, "command_pending": self._motion_active,
                    "send_attempt_count": self._motion_send_attempts,
                    "sent_frame_count": self._motion_sent_frames,
                    "connection_id": id(self._arm) if self._arm is not None else None,
                    "prepared_monotonic": self._prepared_monotonic if prepared else None,
                    "prepared_snapshot": copy.deepcopy(self._prepared_snapshot) if prepared else None}

    @staticmethod
    def _require_normal_motion(status):
        if status.get("preparing"):
            raise RobotReadError("正在准备运动模式，不能同时发送运动目标。")
        if status["locked"]:
            raise RobotReadError("运动已锁定，请检查电子急停及现场状态，人工处理后重新连接。")
        if status["arm_status"] != 0 or status["err_code"] != 0:
            raise RobotReadError(f"控制器异常：状态 {status['arm_status']}，故障码 {status['err_code']}；不会自动复位。")
        if status["ctrl_mode"] not in (0, 1):
            raise RobotReadError("控制器仍在示教或其他控制模式；请先明确退出示教，再执行运动；程序不会自动复位。")
        if status.get("teach_status") not in (0, 2, 6):
            raise RobotReadError("控制器仍在示教记录、轨迹执行或暂停状态；请先明确退出示教，程序不会自动复位或复现轨迹。")

    def _require_ready_for_target(self, status):
        """发送锁内再次检查，直接调用 reader 也不能绕过执行器的到位检查。"""
        self._require_normal_motion(status)
        if status.get("motion_status") != 0:
            raise RobotReadError("控制器未到位，静止不能替代到位确认；未发送运动目标。")
        if (status.get("ctrl_mode") != 1 or status.get("mode_feedback") != 2
                or status.get("teach_status") not in (0, 2)):
            raise RobotReadError("尚未确认退出示教后的 CAN 直线控制模式，未发送运动目标。")
        if self._motion_active:
            raise RobotReadError("已有指令待完成，未发送新的运动目标。")
        if len(status.get("enabled", ())) != 6 or not all(status["enabled"]):
            raise RobotReadError("六关节未全部确认使能，未发送运动目标。")

    def _send_motion_frames_locked(self, messages, phases, completed_phase, *, preparation_sequence=None):
        """预编码整组帧后有界直发；不调用 SDK 会吞 ENOBUFS/ENETDOWN 的 send。"""
        self.diagnostic_event("command_intent", {"completed_phase": completed_phase, "phases": phases})
        if self._arm is None:
            self._reject_command("command_rejected", "机械臂连接已断开，运动指令未发送。")
        comm = self._arm._ctx.get_comm()
        if comm is None or not comm.is_connected() or getattr(comm, "send_bus", None) is None:
            self._reject_command("command_rejected", "CAN 已断开，运动指令未发送。")
        if len(messages) != len(phases) or not messages:
            self._reject_command("command_rejected", "运动发送帧及阶段不完整，未发送任何指令。")
        try:
            frames = [self._arm._parser.pack(message) for message in messages]
        except Exception as exc:
            self.diagnostic_event("command_rejected", {"reason": str(exc), "stage": "encoding"})
            raise
        if any(frame is None for frame in frames):
            self._reject_command("command_rejected", "SDK 未能完整编码运动指令，未发送任何指令。")
        self.diagnostic_event("command_encoded", {
            "completed_phase": completed_phase,
            "frames": [{"can_id": frame.arbitration_id, "data_hex": bytes(frame.data).hex(), "phase": phase}
                       for frame, phase in zip(frames, phases)]})
        for frame, phase in zip(frames, phases):
            preparing = (preparation_sequence is not None and self._preparation_active
                         and preparation_sequence == self._stop_sequence and not self._cancel.is_set())
            if (preparation_sequence is not None and not preparing) or (self._motion_locked and not preparing):
                self._reject_command("command_rejected", "运动已被停止操作锁定，后续指令未发送。")
            self._motion_phase = phase
            self._prepared_monotonic = self._prepared_snapshot = None
            self._motion_active = True
            self._motion_send_attempts += 1
            detail = {"can_id": frame.arbitration_id, "data_hex": bytes(frame.data).hex(), "phase": phase}
            self.diagnostic_event("send_attempt", detail)
            try:
                comm.send_bus.send(frame, timeout=.2)
            except Exception as exc:
                self.diagnostic_event("send_failed", {**detail, "error": str(exc)}, trigger=True)
                self._motion_locked = True
                self._motion_phase = "send_failed"
                error = RobotReadError(
                    f"运动帧 0x{frame.arbitration_id:03X} 发送失败，已锁定后续运动；"
                    f"指令可能已部分送达，需要停止并核对现场：{exc}")
                error.command_pending = True
                raise error from exc
            self.diagnostic_event("send_success", detail)
            self._motion_sent_frames += 1
        self._motion_phase = completed_phase

    def prepare_motion_control(self, *, operator_confirmed=False):
        """显式确认后为本机 S-V1.9-0 / v189 切换 CAN/L，核对全过程保持原位。

        只接受已退出示教且已到位的待机状态，再发送速度 1% 的 CAN/L 模式帧。
        不自动结束示教或进入待机，不发送目标、使能、复位或归零。
        该入口不推断其他固件的切换行为；成功只清软件锁，旧路径须由界面作废。
        """
        if MOTION_CONTROL_HOLD_REASON:
            raise RobotReadError(MOTION_CONTROL_HOLD_REASON)
        if operator_confirmed is not True:
            raise RobotReadError("请先确认机械臂已有可靠支撑、针尖已脱离物体，再准备运动。")
        sequence = self._stop_sequence
        with self._lock:
            if self._recovery_active or self._preparation_active:
                raise RobotReadError("正在恢复示教或准备运动，请等待当前操作结束。")
            if sequence != self._stop_sequence or self._cancel.is_set():
                raise RobotReadError("准备运动已被停止或断开操作取消。")
            if self._motion_active:
                raise RobotReadError("存在可能在途的运动指令，请先停止并核对现场。")
            self._preparation_active = True
            self._motion_locked = True
            self._prepared_monotonic = self._prepared_snapshot = None
            arm = self._arm
            initial_attempts = self._motion_send_attempts
        origin = None
        last_feedback = None

        def checked():
            nonlocal origin, last_feedback
            if (sequence != self._stop_sequence or self._cancel.is_set() or self._arm is not arm):
                raise RobotReadError("准备运动已被停止或断开操作取消。")
            status = self.motion_status()
            if status["arm_status"] != 0 or status["err_code"] != 0:
                raise RobotReadError("控制器不在正常无故障状态，不能准备运动；程序不会自动复位。")
            if status.get("motion_status") != 0:
                raise RobotReadError("控制器未到位，不能用关节静止代替到位确认；准备运动被拒绝。")
            if (status["ctrl_mode"] not in (0, 1) or status["teach_status"] not in (0, 2)
                    or status.get("mode_feedback") not in (0, 1, 2)):
                raise RobotReadError("控制器处于不支持的模式或示教轨迹执行状态，不能准备运动。")
            if len(status["enabled"]) != 6 or not all(status["enabled"]):
                raise RobotReadError("准备运动要求六关节已经使能；本按钮不会自动使能。")
            snapshot = self._checked_motion_pose()
            feedback = (snapshot["feedback_sequence"], snapshot["timestamp"])
            if last_feedback is not None and feedback[0] != last_feedback[0]:
                if not 0 < feedback[1] - last_feedback[1] <= .25:
                    raise RobotReadError("关节反馈不连续，不能确认切换时保持原位。")
            last_feedback = feedback
            if origin is None:
                origin = snapshot
            current = np.asarray(snapshot["T_base_flange"], dtype=float)
            start = np.asarray(origin["T_base_flange"], dtype=float)
            distance = float(np.linalg.norm(current[:3, 3] - start[:3, 3]))
            rotation = start[:3, :3].T @ current[:3, :3]
            angle = math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1., 1.)))
            joint_change = float(np.max(np.abs(np.asarray(snapshot["joint_angles"])
                                               - np.asarray(origin["joint_angles"]))))
            if distance > .0002 or angle > math.radians(.1) or joint_change > math.radians(.1):
                raise RobotReadError(
                    f"准备运动期间位置发生变化：法兰 {distance * 1000:.3f} mm、"
                    f"姿态 {math.degrees(angle):.3f}°、关节 {math.degrees(joint_change):.3f}°；已拒绝继续切换。")
            return status, snapshot

        def require_standby(status):
            if status["ctrl_mode"] != 0 or status["teach_status"] not in (0, 2):
                raise RobotReadError("尚未确认退出示教后的待机状态，拒绝切换 CAN 运动模式。")

        def wait_stationary(*, sent_after=None):
            deadline = time.monotonic() + 2.
            samples = []
            last_sequence = None
            while time.monotonic() < deadline:
                with self._lock:
                    status, snapshot = checked()
                    fresh = sent_after is None or status["packet_timestamps"][0] > sent_after
                    if sent_after is None:
                        require_standby(status)
                    mode_ready = sent_after is None or (fresh and status["ctrl_mode"] == 1
                                 and status["mode_feedback"] == 2 and status["teach_status"] in (0, 2))
                    if not mode_ready:
                        samples.clear()
                        last_sequence = None
                    stamp = snapshot["timestamp"]
                    if (mode_ready and (sent_after is None or stamp > sent_after)
                            and snapshot["feedback_sequence"] != last_sequence):
                        last_sequence = snapshot["feedback_sequence"]
                        samples.append((stamp, snapshot["joint_angles"]))
                        if np.max(np.ptp(np.asarray([item[1] for item in samples]), axis=0)) > math.radians(.05):
                            raise RobotReadError("机械臂关节仍在变化，不能确认连续静止；请保持可靠支撑。")
                        if len(samples) >= 5 and stamp - samples[0][0] >= .5:
                            return status, snapshot
                time.sleep(.02)
            raise RobotReadError("2 秒内未收到所需的新鲜模式确认及连续 0.5 秒静止反馈，准备运动未完成。")

        try:
            with self._lock:
                initial_status, _ = checked()
                logging.getLogger(__name__).info("运动准备入场状态：%s", initial_status)
                require_standby(initial_status)
                if self.firmware != "v189":
                    raise RobotReadError("本机运动准备要求固件选项 ≥1.8.9 · v189，请断开后选择对应选项并重新连接。")
            # 固件查询不持运动发送锁，停止操作不能被其请求等待阻塞。
            firmware = arm.get_firmware(timeout=1.)
            with self._lock:
                if not isinstance(firmware, dict) or firmware.get("software_version") != "S-V1.9-0":
                    raise RobotReadError("未确认本机 S-V1.9-0 固件，不能使用此原位模式切换；未发送运动准备指令。")
                # 查询之后重建连续性窗口，但保留原位基准，不能掩盖查询期间的漂移。
                last_feedback = None
                checked()
            wait_stationary()
            with self._lock:
                checked()
                # 位姿核对后再取最新状态，避免核对期间状态已被其他控制源改变。
                status = self.motion_status()
                require_standby(status)
                if (status["motion_status"] != 0 or status["arm_status"] != 0
                        or status["err_code"] != 0 or len(status["enabled"]) != 6
                        or not all(status["enabled"])):
                    raise RobotReadError("发送准备模式前控制器状态变化，未发送模式指令。")
                mode = copy.copy(arm._msg_mode)
                mode.ctrl_mode, mode.move_mode, mode.move_spd_rate_ctrl, mode.mit_mode = 1, 2, 1, 0
                self._send_motion_frames_locked(
                    [mode], ["preparing_mode"], "prepare_mode_sent", preparation_sequence=sequence)
                sent_after = time.time()
            status, stationary_snapshot = wait_stationary(sent_after=sent_after)
            with self._lock:
                _, snapshot = checked()
                status = self.motion_status()
                if (sequence != self._stop_sequence or self._cancel.is_set() or self._arm is not arm):
                    raise RobotReadError("运动准备已取消或连接已变化，未解除软件锁。")
                if (status["ctrl_mode"] != 1 or status["mode_feedback"] != 2
                        or status["teach_status"] not in (0, 2) or status["motion_status"] != 0
                        or status["arm_status"] != 0 or status["err_code"] != 0
                        or len(status["enabled"]) != 6 or not all(status["enabled"])):
                    raise RobotReadError("运动准备确认后控制器模式再次变化，未解除软件锁。")
                logging.getLogger(__name__).info("运动准备确认状态：%s", status)
                arm._msg_mode = mode
                self._motion_active = False
                self._motion_phase = "prepared"
                self._prepared_monotonic = time.monotonic()
                self._prepared_snapshot = copy.deepcopy(stationary_snapshot)
                self._motion_locked = False
                status["locked"], status["preparing"] = False, False
                return {"firmware": dict(firmware), "status": status,
                        "T_base_flange": snapshot["T_base_flange"], "joint_angles": snapshot["joint_angles"]}
        except Exception as exc:
            with self._lock:
                self._motion_locked = True
                sent = self._motion_send_attempts > initial_attempts and self._motion_active
                cancelled = sequence != self._stop_sequence or self._cancel.is_set() or self._arm is not arm
            stop_error = None
            if sent and not cancelled:
                try:
                    self.stop_motion()
                except Exception as stop_exc:
                    stop_error = str(stop_exc)
            if stop_error:
                raise RobotReadError(f"{exc}；停止指令也未确认送达：{stop_error}") from exc
            if isinstance(exc, RobotReadError):
                raise
            raise RobotReadError(f"准备运动失败：{exc}；软件运动锁保持锁定。") from exc
        finally:
            with self._lock:
                self._preparation_active = False

    def enable_for_motion(self):
        """仅供明确执行运动时调用；不会自动复位急停或修改控制器零位。"""
        if MOTION_CONTROL_HOLD_REASON:
            raise RobotReadError(MOTION_CONTROL_HOLD_REASON)
        with self._lock:
            status = self.motion_status()
            self._require_normal_motion(status)
            if all(status["enabled"]):
                return
            message = self._arm._MSG_MotorEnableDisableConfig(
                joint_index=self._arm._JOINT_NUMS + 1, enable_flag=2)
            self._send_motion_frames_locked([message], ["enable_sending"], "enable_sent")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = self.motion_status()
            self._require_normal_motion(status)
            if all(status["enabled"]):
                return
            time.sleep(0.05)
        raise RobotReadError("六关节未在 2 秒内确认使能，未发送运动目标。")

    def move_flange_linear(self, transform, speed_percent=5):
        """发送一个 base_from_flange 直线目标，位置米；调用本身不代表已到达。"""
        if MOTION_CONTROL_HOLD_REASON:
            raise RobotReadError(MOTION_CONTROL_HOLD_REASON)
        from 采点引导 import matrix_to_pose
        if isinstance(speed_percent, bool) or not isinstance(speed_percent, int) or not 1 <= speed_percent <= 10:
            raise RobotReadError("探针运动速度比例须为 1～10 的整数。")
        pose = matrix_to_pose(transform, "m")
        values = np.r_[np.asarray(pose[:3]) / 1000.0, np.deg2rad(pose[3:])].tolist()
        with self._lock:
            self._require_ready_for_target(self.motion_status())
            snapshot = self._checked_motion_pose()
            # 保留 SDK 当前版本的模式消息类型和坐标编码；L 在各支持版本均为 0x02。
            # 一帧同时设置 CAN/L/速度，避免先发送额外的 move_mode=0xFF 速度帧。
            mode = copy.copy(self._arm._msg_mode)
            mode.ctrl_mode = 1
            mode.move_mode = 2
            mode.move_spd_rate_ctrl = speed_percent
            mode.mit_mode = 0
            targets = self._arm._deal_move_p_msgs(values)
            if len(targets) != 3:
                raise RobotReadError("SDK 未生成完整三帧法兰目标，未发送运动指令。")
            status = self.motion_status()
            self._require_ready_for_target(status)
            logging.getLogger(__name__).info(
                "运动目标发送前核对：状态=%s，控制器法兰=%s，本地法兰=%s，目标_m_rad=%s",
                status, snapshot["controller_T_base_flange"], snapshot["T_base_flange"], values)
            self.diagnostic_event("目标发送前状态", {"status": status, "pose": snapshot,
                                                       "target_flange_m_rad": values,
                                                       "speed_percent": speed_percent})
            self._send_motion_frames_locked(
                [mode, *targets], ["mode_sending", "target_sending", "target_sending", "target_sending"],
                "target_sent")
            self._arm._msg_mode = mode

    def mark_motion_arrived(self):
        """执行线程已验证到达后撤销在途标记；不发送任何命令。"""
        with self._lock:
            self._motion_active = False
            self._motion_phase = "arrived"
            self._prepared_monotonic = self._prepared_snapshot = None

    def _send_motion_control_locked(self, *, emergency_stop=0, grag_teach_ctrl=0):
        """调用者须持发送锁；直接有界发送 SDK 编码帧，通信错误不吞掉。"""
        self.diagnostic_event("control_intent", {"emergency_stop": emergency_stop,
                                                  "teach_control": grag_teach_ctrl})
        if self._arm is None:
            self._reject_command("control_rejected", "机械臂连接已断开，控制指令无法送达。")
        comm = self._arm._ctx.get_comm()
        if comm is None or not comm.is_connected() or getattr(comm, "send_bus", None) is None:
            self._reject_command("control_rejected", "CAN 已断开，控制指令无法送达。")
        try:
            from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgMotionCtrl
            frame = self._arm._parser.pack(ArmMsgMotionCtrl(
                emergency_stop=emergency_stop, grag_teach_ctrl=grag_teach_ctrl,
            ))
        except Exception as exc:
            self.diagnostic_event("control_rejected", {"reason": str(exc), "stage": "encoding"})
            raise
        if frame is None:
            self._reject_command("control_rejected", "SDK 未能编码机械臂控制指令。")
        detail = {"can_id": frame.arbitration_id, "data_hex": bytes(frame.data).hex(),
                  "emergency_stop": emergency_stop, "teach_control": grag_teach_ctrl}
        self.diagnostic_event("send_attempt", detail)
        try:
            comm.send_bus.send(frame, timeout=.2)
        except Exception as exc:
            self.diagnostic_event("send_failed", {**detail, "error": str(exc)}, trigger=True)
            raise
        self.diagnostic_event("send_success", detail)
        return time.time()  # 等待发送成功之后收到的新状态，不能把发送阻塞期间的旧状态当确认。

    def cancel_teaching_recovery(self):
        """取消示教恢复／运动准备的后续发送；供停止、断开和关闭界面优先调用。"""
        self._stop_sequence += 1

    def recover_teaching(self, *, operator_confirmed=False):
        """明确确认可靠支撑后解除电子急停并开启拖动示教；不恢复自动运动。

        不发送使能、目标位置、轨迹复现或清除轨迹指令。仅接收新鲜反馈确认，
        超时／故障不自动补发急停；停止和断开会取消此操作的后续发送。
        """
        if operator_confirmed is not True:
            raise RobotReadError("请先确认机械臂已有可靠支撑、针尖未顶住物体，再恢复示教。")
        sequence = self._stop_sequence
        with self._lock:
            if self._recovery_active or self._preparation_active:
                raise RobotReadError("正在恢复示教或准备运动，请等待本次结果。")
            if sequence != self._stop_sequence or self._cancel.is_set():
                raise RobotReadError("恢复示教已被停止或断开操作取消。")
            self._recovery_active = True
            arm = self._arm
        interrupted_target = None

        def checked_status(*, allow_emergency):
            if (sequence != self._stop_sequence or self._cancel.is_set()
                    or self._arm is not arm):
                raise RobotReadError("恢复示教已被停止或断开操作取消。")
            status = self.motion_status()
            # 急停后的“未到达”可能只是已中断目标，不能等同于当前正在运动。
            interrupted = interrupted_target if interrupted_target is not None else status["arm_status"] == 1
            if (self._motion_active or status["motion_status"] not in (0, 1)
                    or (status["motion_status"] == 1 and not interrupted)):
                raise RobotReadError("机械臂有未完成的运动，不能恢复示教；请先确认并停止运动。")
            allowed = (0, 1) if allow_emergency else (0,)
            if status["arm_status"] not in allowed or status["err_code"] != 0:
                raise RobotReadError(
                    f"控制器状态 {status['arm_status']}，故障码 {status['err_code']}；"
                    "当前状态不支持此恢复按钮，请先检查控制器。")
            if status["ctrl_mode"] not in (0, 1, 2) or status["teach_status"] not in (0, 1, 2, 6):
                raise RobotReadError("控制器处于其他控制或轨迹状态，不能恢复示教。")
            return status

        def require_stationary():
            deadline = time.monotonic() + 2.0
            samples = []
            last_sequence = None
            while time.monotonic() < deadline:
                with self._lock:
                    checked_status(allow_emergency=True)
                    values, stamp, _, _, feedback_sequence = self._feedback(
                        self._feedback_state, "get_joint_angles", "关节角反馈", "0x2A5、0x2A6、0x2A7",
                    )
                if feedback_sequence != last_sequence:
                    if samples and (stamp <= samples[-1][0] or stamp - samples[-1][0] > .25):
                        raise RobotReadError("关节反馈不连续，无法确认机械臂静止；未发送恢复指令。")
                    last_sequence = feedback_sequence
                    samples.append((stamp, values))
                    if np.max(np.ptp(np.asarray([item[1] for item in samples]), axis=0)) > math.radians(.05):
                        raise RobotReadError("机械臂仍在移动，请可靠支撑并保持静止后再恢复示教。")
                    if len(samples) >= 5 and stamp - samples[0][0] >= .5:
                        return
                time.sleep(.02)
            raise RobotReadError("2 秒内未取得连续 0.5 秒的新鲜静止关节反馈，未发送恢复指令。")

        def wait_status(sent_after, *, teaching):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with self._lock:
                    status = checked_status(allow_emergency=not teaching)
                    if (status["packet_timestamps"][0] > sent_after
                            and status["arm_status"] == 0
                            and (not teaching or (status["teach_status"] == 1
                                                  and status["ctrl_mode"] in (0, 2)))):
                        return status
                time.sleep(.02)
            action = "开启拖动示教" if teaching else "解除电子急停"
            raise RobotReadError(f"{action}后，2 秒内未收到对应的新鲜状态确认；请保持支撑并检查控制器。")

        try:
            with self._lock:
                status = checked_status(allow_emergency=True)
                # 恢复拖动不授权自动运动；同时阻止等待期间其他线程发送运动目标。
                self._motion_locked = True
                self._prepared_monotonic = self._prepared_snapshot = None
                status["locked"] = True
                if (status["arm_status"] == 0 and status["teach_status"] == 1
                        and status["ctrl_mode"] in (0, 2)):
                    return status
                interrupted_target = status["arm_status"] == 1
            require_stationary()
            with self._lock:
                status = checked_status(allow_emergency=True)
                need_reset = status["arm_status"] == 1
                if need_reset:
                    sent_after = self._send_motion_control_locked(emergency_stop=2)
            if need_reset:
                wait_status(sent_after, teaching=False)
            with self._lock:
                checked_status(allow_emergency=False)
                sent_after = self._send_motion_control_locked(grag_teach_ctrl=1)
            return wait_status(sent_after, teaching=True)
        except RobotReadError:
            raise
        except Exception as exc:
            raise RobotReadError(f"恢复示教失败：{exc}；请保持支撑并检查控制器状态。") from exc
        finally:
            with self._lock:
                self._recovery_active = False

    def stop_motion(self):
        """同步锁定并有界发送电子急停；失败明确报告，不自动 reset。"""
        self.diagnostic_event("请求电子急停", trigger=True)
        # 先锁定，让等待同一锁的使能/移动请求也无法继续发送。
        self._motion_locked = True
        self.cancel_teaching_recovery()
        if not self._lock.acquire(timeout=.5):
            self.diagnostic_event("电子急停未取得发送锁", trigger=True)
            raise RobotReadError("0.5 秒内未取得运动发送锁，电子急停未送达；请使用机械臂实体急停。")
        try:
            self._motion_locked = True  # 在取得锁后再次置位，保证并发准备操作不能覆盖停止请求。
            self._prepared_monotonic = self._prepared_snapshot = None
            self._send_motion_control_locked(emergency_stop=1)
            self._motion_active = False
            self._motion_phase = "stop_sent"
            return {"requested": True, "sent": True, "error": None}
        except RobotReadError as exc:
            raise RobotReadError(f"电子急停发送失败：{exc}；请使用机械臂实体急停。") from exc
        except Exception as exc:
            raise RobotReadError(f"电子急停发送失败：{exc}；请使用机械臂实体急停。") from exc
        finally:
            self._lock.release()
