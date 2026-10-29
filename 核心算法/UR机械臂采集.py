"""UR 法兰位姿只读采集：只接收 Secondary 只读端口 30012 的状态。

官方协议：
https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/primary-secondary-guide.html
同一 Robot State 包内读取时间戳、六关节 actual 值、TCP 位姿和实际 TCP 偏置。
T_base_flange = T_base_tcp @ inverse(T_flange_tcp)，复用现有旋转向量转换。
不导入控制 SDK，不发送应用数据，不修改机械臂或工具设置；关闭仅断开套接字。
"""

from __future__ import annotations

import copy
import ipaddress
import socket
import struct
import threading
import time

import numpy as np

from 四点标定 import rotation_matrix
from 机械臂采集 import RobotReadError


class URReader:
    """使用控制器实际 TCP 偏置还原法兰；需提供 UR 的 IP 地址。

    支持带完整 CartesianInfo 偏置的 Secondary 状态格式；格式缺失时拒绝采集。
    connect/read_pose 可在界面后台线程调用；每次 read_pose 返回新的状态帧。
    """

    read_only = True
    supports_motion = False
    _PORT = 30012
    _MAX_AGE = 0.5
    _READ_TIMEOUT = 0.6
    _MAX_PACKET = 65536

    def __init__(self, host: str, robot_model: str | None = None):
        try:
            if not isinstance(host, str) or not host.strip():
                raise ValueError
            address = ipaddress.ip_address(host.strip())
        except ValueError as exc:
            raise RobotReadError("请输入 UR 控制器的有效 IP 地址。") from exc
        if robot_model is not None and not isinstance(robot_model, str):
            raise RobotReadError("UR 型号须为文字，也可以留空。")
        self.host = self.channel = str(address)
        self.robot_model = (robot_model or "").strip() or "未指定"
        self._family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        self._condition = threading.Condition(threading.RLock())
        self._socket = None
        self._worker = None
        self._generation = 0
        self._latest = None
        self._last_read_timestamp = -1
        self._error = None

    @property
    def capabilities(self):
        return {"read_only": True, "read_pose": True, "motion": False,
                "teaching_recovery": False, "write_tcp": False}

    @staticmethod
    def _close_socket(sock):
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    @staticmethod
    def _decode_state(packet):
        """严格解析一整帧；不拼接不同控制周期的子包。"""
        if len(packet) < 5 or struct.unpack_from("!I", packet)[0] != len(packet):
            raise RobotReadError("UR 状态包长度不正确。")
        if packet[4] != 16:
            return None  # 版本等非 Robot State 消息无需参与位姿采集。
        parts, offset = {}, 5
        while offset < len(packet):
            if len(packet) - offset < 5:
                raise RobotReadError("UR 状态子包不完整。")
            size, kind = struct.unpack_from("!IB", packet, offset)
            if size < 5 or offset + size > len(packet):
                raise RobotReadError("UR 状态子包长度不正确。")
            if kind in (0, 1, 4):
                if kind in parts:
                    raise RobotReadError("UR 同一状态帧包含重复的位姿或关节子包。")
                parts[kind] = packet[offset + 5:offset + size]
            offset += size
        if not all(kind in parts for kind in (0, 1, 4)):
            raise RobotReadError("UR 状态帧缺少时间戳、关节或 TCP 位姿，不能采集法兰。")
        if len(parts[0]) < 8 or len(parts[1]) != 6 * 41 or len(parts[4]) < 12 * 8:
            raise RobotReadError(
                "UR 状态格式不兼容或缺少实际 TCP 偏置，不能把 TCP 当作法兰；"
                "请核对控制器版本与 Secondary 状态接口。")
        timestamp_us = struct.unpack_from("!Q", parts[0])[0]
        joints = [struct.unpack_from("!d", parts[1], i * 41)[0] for i in range(6)]
        cartesian = struct.unpack_from("!12d", parts[4])
        if timestamp_us == 0 or not np.isfinite([*joints, *cartesian]).all():
            raise RobotReadError("UR 位姿、关节或时间戳无效，不能作为标定样本。")
        tcp, tool = np.eye(4), np.eye(4)
        tcp[:3, 3], tool[:3, 3] = cartesian[:3], cartesian[6:9]
        tcp[:3, :3] = rotation_matrix(cartesian[3:6], "rotation_vector_rad")
        tool[:3, :3] = rotation_matrix(cartesian[9:12], "rotation_vector_rad")
        flange = np.eye(4)
        flange[:3, :3] = tcp[:3, :3] @ tool[:3, :3].T
        flange[:3, 3] = tcp[:3, 3] - flange[:3, :3] @ tool[:3, 3]
        if not np.isfinite(flange).all():
            raise RobotReadError("UR 法兰位姿换算出现无效数值。")
        return {"T_base_flange": flange.tolist(), "joint_angles": joints,
                "controller_timestamp_us": timestamp_us,
                "controller_timestamp": timestamp_us / 1e6,
                "actual_tcp_pose": list(cartesian[:6]),
                "tcp_offset": list(cartesian[6:]),
                "tcp_pose_format": "xyz_m_rotation_vector_rad",
                "T_flange_controller_tcp": tool.tolist()}

    def _receive(self, sock, generation):
        buffer = bytearray()
        partial_since = None
        try:
            while True:
                with self._condition:
                    if self._socket is not sock or self._generation != generation:
                        return
                try:
                    chunk = sock.recv(8192)
                except socket.timeout:
                    if partial_since is not None and time.monotonic() - partial_since > self._MAX_AGE:
                        raise RobotReadError("UR 状态包接收不完整或超时，请检查网络。")
                    continue
                if not chunk:
                    raise RobotReadError("UR 只读连接已关闭，请检查控制器和网络。")
                if partial_since is not None and time.monotonic() - partial_since > self._MAX_AGE:
                    raise RobotReadError("UR 状态包传输超过允许时间，请检查网络后重新连接。")
                if not buffer:
                    partial_since = time.monotonic()
                buffer.extend(chunk)
                while len(buffer) >= 4:
                    size = struct.unpack_from("!I", buffer)[0]
                    if not 5 <= size <= self._MAX_PACKET:
                        raise RobotReadError("UR 返回了不支持的消息长度。")
                    if len(buffer) < size:
                        break
                    packet = bytes(buffer[:size])
                    del buffer[:size]
                    received_mono, received_epoch = time.monotonic(), time.time()
                    packet_age = received_mono - partial_since
                    partial_since = received_mono if buffer else None
                    state = self._decode_state(packet)
                    if state is None:
                        continue
                    with self._condition:
                        if self._socket is not sock or self._generation != generation:
                            return
                        previous = self._latest
                        if previous is not None:
                            stamp = state["controller_timestamp_us"]
                            if stamp < previous["controller_timestamp_us"]:
                                raise RobotReadError("UR 控制器时间戳倒退，请重新连接后采集。")
                            if stamp == previous["controller_timestamp_us"]:
                                continue  # 重复包不能刷新接收时间或伪装成新样本。
                        state.update({
                            "transform_direction": "base_from_flange", "translation_unit": "m",
                            "joint_angle_unit": "rad", "frame": "flange", "model": "ur",
                            "robot_model": self.robot_model, "host": self.host, "channel": self.channel,
                            "port": self._PORT, "read_only": True,
                            "pose_source": "ur_secondary_tcp_offset",
                            "pose_source_label": "UR 实际 TCP 位姿与实际工具偏置还原法兰",
                            "monotonic": received_mono - packet_age, "timestamp": received_epoch - packet_age,
                            "packet_timestamps": [received_epoch - packet_age],
                            "feedback_snapshot": "secondary_robot_state_packet",
                            "feedback_sequence": 1 if previous is None else previous["feedback_sequence"] + 1,
                            "connection_id": f"ur:{id(self)}:{generation}",
                        })
                        self._latest = state
                        self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if self._socket is sock and self._generation == generation:
                    self._error = exc if isinstance(exc, RobotReadError) else RobotReadError(f"UR 只读接收失败：{exc}")
                    self._condition.notify_all()
        finally:
            self._close_socket(sock)

    def connect(self):
        """连接 30012 并验证首帧；只接收状态，不发送应用数据。"""
        with self._condition:
            if self._socket is not None:
                raise RobotReadError("UR 已连接或正在连接，请先断开再连接。")
            self._generation += 1
            generation = self._generation
            sock = socket.socket(self._family, socket.SOCK_STREAM)
            self._socket = sock
            self._latest, self._error = None, None
            self._last_read_timestamp = -1
        try:
            sock.settimeout(3.0)
            sock.connect((self.host, self._PORT))
            sock.settimeout(0.2)
            with self._condition:
                if self._socket is not sock or self._generation != generation:
                    raise RobotReadError("UR 连接已取消。")
                self._worker = threading.Thread(target=self._receive, args=(sock, generation), daemon=True)
                self._worker.start()
            self.read_pose()
        except Exception as exc:
            with self._condition:
                if self._socket is sock:
                    self._socket = None
                    self._latest = None
                    self._condition.notify_all()
            self._close_socket(sock)
            if isinstance(exc, RobotReadError):
                raise
            raise RobotReadError(
                f"无法读取 UR {self.host}:30012：{exc}。请检查 IP、网线及控制器 Secondary 服务。") from exc

    def read_pose(self):
        """最多等 0.6 秒，返回相对上次读取具有更新控制器戳的完整法兰快照。"""
        deadline = time.monotonic() + self._READ_TIMEOUT
        with self._condition:
            generation = self._generation
            while True:
                if self._generation != generation:
                    raise RobotReadError("UR 连接已变化，本次读取已取消。")
                if self._socket is None:
                    raise RobotReadError("尚未连接 UR，请先填写 IP 并连接。")
                if self._error is not None:
                    raise self._error
                state, now = self._latest, time.monotonic()
                if (state is not None and now - state["monotonic"] <= self._MAX_AGE
                        and state["controller_timestamp_us"] > self._last_read_timestamp):
                    self._last_read_timestamp = state["controller_timestamp_us"]
                    return copy.deepcopy(state)
                remaining = deadline - now
                if remaining <= 0:
                    raise RobotReadError("未收到 UR 新的完整状态帧，请检查网络与控制器状态后重试。")
                self._condition.wait(remaining)

    def cancel_teaching_recovery(self):
        """UR 接入没有示教恢复操作；保留公共界面的取消接口。"""

    def disconnect(self):
        """只关闭只读套接字；不向机器人发送停止、复位或其他控制指令。"""
        with self._condition:
            self._generation += 1
            sock, self._socket = self._socket, None
            worker, self._worker = self._worker, None
            self._latest, self._error = None, None
            self._condition.notify_all()
        self._close_socket(sock)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=0.3)
