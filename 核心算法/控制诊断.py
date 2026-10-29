"""PiPER 控制诊断：独立 SocketCAN 只收、内存留存、显式导出。

本模块没有发送接口，不配置 CAN，不修改机械臂模式或使能。字段单位依据
当前 pyAgxArm d90e234 的 PiPER 解码器；原始帧始终保留，便于厂家复核。
接收调用时间使用 monotonic_ns 排序，另存内核墙钟时间；两者不可混作
控制器内部执行时间。发送成功仅表示本机 CAN 发送调用返回，不表示到位。
"""

from __future__ import annotations

from collections import Counter, deque
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import struct
import threading
import time
from uuid import uuid4


_FEEDBACK_IDS = (0x2A1, *range(0x2A2, 0x2A8), *range(0x251, 0x257), *range(0x261, 0x267))
_PRE_NS, _POST_NS = 5_000_000_000, 2_000_000_000
_FRAME_LIMIT, _EVENT_LIMIT, _FROZEN_LIMIT = 40000, 2000, 60000
# 整轮原始帧紧凑存内存，约 100 MiB；容量满时停止，绝不覆盖开头证据。
_SESSION_FRAME_LIMIT, _SESSION_EVENT_LIMIT = 2_000_000, 20_000
_SESSION_RECORD = struct.Struct("<qqqqI16s")
_SO_TIMESTAMPNS, _SO_RXQ_OVFL = 35, 40  # Linux SocketCAN ancillary messages.


def _plain(value):
    """将轻量命令上下文转成 JSON 值，兼容现有 numpy 位姿。"""
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def decode_frame(identifier, data):
    """仅解码完整标准数据帧；未知字段保留原始字节，不猜测固件行为。"""
    if len(data) != 8:
        return {"kind": "invalid_length", "length": len(data)}
    if identifier == 0x2A1:
        return dict(zip(("ctrl_mode", "arm_status", "mode_feedback", "teach_status",
                         "motion_status", "trajectory_num"), data[:6]),
                    kind="controller_status", err_code=int.from_bytes(data[6:8], "big"))
    if 0x2A2 <= identifier <= 0x2A4 or 0x152 <= identifier <= 0x154:
        offset = identifier - (0x2A2 if identifier >= 0x2A2 else 0x152)
        values = struct.unpack(">ii", data)
        fields = (("x_mm", "y_mm"), ("z_mm", "rx_deg"), ("ry_deg", "rz_deg"))[offset]
        return dict(zip(fields, (item / 1000 for item in values)),
                    kind="flange_feedback" if identifier >= 0x2A2 else "flange_target")
    if 0x2A5 <= identifier <= 0x2A7 or 0x155 <= identifier <= 0x157:
        offset = identifier - (0x2A5 if identifier >= 0x2A5 else 0x155)
        return {"kind": "joint_feedback" if identifier >= 0x2A5 else "joint_target",
                "joints": [2 * offset + 1, 2 * offset + 2],
                "degrees": [item / 1000 for item in struct.unpack(">ii", data)]}
    if 0x251 <= identifier <= 0x256:
        velocity, current, position = struct.unpack(">hhi", data)
        return {"kind": "motor_high_speed", "joint": identifier - 0x250,
                "velocity_rad_s": velocity / 1000, "current_A": current / 1000,
                "position_rad": position / 1000}
    if 0x261 <= identifier <= 0x266:
        voltage, driver_temp, motor_temp, flags, bus_current = struct.unpack(">HhbBH", data)
        names = ("voltage_low", "motor_overheat", "overcurrent", "driver_overheat",
                 "collision", "driver_error", "enabled", "stall")
        return {"kind": "motor_low_speed", "joint": identifier - 0x260,
                "voltage_V": voltage / 10, "driver_temperature_C": driver_temp,
                "motor_temperature_C": motor_temp, "foc_status": flags,
                "bus_current_A": bus_current / 1000,
                **{name: bool(flags & (1 << index)) for index, name in enumerate(names)}}
    if identifier == 0x150:
        return {"kind": "motion_control", "emergency_stop": data[0],
                "trajectory_control": data[1], "teach_control": data[2]}
    if identifier == 0x151:
        return {"kind": "mode_control", "ctrl_mode": data[0], "move_mode": data[1],
                "speed_percent": data[2], "mit_mode": data[3]}
    if identifier == 0x471:
        return {"kind": "enable_control", "joint_index": data[0], "enable_flag": data[1]}
    return {"kind": "raw"}


class ControlDiagnostics:
    """一个只读记录会话；停止／关闭不保存，export 是唯一文件写入入口。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._socket = self._worker = None
        self._running = False
        self._channel = None
        self._started_ns = self._started_wall_ns = None
        self._active_start_ns = None
        self._elapsed_recording_ns = 0
        self._intervals = []
        self._interval_count = 0
        self._frames, self._events = deque(), deque()
        self._session_frames = bytearray()
        self._session_events = []
        self._capacity_reached = False
        self._first_frame_ns = self._last_frame_ns = None
        self._counts = Counter()
        self._last_seen, self._latest = {}, {}
        self._frame_count = self._event_count = 0
        self._overwritten = self._expired = self._event_overwritten = 0
        self._kernel_drops = self._socket_drop_previous = 0
        self._malformed = self._truncated = 0
        self._error = None
        self._trigger = None
        self._timestamp_supported = self._overflow_supported = False
        self._recv_buffer_bytes = None

    def start(self, channel):
        if not isinstance(channel, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", channel):
            raise ValueError("请输入有效的 CAN 接口名称，例如 can1。")
        with self._lock:
            if self._running:
                raise RuntimeError("诊断记录已经开启。")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("上一次诊断正在结束，请稍后再开始。")
            if self._capacity_reached:
                raise RuntimeError("本轮记录容量已满，请先保存，再开始新一轮诊断。")
            if self._channel is not None and self._channel != channel and self._frame_count:
                raise RuntimeError("已有另一 CAN 接口的记录，请先导出，再新建诊断会话。")
            sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            try:
                sock.settimeout(.1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                self._recv_buffer_bytes = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                for option, attribute in ((_SO_TIMESTAMPNS, "_timestamp_supported"),
                                          (_SO_RXQ_OVFL, "_overflow_supported")):
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, option, 1)
                        setattr(self, attribute, True)
                    except OSError:
                        setattr(self, attribute, False)
                # 接收错误帧；本 socket 无 send 调用，本机其他 socket 的发送也会回环至此。
                sock.setsockopt(socket.SOL_CAN_RAW, getattr(socket, "CAN_RAW_ERR_FILTER", 2),
                                struct.pack("=I", 0x1FFFFFFF))
                sock.bind((channel,))
            except Exception:
                sock.close()
                raise
            self._socket, self._channel = sock, channel
            self._socket_drop_previous = 0
            self._stop.clear()
            self._error, self._running = None, True
            now, wall = time.monotonic_ns(), time.time_ns()
            if self._started_ns is None:
                self._started_ns, self._started_wall_ns = now, wall
            self._active_start_ns = now
            self._interval_count += 1
            self._intervals.append({"start_monotonic_ns": now, "start_wall_time_ns": wall,
                                    "stop_monotonic_ns": None, "stop_wall_time_ns": None,
                                    "stop_reason": None})
            self.event("开始只读记录", {"channel": channel})
            self._worker = threading.Thread(target=self._receive, args=(sock,), daemon=True,
                                            name="PiPER控制诊断只收")
            self._worker.start()

    def stop(self):
        with self._lock:
            now, wall = time.monotonic_ns(), time.time_ns()
            was_running = self._running
            self._running = False
            self._stop.set()
            sock, self._socket = self._socket, None
            worker = self._worker
            if was_running:
                self._append_event("停止只读记录", None, now, wall)
            self._end_interval(now, wall, "手动停止或连接断开")
            if self._trigger and not self._trigger["complete"]:
                self._finish_trigger(time.monotonic_ns(), "记录已停止，后窗口可能不足 2 秒")
        if sock is not None:
            sock.close()
        if worker and worker is not threading.current_thread():
            worker.join(timeout=.4)

    def _end_interval(self, now, wall, reason):
        if self._active_start_ns is not None:
            self._elapsed_recording_ns += now - self._active_start_ns
            self._active_start_ns = None
            self._intervals[-1].update(stop_monotonic_ns=now, stop_wall_time_ns=wall,
                                       stop_reason=reason)

    def _receive(self, sock):
        try:
            while not self._stop.is_set():
                try:
                    packet, ancillary, flags, _ = sock.recvmsg(16, 128)
                except socket.timeout:
                    with self._lock:
                        self._advance_trigger(time.monotonic_ns())
                    continue
                now, wall = time.monotonic_ns(), time.time_ns()
                kernel_wall = drops = None
                for level, option, value in ancillary:
                    if level == socket.SOL_SOCKET and option == _SO_TIMESTAMPNS and len(value) >= struct.calcsize("@ll"):
                        seconds, nanos = struct.unpack("@ll", value[:struct.calcsize("@ll")])
                        kernel_wall = seconds * 1_000_000_000 + nanos
                    elif level == socket.SOL_SOCKET and option == _SO_RXQ_OVFL and len(value) >= 4:
                        drops = struct.unpack("=I", value[:4])[0]
                self._ingest(packet, now, wall, kernel_wall, drops, flags)
        except Exception as exc:
            if not self._stop.is_set():
                with self._lock:
                    self._error = str(exc)
                self.event("诊断接收异常", {"error": str(exc)}, trigger=True)
        finally:
            with self._lock:
                self._running = False
                self._end_interval(time.monotonic_ns(), time.time_ns(), "接收线程结束")
                if self._socket is sock:
                    self._socket = None
                if self._trigger and not self._trigger["complete"]:
                    self._finish_trigger(time.monotonic_ns(), "接收结束，后窗口可能不足 2 秒")
            sock.close()

    def _ingest(self, packet, now, wall, kernel_wall=None, drops=None, flags=0):
        with self._lock:
            if not self._running:
                return
            self._advance_trigger(now)
            if drops is not None:
                self._kernel_drops += (drops - self._socket_drop_previous) & 0xFFFFFFFF
                self._socket_drop_previous = drops
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                self._truncated += 1
            if len(packet) != 16:
                self._malformed += 1
                self.event("CAN 报文长度异常", {"packet_hex": packet.hex(), "flags": flags,
                           "monotonic_ns": now, "wall_time_ns": wall}, trigger=True)
                return
            self._session_frames.extend(_SESSION_RECORD.pack(
                now, wall, kernel_wall if kernel_wall is not None else -1,
                drops if drops is not None else -1, flags, packet))
            if self._first_frame_ns is None:
                self._first_frame_ns = now
            self._last_frame_ns = now
            can_id, dlc, data = struct.unpack("=IB3x8s", packet)
            identifier = can_id & 0x1FFFFFFF
            data = data[:min(dlc, 8)]
            self._frame_count += 1
            entry = {"sequence": self._frame_count, "monotonic_ns": now, "wall_time_ns": wall,
                     "kernel_wall_time_ns": kernel_wall, "can_id": identifier,
                     "socket_drop_count": drops, "recv_flags": flags, "packet_hex": packet.hex(),
                     "extended": bool(can_id & 0x80000000), "remote": bool(can_id & 0x40000000),
                     "error_frame": bool(can_id & 0x20000000), "dlc": dlc, "data_hex": data.hex()}
            self._counts[identifier] += 1
            valid = not can_id & 0xE0000000 and dlc == 8
            decoded = decode_frame(identifier, data) if valid else {"kind": "raw"}
            if valid:
                self._last_seen[identifier] = now
            self._frames.append(entry)
            while self._frames and self._frames[0]["monotonic_ns"] < now - _PRE_NS:
                self._frames.popleft()
                self._expired += 1
            while len(self._frames) > _FRAME_LIMIT:
                self._frames.popleft()
                self._overwritten += 1
            self._append_frozen("frames", entry)
            if identifier == 0x2A1 and valid:
                previous = self._latest.get("controller")
                self._latest["controller"] = {**decoded, "monotonic_ns": now}
                changed = previous is None or any(previous[key] != decoded[key] for key in ("arm_status", "err_code"))
                # 0x0B/0x0C/0x0D 是示教记录／执行／暂停状态，不是故障。
                if changed and (decoded["arm_status"] not in (0, 11, 12, 13) or decoded["err_code"] != 0):
                    self.event("控制器报告异常", decoded, trigger=True)
            elif 0x261 <= identifier <= 0x266 and valid:
                previous = self._latest.get(f"motor_{decoded['joint']}")
                self._latest[f"motor_{decoded['joint']}"] = {**decoded, "monotonic_ns": now}
                if decoded["foc_status"] & 0xBF and (previous is None or previous["foc_status"] != decoded["foc_status"]):
                    self.event("电机报告异常", decoded, trigger=True)
                elif previous and previous["enabled"] and not decoded["enabled"]:
                    self.event("关节使能丢失", decoded, trigger=True)
            elif entry["error_frame"]:
                self.event("CAN 错误帧", {"can_id": identifier, "data_hex": data.hex()}, trigger=True)
            if self._frame_count >= _SESSION_FRAME_LIMIT:
                self._stop_for_capacity(now, wall)

    def _stop_for_capacity(self, now, wall):
        """只停止本记录器接收；不操作机械臂，不写磁盘。"""
        if self._capacity_reached:
            return
        self._capacity_reached = True
        self._error = "本轮记录容量已满，已停止接收并保留全部已录数据；请保存后开始新一轮。"
        self._append_event("记录容量已满", {"reason": self._error}, now, wall)
        self._running = False
        self._stop.set()
        self._end_interval(now, wall, "记录容量已满")
        if self._trigger and not self._trigger["complete"]:
            self._finish_trigger(now, "记录容量已满，后窗口可能不足 2 秒")

    def _append_event(self, label, details, now, wall):
        self._event_count += 1
        entry = {"sequence": self._event_count, "monotonic_ns": now, "wall_time_ns": wall,
                 "label": str(label), "details": _plain(details)}
        self._session_events.append(entry)
        self._events.append(entry)
        while self._events and self._events[0]["monotonic_ns"] < now - _PRE_NS:
            self._events.popleft()
        while len(self._events) > _EVENT_LIMIT:
            self._events.popleft()
            self._event_overwritten += 1
        self._append_frozen("events", entry)
        return entry

    def event(self, label, details=None, *, trigger=False):
        with self._lock:
            if not self._running:
                return
            now, wall = time.monotonic_ns(), time.time_ns()
            self._advance_trigger(now)
            entry = self._append_event(label, details, now, wall)
            if trigger and self._trigger is None:
                frames = [item for item in self._frames if item["monotonic_ns"] >= now - _PRE_NS]
                self._trigger = {"label": str(label), "monotonic_ns": now, "wall_time_ns": wall,
                                 "requested_pre_s": 5, "requested_post_s": 2,
                                 "frames": frames, "events": list(self._events),
                                 "complete": False, "capacity_drops": 0,
                                 "pre_coverage_s": (now - frames[0]["monotonic_ns"]) / 1e9 if frames else 0,
                                 "post_coverage_s": 0,
                                 "completion_reason": None}
            if len(self._session_events) >= _SESSION_EVENT_LIMIT:
                self._stop_for_capacity(now, wall)
            return entry

    def mark(self, label="操作者标记异常", details=None):
        with self._lock:
            if not self._running or not self._frames:
                raise RuntimeError("请先开始诊断并等待收到 CAN 反馈，再标记异常。")
            return self.event(label, details, trigger=True)

    def _append_frozen(self, name, entry):
        if self._trigger and not self._trigger["complete"]:
            if len(self._trigger[name]) < (_FROZEN_LIMIT if name == "frames" else _EVENT_LIMIT):
                self._trigger[name].append(entry)
                if name == "frames":
                    self._trigger["post_coverage_s"] = max(0, (entry["monotonic_ns"] - self._trigger["monotonic_ns"]) / 1e9)
            else:
                self._trigger["capacity_drops"] += 1

    def _advance_trigger(self, now):
        if self._trigger and not self._trigger["complete"] and now >= self._trigger["monotonic_ns"] + _POST_NS:
            self._finish_trigger(now, "后窗口已结束；首个异常片段已冻结，整轮记录继续保留")

    def _finish_trigger(self, now, reason):
        self._trigger["complete"] = True
        self._trigger["ended_monotonic_ns"] = now
        self._trigger["completion_reason"] = reason
        frames = self._trigger["frames"]
        self._trigger["post_coverage_s"] = max(0, (frames[-1]["monotonic_ns"] - self._trigger["monotonic_ns"]) / 1e9) if frames else 0

    def status(self):
        with self._lock:
            now = time.monotonic_ns()
            self._advance_trigger(now)
            missing = [f"0x{identifier:03X}" for identifier in _FEEDBACK_IDS if identifier not in self._last_seen]
            stale = [f"0x{identifier:03X}" for identifier in _FEEDBACK_IDS
                     if identifier in self._last_seen and now - self._last_seen[identifier] > 500_000_000]
            reason = self._error or ("等待控制器／电机反馈：" + "、".join(missing) if missing else
                                    "反馈超过 0.5 秒未更新：" + "、".join(stale) if stale else "控制器、六关节、法兰和电机反馈均已采到")
            if self._trigger:
                reason += "；首个异常证据已冻结" if self._trigger["complete"] else "；正在保留异常后 2 秒"
            return {"running": self._running, "channel": self._channel, "frames": self._frame_count,
                    "events": self._event_count, "buffered_frames": len(self._frames),
                    "retention": "full_session", "retained_frames": len(self._session_frames) // _SESSION_RECORD.size,
                    "retained_events": len(self._session_events),
                    "session_frame_limit": _SESSION_FRAME_LIMIT, "session_event_limit": _SESSION_EVENT_LIMIT,
                    "capacity_reached": self._capacity_reached,
                    "retained_raw_bytes": len(self._session_frames),
                    "first_frame_monotonic_ns": self._first_frame_ns,
                    "last_frame_monotonic_ns": self._last_frame_ns,
                    "retained_duration_s": ((self._last_frame_ns - self._first_frame_ns) / 1e9
                                            if self._first_frame_ns is not None else 0),
                    "session_discarded_frames": 0,
                    "overwritten_frames": self._overwritten, "expired_frames": self._expired,
                    "overwritten_events": self._event_overwritten, "kernel_drops": self._kernel_drops,
                    "malformed_packets": self._malformed, "truncated_packets": self._truncated,
                    "kernel_drop_monitoring": self._overflow_supported,
                    "kernel_timestamp_available": self._timestamp_supported,
                    "receive_buffer_bytes": self._recv_buffer_bytes, "error": self._error,
                    "triggered": self._trigger is not None,
                    "trigger_complete": bool(self._trigger and self._trigger["complete"]),
                    "missing_ids": missing, "stale_ids": stale,
                    "started_at": datetime.fromtimestamp(self._started_wall_ns / 1e9, timezone.utc).isoformat() if self._started_wall_ns else None,
                    "elapsed_s": (self._elapsed_recording_ns + (now - self._active_start_ns if self._active_start_ns is not None else 0)) / 1e9,
                    "latest_state": copy.deepcopy(self._latest), "has_data": bool(self._frame_count or self._event_count),
                    "available": hasattr(socket, "AF_CAN"), "recommended_reason": reason,
                    "counts_by_id": {f"0x{key:03X}": value for key, value in sorted(self._counts.items())}}

    def _capture(self):
        with self._lock:
            status = self.status()
            frozen = ({**self._trigger, "frames": list(self._trigger["frames"]),
                       "events": list(self._trigger["events"])} if self._trigger else None)
            packed = bytes(self._session_frames)
            result = {"format": "PiPER控制诊断-v2", "status": status,
                      "recording_intervals": copy.deepcopy(list(self._intervals)),
                      "older_intervals_omitted": max(0, self._interval_count - len(self._intervals)),
                      "notes": ["只读 SocketCAN 记录；不会使能、复位、切模式或发送位置。",
                                "原始帧可能包含本机其他发送者的回环；仅程序发送事件可归因本程序。",
                                "monotonic_ns 为本机收包／事件时间，内核墙钟另存；均非控制器内部执行时间。",
                                "send_success 仅表示 CAN 发送调用返回，不能证明控制器接收或执行。",
                                "frames/events 保留本轮从开始到停止的全部已录原始帧和程序事件；容量满即停止，不滚动覆盖。",
                                "recent_frames/recent_events 及 trigger 是辅助窗口；expired/overwritten 仅指辅助窗口淘汰，不代表整轮数据丢失。",
                                "仅调用导出接口时写磁盘；手动会话由保存按钮触发，自动运动会话由流程结束触发。容量停止、断线、接收错误及内核丢帧仍可能造成未录到的数据，须核对状态和时间边界。",
                                "标准 CAN 无连续帧序号；上游总线／设备漏报无法仅靠本机记录证明为零。"],
                      "recent_frames": list(self._frames), "recent_events": list(self._events),
                      "events": copy.deepcopy(self._session_events), "trigger": frozen}
        # 解码与导出在接收锁外完成，避免磁盘和 JSON 处理阻塞接收。
        def decoded(entries):
            return [{**entry, "decoded": decode_frame(entry["can_id"], bytes.fromhex(entry["data_hex"]))
                     if entry["dlc"] == 8 and not (entry["error_frame"] or entry["remote"] or entry["extended"]) else {"kind": "raw"}}
                    for entry in entries]
        result["recent_frames"] = decoded(result["recent_frames"])
        result["recent_events"] = copy.deepcopy(result["recent_events"])
        if result["trigger"]:
            result["trigger"]["frames"] = decoded(result["trigger"]["frames"])
            result["trigger"]["events"] = copy.deepcopy(result["trigger"]["events"])
        return result, packed

    @staticmethod
    def _session_entries(packed):
        for sequence, (now, wall, kernel_wall, drops, flags, packet) in enumerate(
                _SESSION_RECORD.iter_unpack(packed), 1):
            can_id, dlc, data = struct.unpack("=IB3x8s", packet)
            identifier = can_id & 0x1FFFFFFF
            data = data[:min(dlc, 8)]
            yield {"sequence": sequence, "monotonic_ns": now, "wall_time_ns": wall,
                   "kernel_wall_time_ns": None if kernel_wall == -1 else kernel_wall,
                   "socket_drop_count": None if drops == -1 else drops,
                   "can_id": identifier, "extended": bool(can_id & 0x80000000),
                   "remote": bool(can_id & 0x40000000), "error_frame": bool(can_id & 0x20000000),
                   "dlc": dlc, "data_hex": data.hex(), "packet_hex": packet.hex(), "recv_flags": flags,
                   "decoded": decode_frame(identifier, data) if dlc == 8 and not can_id & 0xE0000000
                              else {"kind": "raw"}}

    def snapshot(self):
        result, packed = self._capture()
        result["frames"] = list(self._session_entries(packed))
        return result

    def export(self, path, *, context=None):
        destination = Path(path)
        data, packed = self._capture()
        if context is not None:
            data["automatic_session"] = _plain(context)
        # 逐帧编码导出，避免把几百万条解码字典同时展开到内存。
        with destination.open("w", encoding="utf-8") as handle:
            header = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write(header[:-1] + ',"frames":[')
            for index, entry in enumerate(self._session_entries(packed)):
                if index:
                    handle.write(",")
                json.dump(entry, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write("]}\n")
        return destination


class AutomaticDiagnosticExport:
    """已结束会话的独立导出任务；由界面后台线程调用 export，可失败重试。"""

    def __init__(self, output_dir, recorder, context):
        self.recorder = recorder
        self.context = copy.deepcopy(context)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.path = Path(output_dir) / f"运动诊断_{stamp}_{uuid4().hex[:8]}.json"
        self.saved = False
        self.error = None
        self._lock = threading.Lock()

    def export(self):
        """仅完整 JSON 写完后发布最终文件；失败保留原会话和失败原因。"""
        with self._lock:
            if self.saved:
                return self.path
            temporary = self.path.with_suffix(".json.写入中")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.recorder.export(temporary, context=self.context)
                os.replace(temporary, self.path)
            except Exception as exc:
                self.error = f"自动保存运动诊断失败：{exc}"
                self.context.setdefault("export_failures", []).append({
                    "wall_time_ns": time.time_ns(), "error": self.error})
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise RuntimeError(self.error) from exc
            self.saved, self.error = True, None
            return self.path


class AutomaticControlDiagnostics:
    """选点至流程结束的一轮自动只读诊断，不连接机械臂或发送任何指令。

    begin 绑定已有 reader；finish 停止接收、解绑并返回导出任务。界面须在
    运动线程结束后调用 finish，再在后台调用任务 export，避免阻塞界面。
    上一轮未成功保存时拒绝新建，保留全部内存记录供原任务重试。
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.recorder = ControlDiagnostics()
        self.reader = None
        self.last_export = None
        self._active = False
        self._context = {}
        self._lock = threading.RLock()

    @property
    def active(self):
        with self._lock:
            return self._active

    def begin(self, reader, context=None):
        with self._lock:
            if self._active:
                state = self.recorder.status()
                if self.reader is not reader:
                    raise RuntimeError("机械臂连接已变化，请先结束并保存当前自动诊断。")
                if not state["running"]:
                    raise RuntimeError(state["error"] or self._context.get("start_error")
                                       or "本轮诊断接收已停止，请先结束并保存当前记录。")
                self.event("继续本轮自动诊断", context)
                return self.recorder
            if self.last_export is not None and not self.last_export.saved:
                raise RuntimeError(self.last_export.error or "上一轮运动诊断尚未保存完成，请稍候。")
            if not callable(getattr(reader, "attach_diagnostics", None)):
                raise RuntimeError("当前机械臂连接不支持完整控制诊断，无法开始本轮记录。")
            self.recorder = ControlDiagnostics()
            self.last_export = None
            self.reader, self._active = reader, True
            self._context = {"mode": "automatic", "start_monotonic_ns": time.monotonic_ns(),
                             "start_wall_time_ns": time.time_ns(), "start_context": _plain(context),
                             "after_capture_events": [], "export_failures": []}
            try:
                # 绑定本封装，使接收意外结束后的程序停止／异常事件仍有留存位置。
                reader.attach_diagnostics(self)
                self.recorder.start(reader.channel)
            except Exception as exc:
                self._context["start_error"] = f"自动诊断未能开始采集：{exc}"
                self.event("自动诊断启动失败", {"error": str(exc)}, trigger=True)
                reader.attach_diagnostics(None)
                raise RuntimeError(self._context["start_error"]) from exc
            self.event("自动诊断开始", context)
            return self.recorder

    def event(self, label, details=None, *, trigger=False):
        with self._lock:
            if not self._active:
                return None
            entry = self.recorder.event(label, details, trigger=trigger)
            if entry is None:
                # 容量超限、断线等情况不能丢掉随后到来的退出原因及最终反馈。
                entry = {"monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
                         "label": str(label), "details": _plain(details), "trigger": bool(trigger)}
                self._context["after_capture_events"].append(entry)
            return entry

    def stop(self):
        """供 reader 断开时结束接收；会话仍保留，等待界面 finish 归档。"""
        self.recorder.stop()

    def status(self):
        with self._lock:
            state = self.recorder.status()
            state.update(automatic_active=self._active,
                         automatic_saved=bool(self.last_export and self.last_export.saved),
                         automatic_path=str(self.last_export.path) if self.last_export else None,
                         automatic_error=(self.last_export.error if self.last_export else None)
                                         or self._context.get("start_error"))
            return state

    def finish(self, reason, context=None):
        with self._lock:
            if not self._active:
                return self.last_export
            self.event("自动诊断结束", {"reason": str(reason), "context": context})
            before_stop = self.recorder.status()
            self.recorder.stop()
            if self.reader is not None:
                self.reader.attach_diagnostics(None)
            self._context.update(end_monotonic_ns=time.monotonic_ns(), end_wall_time_ns=time.time_ns(),
                                 end_reason=str(reason), end_context=_plain(context),
                                 capture_state_at_end=before_stop)
            issues = []
            if self._context.get("start_error"):
                issues.append(self._context["start_error"])
            if before_stop["error"]:
                issues.append(before_stop["error"])
            if not before_stop["running"] and not self._context.get("start_error"):
                issues.append("流程结束前 CAN 接收已停止，停止之后没有完整 CAN 记录。")
            if not before_stop["frames"]:
                issues.append("本轮没有采集到 CAN 报文。")
            for key, label in (("kernel_drops", "内核报告丢帧"), ("truncated_packets", "接收报文被截断"),
                               ("malformed_packets", "接收报文长度异常")):
                if before_stop[key]:
                    issues.append(f"{label}：{before_stop[key]}")
            if before_stop["missing_ids"]:
                issues.append("本轮未采到反馈：" + "、".join(before_stop["missing_ids"]))
            if before_stop["stale_ids"]:
                issues.append("结束时反馈超过 0.5 秒未更新：" + "、".join(before_stop["stale_ids"]))
            self._context["capture_issues"] = issues
            self.last_export = AutomaticDiagnosticExport(self.output_dir, self.recorder, self._context)
            self.reader, self._active = None, False
            return self.last_export
