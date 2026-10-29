"""PiPER / UR 只读反馈；逐点确认触点后检查停稳，不发送运动指令。"""

import queue
import ipaddress
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from 机械臂采集 import PiperReader
from UR机械臂采集 import URReader
from 连续采集 import ContinuousCaptureGate
from 采点引导 import matrix_to_pose

FIRMWARES = {
    "≤1.8.2 · default": "default", "1.8.3—1.8.7 · v183": "v183",
    "1.8.8 · v188": "v188", "≥1.8.9 · v189": "v189",
}
ROBOT_CHANNEL = "can1"
ROBOT_KINDS = ('PiPER X（右臂 can1）', 'UR（网络连接）')


class AutoCaptureMixin:
    def robot_identity(self):
        if self.robot_kind.get() == ROBOT_KINDS[1]:
            host = self.ur_host.get().strip()
            try:
                host = str(ipaddress.ip_address(host))
            except ValueError:
                pass  # 连接时由 URReader 给出中文输入错误。
            return {'model': 'ur', 'robot_model': self.ur_model.get().strip() or '未指定',
                    'host': host, 'channel': host, 'interface': 'secondary_read_only'}
        return {'model': 'piper_x', 'arm': 'right', 'channel': ROBOT_CHANNEL,
                'firmware': FIRMWARES[self.robot_firmware.get()]}

    def robot_label(self):
        return 'UR' if self.robot_kind.get() == ROBOT_KINDS[1] else 'PiPER X · can1'

    def robot_connection_text(self):
        if self.robot_connecting:
            return '取消连接'
        return self.robot_label() + (' 已连接 · 断开' if self.robot_connected else ' · 连接')

    def _records_robot(self):
        identity = self.robot_identity()
        if any(any(record.get('source', {}).get(key) != value for key, value in identity.items())
               for record in self.records):
            raise ValueError('本轮记录与当前机械臂不一致，请先导出并清空本轮，不能混合不同机械臂的数据。')
        return identity

    def _robot_selection_changed(self, *_):
        selected = self.robot_kind.get()
        if selected == self._selected_robot_kind:
            return
        if (selected not in ROBOT_KINDS or self.robot_connected or self.robot_connecting
                or self.records or self.tasks_busy()):
            self.robot_kind.set(self._selected_robot_kind)
            self._status('请先结束任务、断开机械臂；本轮有数据时先导出并清空，再切换机械臂。', True)
            return
        self._selected_robot_kind = selected
        self.live_snapshot = None
        self.hole_panel.history.clear()
        self.hole_panel._stop_vision()
        self.hole_panel.mount_checked.set(False)
        self.hole_panel._mount_state = False
        self.hole_panel._settings_changed()
        self.board_panel._stop_detection()
        self.board_panel.control_ready = False
        self.board_panel.invalidate('机械臂选择已变化，旧目标和路径已作废。')
        self._robot_fields()
        self.robot_status.configure(text=self.robot_label()+' 未连接', fg=self.colors['muted'])
        self.live_pose.configure(text='XYZ（mm） —\nRPY（°） —')
        self._status('UR：填写控制柜 IP 后连接；型号可稍后填写。当前支持 TCP 标定采集，示教请使用 UR 示教器。'
                     if selected == ROBOT_KINDS[1] else '已切换 PiPER X · can1，请选择匹配固件后连接。')
        self._controls()

    def _init_capture(self):
        self.robot_events = queue.Queue(maxsize=32)
        self.robot_generation = 0
        self.robot_stop = threading.Event()
        self.robot_connected = self.robot_connecting = False
        self.live_snapshot = None
        self.robot_reader = None
        self.collecting = False
        self.contact_confirmation = None
        self.gate = ContinuousCaptureGate()
        self._closing = False

    def _queue_robot(self, generation, kind, data=None):
        try:
            self.robot_events.put_nowait((generation, kind, data))
        except queue.Full:
            try:
                self.robot_events.get_nowait()
            except queue.Empty:
                pass
            self.robot_events.put_nowait((generation, kind, data))

    def toggle_robot(self):
        if self.teaching_recovery_busy:
            self._status('正在恢复示教，请等待状态确认后再切换连接。', error=True)
            return
        if self.robot_connected or self.robot_connecting:
            self.disconnect_robot()
            return
        try:
            identity = self._records_robot()
        except ValueError as exc:
            self._status(str(exc), error=True)
            return
        generation = self.robot_generation = self.robot_generation + 1
        stop = self.robot_stop = threading.Event()
        label = self.robot_label()
        self.robot_connecting = True
        self.robot_status.configure(text=f"正在连接 {label}，仅读取反馈…", fg=self.colors["muted"])
        self._controls()

        def worker():
            reader = None
            try:
                reader = (URReader(host=identity['host'], robot_model=identity['robot_model'])
                          if identity['model'] == 'ur' else
                          PiperReader(channel=ROBOT_CHANNEL, firmware=identity['firmware']))
                reader.connect()
                if stop.is_set():
                    return
                self._queue_robot(generation, "connected", reader)
                while not stop.is_set():
                    self._queue_robot(generation, "pose", reader.read_pose())
                    stop.wait(0.05)
            except Exception as exc:
                if not stop.is_set():
                    self.logger.warning("%s 读取失败：%s", label, exc)
                    self._queue_robot(generation, "error", str(exc))
            finally:
                if reader is not None:
                    try:
                        reader.disconnect()
                    except Exception:
                        self.logger.exception("断开机械臂只读连接失败")

        threading.Thread(target=worker, name="TCP机械臂只读", daemon=True).start()

    def disconnect_robot(self):
        if self.teaching_recovery_busy:
            self.teaching_recovery_cancelled = True
        if self.robot_reader is not None:
            self.robot_reader.cancel_teaching_recovery()
        # 先经共享连接同步发送停止，再通知反馈线程断开 CAN。
        for panel in self._motion_panels():
            if panel.busy:
                panel.stop_motion('机械臂断开，已请求停止任务。')
        if hasattr(self, 'hole_panel'):
            self.hole_panel.history.clear()
        if hasattr(self, 'board_panel'):
            self.board_panel.control_ready = False
            self.board_panel.invalidate('机械臂连接已断开，请重新锁定标定板。')
        self.robot_reader = None
        self.robot_stop.set()
        self.robot_generation += 1
        self.robot_connected = self.robot_connecting = False
        self.live_snapshot = None
        if self.collecting:
            self.stop_capture("机械臂连接已断开；已采数据保留，请恢复连接后继续。")
        self.robot_status.configure(text=self.robot_label()+" 未连接", fg=self.colors["muted"])
        self.live_pose.configure(text="XYZ（mm） —\nRPY（°） —")
        self._controls()

    def start_capture(self):
        blocked = self.task_blocked('capture')
        if blocked:
            self._status(blocked, error=True)
            return False
        if self.collecting or self.audit_busy:
            return False
        if not self.robot_connected or self.live_snapshot is None:
            self._status("请先连接所选机械臂，确认已收到实时法兰位姿。", error=True)
            return False
        if not 0 <= time.monotonic() - self.live_snapshot["monotonic"] <= 0.5:
            self._status("机械臂反馈已过期，请检查连接。", error=True)
            return False
        if not self.camera.is_live():
            self._status("请先点击“打开相机”，看到实时画面后再开始采集。", error=True)
            return False
        try:
            self._records_robot()
            self._criteria()
        except ValueError as exc:
            self._status(str(exc), error=True)
            return False
        self.filter_value.set('全部数据')
        self._invalidate_review()
        self.collecting = True
        self.contact_confirmation = None
        self.gate.start()
        self.capture_state.configure(text="等待确认触点 · 尚未允许采点", fg=self.colors["muted"])
        self.stability_status.configure(text="每次确认只采一条；未确认时，停稳多久都不会记录。")
        self.guidance.configure(text="让针尖轻触同一个固定点，停止调整，再点击“确认触点并采集”。完成全部采点后点击“停止并审查”。")
        self._status("采集会话已开始。每个姿态都需先确认触点，程序再自动读数和审查。")
        self._controls()
        self.logger.info("用户开始 %s 逐点确认采集", self.robot_label())
        return True

    def confirm_contact(self):
        if not self.collecting or self.audit_busy or self.gate.armed:
            return False
        blocked = self.task_blocked('capture')
        if blocked:
            self._status(blocked, error=True)
            return False
        snapshot = self.live_snapshot
        if not self.robot_connected or snapshot is None or not self.camera.is_live():
            self.stop_capture("相机或机械臂反馈不可用，已停止采集；恢复连接后重新开始并确认触点。")
            return False
        pose = matrix_to_pose(snapshot["T_base_flange"], "m")
        outcome = self.gate.confirm_contact(pose, snapshot["monotonic"])
        if outcome["fatal"]:
            self.stop_capture(outcome["message"])
            return False
        if not outcome["accepted"]:
            self._status(outcome["message"], error=True)
            return False
        self.contact_confirmation = {
            "confirmed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="milliseconds"),
            "method": "operator_button", "feedback_timestamp": snapshot.get("timestamp"),
            **outcome["confirmation"],
        }
        self.capture_state.configure(text="已确认触点 · 正在检查停稳", fg=self.colors["green"])
        self.guidance.configure(text="请保持针尖轻触同一点。确认后连续停稳 1 秒采一条；位姿变化超限会取消本次确认。")
        self._status("已确认本次触点，开始检查新的反馈。")
        self._controls()
        self.logger.info("用户确认本次触点，等待停稳采样")
        return True

    def stop_capture(self, reason="采集已停止，正在自动审查全部数据。", review=True):
        was_running = self.collecting
        self.collecting = False
        self.contact_confirmation = None
        self.gate.stop()
        self.capture_state.configure(text="已停止 · 数据保留在本次窗口", fg=self.colors["muted"])
        self.guidance.configure(text=reason)
        self._status(reason)
        if was_running:
            self.logger.info("停止连续采集，本次窗口共有 %d 条", len(self.records))
        if review and self.records and not self._closing:
            self.review_records()
        self._controls()

    def _receive_snapshot(self, snapshot):
        try:
            identity = self.robot_identity()
            if (snapshot.get("frame") != "flange" or snapshot.get("translation_unit") != "m"
                    or snapshot.get("transform_direction") != "base_from_flange"
                    or snapshot.get("channel") != identity['channel']
                    or snapshot.get("model") != identity['model']
                    or (identity['model'] == 'ur' and snapshot.get('robot_model') != identity['robot_model'])):
                raise ValueError("反馈必须来自当前选择的机械臂，并为基座下法兰位姿，平移单位为米。")
            pose = matrix_to_pose(snapshot["T_base_flange"], "m")
            stamp = float(snapshot["monotonic"])
            if not 0 <= time.monotonic() - stamp <= 0.5:
                raise ValueError("机械臂反馈过期或时间无效。")
        except (KeyError, TypeError, ValueError) as exc:
            self.live_snapshot = None
            if self.collecting:
                self.stop_capture(str(exc) + " 已停止采集，请恢复反馈后重新开始。")
            self.robot_status.configure(text=str(exc), fg=self.colors["error"])
            return
        self.live_snapshot = snapshot
        if hasattr(self, 'hole_panel'):
            self.hole_panel.history.add(snapshot)
        self.robot_status.configure(text=self.robot_label()+" 已连接 · " + snapshot.get("pose_source_label", "法兰反馈"), fg=self.colors["green"])
        self.live_pose.configure(text="XYZ（mm） " + " / ".join(f"{v:.3f}" for v in pose[:3])
                                 + "\nRPY（°） " + " / ".join(f"{v:.2f}" for v in pose[3:]))
        if not self.collecting:
            return
        if not self.camera.is_live():
            self.stop_capture("相机画面已中断，采集已停止；恢复实时画面后重新开始。")
            return
        outcome = self.gate.push(pose, stamp)
        if outcome["fatal"]:
            self.stop_capture(outcome["message"])
            return
        stability = outcome["stability"]
        cancelled = stability["reason"] == "confirmation_cancelled"
        if cancelled:
            self.stability_status.configure(text=(
                f"取消时变化：位移 {stability['translation_range_mm']:.4f} mm / 上限 {self.gate.collector.max_translation_mm:g} mm；"
                f"转角 {stability['rotation_range_deg']:.4f}° / 上限 {self.gate.collector.max_rotation_deg:g}°"))
            self.capture_state.configure(text="本次未采到 · 确认已取消，请重新对点", fg=self.colors["warning"])
            if self.contact_confirmation is not None:
                self._status(outcome["message"], error=True)
                self.logger.warning("本次触点确认取消，未新增记录：%s", outcome["message"])
        else:
            self.stability_status.configure(text=(f"确认后停稳 {min(stability['duration_s'], 1):.1f} / 1 秒  ·  "
                f"位移 {stability['translation_range_mm']:.3f} mm  ·  转角 {stability['rotation_range_deg']:.3f}°")
                if outcome["armed"] else "等待确认触点；未确认时，停稳多久都不会记录。")
        self.guidance.configure(text=outcome["message"])
        if outcome["capture"]:
            if self.contact_confirmation is None or outcome["confirmation"] is None:
                self.stop_capture("缺少本次触点确认，未记录采样；请重新开始并确认触点。")
                return
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            record = {
                "id": self.next_record_id, "captured_at": now.isoformat(timespec="milliseconds"),
                "pose": outcome["pose"], "status": "待审查", "reason": "操作者已确认触点，并通过停稳检查；等待批量审查。",
                "residual_mm": None, "nearest_angle_deg": None,
                "source": {**identity, "pose_source": snapshot.get("pose_source"),
                    "timestamp": snapshot.get("timestamp"), "camera_preview_live": self.camera.is_live(),
                    "joint_angles": (list(snapshot["joint_angles"])
                                     if snapshot.get("joint_angles") is not None else None),
                    "joint_angle_unit": snapshot.get("joint_angle_unit"),
                    "packet_timestamps": list(snapshot.get("packet_timestamps", ())),
                    "feedback_snapshot": snapshot.get("feedback_snapshot"),
                    "feedback_sequence": snapshot.get("feedback_sequence"),
                    "sdk_revision": snapshot.get("sdk_revision"),
                    "controller_timestamp": snapshot.get("controller_timestamp"),
                    "T_flange_controller_tcp": snapshot.get("T_flange_controller_tcp"),
                    "controller_tcp_offset": snapshot.get("tcp_offset"),
                    "contact_confirmation": {**self.contact_confirmation, **outcome["confirmation"]},
                    "contact_basis": "操作者逐点点击确认触点；程序只核对停稳，没有接触传感器"},
                "stability": {key: stability[key] for key in ("duration_s", "count", "translation_range_mm", "rotation_range_deg")},
            }
            self.next_record_id += 1
            self.records.append(record)
            self._insert_record(record)
            self._update_counts()
            self.logger.info("触点确认采集记录 %d，等待批量审查", record["id"])
            self._status(f"已采集 {len(self.records)} 条。调整下一个姿态并重新确认触点，完成后停止并审查。")
        if not outcome["armed"]:
            self.contact_confirmation = None
            if not cancelled:
                self.capture_state.configure(text="等待确认下一次触点 · 当前不会采点", fg=self.colors["muted"])

    def _poll(self):
        if self._closing:
            return
        for _ in range(40):
            try:
                generation, kind, data = self.robot_events.get_nowait()
            except queue.Empty:
                break
            if generation != self.robot_generation:
                continue
            if kind == "connected":
                self.robot_reader = data
                self.robot_connecting = False
                self.robot_connected = True
                self.logger.info("%s 只读连接建立", self.robot_label())
            elif kind == "error":
                self.disconnect_robot()
                self.robot_status.configure(text=data, fg=self.colors["error"])
                self.guidance.configure(text=('请核对控制柜 IP、电脑网络与 UR 只读接口 30012。'
                    if self.robot_kind.get() == ROBOT_KINDS[1] else '请检查右臂 USB-CAN 和系统 can1 配置。'))
                self._status(data, error=True)
            elif kind == "pose" and self.robot_connected:
                self._receive_snapshot(data)
        if self.live_snapshot is not None and time.monotonic() - self.live_snapshot["monotonic"] > 0.5:
            self.live_snapshot = None
            if self.collecting:
                self.stop_capture("机械臂反馈已中断，采集已停止；恢复连接后点击开始采集。")
            self.robot_status.configure(text="机械臂反馈已过期，请检查连接。", fg=self.colors["error"])
        if self.collecting and not self.camera.is_live():
            self.stop_capture("相机画面已中断，采集已停止；恢复实时画面后重新开始。")
        self._poll_review()
        self._poll_teaching_recovery()
        self._controls()
        self.after(80, self._poll)
