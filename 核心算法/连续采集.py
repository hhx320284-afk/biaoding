"""人工确认触点后的一次性停稳采点；不发送运动命令、不判断物理接触。

start 只开启会话。每次采点都必须先 confirm_contact，确认后的第一帧
新反馈建立基准并开始稳定窗口；连续稳定 1 秒才记录一条，随后撤销确认。偏离基准或窗口内
移动超过 0.05 mm / 0.05° 即撤销确认，重新停稳也不会自行采点。

反馈间隔超过 0.25 秒、读数过期超过 0.5 秒或时间异常立即停采，恢复
反馈不会自动继续，须再次 start。完全没有 push 时，调用方仍须用
实时反馈超时检查暂停采集；本类不创建后台线程或定时器。
"""

from __future__ import annotations

import math
import time

import numpy as np

from 四点标定 import rotation_matrix
from 采点引导 import StablePoseCollector


class ContinuousCaptureGate:
    """输入位姿为毫米、RPY 度；sampled_at 与 now 均使用 monotonic 时钟。"""

    def __init__(self):
        self.collector = StablePoseCollector()
        self.active = False
        self.armed = False
        self.failure = None
        self.fatal = False
        self._started_at = None
        self._last_timestamp = None
        self._confirmation = None
        self._confirmation_after = None
        self._reference_pose = None
        self._reference_rotation = None
        self._stability = self._empty_stability("stopped", "尚未开始采集。")

    @staticmethod
    def _empty_stability(reason, message):
        return {
            "stable": False, "progress": 0.0, "duration_s": 0.0,
            "count": 0, "translation_range_mm": 0.0,
            "rotation_range_deg": 0.0, "pose": None,
            "reason": reason, "message": message,
        }

    def _result(self, message, capture=False, pose=None, accepted=False, confirmation=None):
        current = self._confirmation if confirmation is None else confirmation
        return {
            "capture": capture, "pose": pose,
            "stability": dict(self._stability), "message": message,
            "active": self.active, "fatal": self.fatal, "failure": self.failure,
            "armed": self.armed, "accepted": accepted,
            "confirmation": (None if current is None else {
                "monotonic": current["monotonic"], "sampled_at": current["sampled_at"],
                "pose_mm_rpy": list(current["pose_mm_rpy"]),
                "reference_sampled_at": current["reference_sampled_at"],
                "reference_pose_mm_rpy": (None if current["reference_pose_mm_rpy"] is None
                                           else list(current["reference_pose_mm_rpy"])),
            }),
        }

    def _clear_confirmation(self):
        self.armed = False
        self._confirmation = None
        self._confirmation_after = None
        self._reference_pose = None
        self._reference_rotation = None
        self.collector.reset()

    def _cancel_confirmation(self, message, translation=0.0, rotation=0.0):
        message = (f"{message} 本次位移 {translation:.6f} mm（上限 {self.collector.max_translation_mm:g} mm），"
                   f"转角 {rotation:.6f}°（上限 {self.collector.max_rotation_deg:g}°）。")
        self._clear_confirmation()
        self._stability = self._empty_stability("confirmation_cancelled", message)
        self._stability.update(translation_range_mm=float(translation), rotation_range_deg=float(rotation))
        return self._result(message)

    @staticmethod
    def _sample(pose_mm_rpy, sampled_at, now):
        timestamp = float(sampled_at)
        now = time.monotonic() if now is None else float(now)
        pose = np.asarray(pose_mm_rpy, dtype=float)
        if (not math.isfinite(timestamp) or not math.isfinite(now)
                or pose.shape != (6,) or not np.isfinite(pose).all()):
            raise ValueError("位姿或时间无效。")
        rotation = rotation_matrix(pose[3:], "rpy_deg")
        return pose, rotation, timestamp, now

    def start(self, now=None):
        """开启／恢复会话，但尚未授权采点；清除旧确认和稳定窗口。"""
        now = time.monotonic() if now is None else float(now)
        if not math.isfinite(now):
            raise ValueError("开始采集的时间必须为有限数值。")
        self._clear_confirmation()
        self.active = True
        self.failure = None
        self.fatal = False
        self._started_at = now
        self._last_timestamp = None
        message = "采集会话已开始。请将针尖对准同一固定点，再点击“确认触点并采集”。"
        self._stability = self._empty_stability("await_confirmation", message)
        return self._result(message)

    def confirm_contact(self, pose_mm_rpy, sampled_at, now=None):
        """人工确认一次触点；点击时缓存只供追溯，第一帧新反馈才建立基准。"""
        if not self.active:
            return self._result("请先开始采集会话，再确认触点。")
        if self.armed:
            return self._result("本次触点已确认，正在检查稳定性；重复确认不会延长或重置计时。")
        try:
            pose, rotation, timestamp, now = self._sample(pose_mm_rpy, sampled_at, now)
        except (TypeError, ValueError, OverflowError):
            return self._fail("invalid", "确认所用位姿或时间无效，已暂停采集；检查连接后重新开始。")
        age = now - timestamp
        if age > self.collector.max_age_s or age < -0.01:
            return self._fail("stale", "确认所用反馈已过期或时间异常，已暂停采集；恢复实时反馈后重新开始。")
        if self._last_timestamp is not None:
            if timestamp < self._last_timestamp:
                return self._fail("timestamp_regressed", "确认所用反馈时间倒退，已暂停采集；检查连接后重新开始。")
            if timestamp - self._last_timestamp > self.collector.max_gap_s:
                return self._fail("gap", "反馈中断超过 0.25 秒，已暂停采集；检查 CAN 后重新开始。")
        self.collector.reset()
        self.armed = True
        self._last_timestamp = timestamp
        self._reference_pose = None
        self._reference_rotation = None
        self._confirmation = {"monotonic": now, "sampled_at": timestamp, "pose_mm_rpy": pose.tolist(),
                              "reference_pose_mm_rpy": None, "reference_sampled_at": None}
        self._confirmation_after = max(now, timestamp)
        message = "触点已确认。请保持针尖不动，等待确认之后的新反馈连续稳定 1 秒；移动将撤销本次确认。"
        self._stability = self._empty_stability("waiting", message)
        return self._result(message, accepted=True)

    def stop(self):
        self.active = False
        self._clear_confirmation()
        self._last_timestamp = None
        message = "采集已暂停。"
        self._stability = self._empty_stability("stopped", message)
        return self._result(message)

    def _fail(self, reason, message):
        self.active = False
        self.fatal = True
        self.failure = reason
        self._clear_confirmation()
        self._stability = self._empty_stability(reason, message)
        return self._result(message)

    def push(self, pose_mm_rpy, sampled_at, now=None):
        """处理新反馈；一次确认最多返回一次 capture=True，使用最后一帧。"""
        if not self.active:
            return self._result(self._stability["message"])
        try:
            pose, rotation, timestamp, now = self._sample(pose_mm_rpy, sampled_at, now)
        except (TypeError, ValueError, OverflowError):
            return self._fail("invalid", "反馈位姿或时间无效，已暂停采集；检查连接后重新开始。")

        age = now - timestamp
        if age > self.collector.max_age_s or age < -0.01:
            return self._fail("stale", "反馈过期或时间异常，已暂停采集；恢复实时反馈后重新开始。")
        duplicate = self._last_timestamp == timestamp
        if self._last_timestamp is not None:
            if timestamp < self._last_timestamp:
                return self._fail("timestamp_regressed", "反馈时间倒退，已暂停采集；检查连接与时钟后重新开始。")
            if timestamp - self._last_timestamp > self.collector.max_gap_s:
                return self._fail("gap", "反馈中断超过 0.25 秒，已暂停采集；检查 CAN 后重新开始。")
        self._last_timestamp = timestamp
        if not self.armed:
            # 撤销原因必须保留到下一次明确操作，不能被下一帧未确认状态冲掉。
            if self._stability["reason"] == "confirmation_cancelled":
                return self._result(self._stability["message"])
            message = "尚未确认本次触点，不会采点。请将针尖对准同一固定点，再点击“确认触点并采集”。"
            self._stability = self._empty_stability("await_confirmation", message)
            return self._result(message)
        if timestamp <= self._confirmation_after:
            return self._result("正在等待点击确认之后的新反馈；此前的缓存不计入稳定时长。")
        if self._reference_pose is None:
            # 点击时显示的反馈可能稍旧，不能拿它判断点击之后是否移动。
            self._reference_pose = pose.copy()
            self._reference_rotation = rotation.copy()
            self._confirmation["reference_pose_mm_rpy"] = pose.tolist()
            self._confirmation["reference_sampled_at"] = timestamp
        displacement = float(np.linalg.norm(pose[:3] - self._reference_pose[:3]))
        relative = self._reference_rotation.T @ rotation
        sine = 0.5 * math.hypot(relative[2, 1] - relative[1, 2],
                                relative[0, 2] - relative[2, 0],
                                relative[1, 0] - relative[0, 1])
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.degrees(math.atan2(sine, cosine))
        if (displacement > self.collector.max_translation_mm + 1e-9
                or angle > self.collector.max_rotation_deg + 1e-9):
            return self._cancel_confirmation(
                "确认后发生移动，已撤销本次触点确认，未采点。请重新对准固定点并确认。", displacement, angle)
        if duplicate:
            return self._result("尚未收到新反馈，本次缓存读数不计入稳定时长。")

        self._stability = self.collector.update(pose, timestamp, now)
        if self._stability["reason"] in ("invalid", "stale", "gap"):
            return self._fail(self._stability["reason"], "反馈异常，已暂停采集；检查连接后重新开始。")
        if self._stability["reason"] == "moving":
            return self._cancel_confirmation(
                "确认后的稳定窗口内发生移动，已撤销本次确认，未采点。请重新对准固定点并确认。",
                self._stability["translation_range_mm"], self._stability["rotation_range_deg"])
        if not self._stability["stable"]:
            return self._result(self._stability["message"])
        captured = list(self._stability["pose"])
        confirmation = self._confirmation
        self._clear_confirmation()
        return self._result("已记录本次确认的一条数据。下一姿态须重新对准固定点并确认，不会因停稳自行采点。",
                            capture=True, pose=captured, confirmation=confirmation)
