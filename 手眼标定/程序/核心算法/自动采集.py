"""停稳采样门控：仅接受连续、新鲜的反馈和新姿态，不控制机械臂。"""
from collections import deque
import math

import numpy as np

from 手眼算法 import validate_transform


def pose_distance(first, second):
    first, second = np.asarray(first), np.asarray(second)
    millimeters = float(np.linalg.norm(first[:3, 3] - second[:3, 3]) * 1000)
    cosine = (np.trace(first[:3, :3].T @ second[:3, :3]) - 1) / 2
    degrees = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
    return millimeters, degrees


class CaptureGate:
    def __init__(self):
        self.history = deque(maxlen=100)

    def clear(self):
        self.history.clear()

    def push(self, feedback, now):
        stamp = float(feedback['monotonic'])
        if not math.isfinite(stamp) or not -0.05 <= now - stamp <= 0.25:
            self.clear()
            raise ValueError('机械臂反馈已过期，暂停采集。')
        pose = validate_transform(feedback['T_base_end'], '法兰位姿')
        if self.history and stamp <= self.history[-1][0]:
            return False  # 重复读取同一反馈不能计作持续停稳。
        if self.history and stamp - self.history[-1][0] > 0.25:
            self.clear()
        self.history.append((stamp, pose, feedback))
        return True

    def ready(self, now, frame_time, wait_seconds, accepted, move_mm, turn_deg):
        if len(self.history) < 3:
            return None, '正在读取机械臂，等待停稳…'
        newest = self.history[-1][0]
        if now - newest > 0.25:
            return None, '等待新鲜机械臂反馈…'
        # 相机时间是主机收帧时间；需在停稳窗口之后取新帧。
        if not math.isfinite(frame_time) or not 0 <= now - frame_time <= 0.2:
            return None, '等待新的相机画面…'
        start = min(frame_time, newest) - wait_seconds
        before = [entry for entry in self.history if entry[0] <= start]
        if not before:
            return None, f'等待机械臂连续停稳 {wait_seconds:g} 秒…'
        anchor = before[-1]
        window = [entry for entry in self.history if entry[0] >= anchor[0]]
        if any(b[0] - a[0] > 0.25 for a, b in zip(window, window[1:])):
            return None, '机械臂反馈不连续，等待重新停稳…'
        # 检查整个时间窗，不能仅比较相邻帧而放过慢速持续运动。
        if any(pose_distance(anchor[1], item[1])[0] > 0.8 or
               pose_distance(anchor[1], item[1])[1] > 0.3 for item in window):
            return None, '机械臂正在移动，停稳后自动采集…'
        nearest = min(window, key=lambda item: abs(item[0] - frame_time))
        if abs(nearest[0] - frame_time) > 0.15:
            return None, '图像与机械臂时间差偏大，等待下一帧…'
        for previous in accepted:
            movement, rotation = pose_distance(nearest[1], previous)
            if movement < move_mm and rotation < turn_deg:
                return None, '此姿态已经采过，请移动或转动机械臂。'
        return nearest[2], ''
