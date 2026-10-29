"""选孔的坐标链、曝光邻近反馈与针尖接近路径；全部使用米。"""
import json
import re
import threading
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(value):
    T = np.asarray(value, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4)
            or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-4):
        raise ValueError('位姿矩阵必须是有限数值的刚体变换。')
    return T.copy()


def load_handeye(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if (data.get('schema') != 'handeye_result_v2'
            or data.get('configuration') != 'eye_in_hand'
            or data.get('translation_unit') != 'm'
            or data.get('transform_direction') != 'tcp_to_camera'):
        raise ValueError('请选择 VLA 手眼标定导出的 eye_in_hand / handeye_result_v2 文件；不能使用旧 UR 安装矩阵。')
    # 导出名 T_cam_tcp 的方向是 TCP→相机；先求相机→标定时的 TCP。
    T = transform(data['tcp_definition']['T_flange_tcp']) @ np.linalg.inv(transform(data['result']['T_cam_tcp']))
    camera = data.get('camera') or {}
    serial = camera.get('serial_number') or camera.get('serial')
    if not serial:
        match = re.search(r'(\d{12})', str(camera.get('device', '')))
        serial = match.group(1) if match else None
    return dict(T_flange_camera=transform(T), serial=str(serial) if serial else None, payload=data)


class PoseHistory:
    def __init__(self):
        self._items = deque(maxlen=400)
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._items.clear()

    def add(self, snapshot):
        item = (float(snapshot['monotonic']), transform(snapshot['T_base_flange']))
        with self._lock:
            if self._items and item[0] <= self._items[-1][0]:
                return
            self._items.append(item)

    def at(self, stamp, tolerance=.10):
        with self._lock:
            values = list(self._items)
        if not values:
            raise ValueError('尚无与画面同步的机器人反馈。')
        times = np.asarray([v[0] for v in values])
        i = int(np.searchsorted(times, stamp))
        if i == 0 or i == len(values):
            nearest = values[0 if i == 0 else -1]
            if abs(stamp - nearest[0]) > tolerance:
                raise ValueError('图像时间附近没有新鲜机器人反馈；本次目标不可用于运动。')
            return nearest[1].copy()
        a, b = values[i-1], values[i]
        if max(stamp - a[0], b[0] - stamp) > tolerance:
            raise ValueError('图像与机器人反馈间隔过大。')
        fraction = (stamp - a[0]) / (b[0] - a[0])
        T = np.eye(4)
        T[:3, 3] = a[1][:3, 3] * (1-fraction) + b[1][:3, 3] * fraction
        T[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[1][:3, :3], b[1][:3, :3]])))(fraction).as_matrix()
        return T

    def stable(self, stamp, window=.7):
        with self._lock:
            values = [item for item in self._items if stamp-window-.05 <= item[0] <= stamp]
        if len(values) < 8 or values[-1][0]-values[0][0] < window-.05:
            return False
        if np.max(np.diff([v[0] for v in values])) > .15:
            return False
        poses = [v[1] for v in values]
        positions = np.stack([p[:3,3] for p in poses])
        if np.linalg.norm(positions[:,None]-positions[None,:],axis=2).max() > .0001:
            return False
        for i, a in enumerate(poses):
            if any(np.rad2deg(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()) > .1 for b in poses[i+1:]):
                return False
        return True


def tool_pose(point, outward_normal, current_R):
    """探针沿法兰 +Z 伸出，使 +Z 指向孔内，并选择最接近当前的绕轴角。"""
    z = -np.asarray(outward_normal, float)
    z /= np.linalg.norm(z)
    x = np.asarray(current_R)[:, 0]
    x = x - z * np.dot(x, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.asarray(current_R)[:, 1], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = point
    return transform(T)


def hole_target(T_base_cad, hole, current_R, offset_m):
    T = transform(T_base_cad)
    normal = T[:3, :3] @ np.asarray(hole['normal_cad'], float)
    point = (T @ np.r_[hole['center_cad_m'], 1])[:3]
    return tool_pose(point + offset_m * normal, normal, current_R)


def travel_targets(start_tcp, target_tcp, clearance_m=.05, *, align_after_approach=False):
    """先保姿退针、抬升和横移，下降时调姿，最后沿目标针轴接近。

    align_after_approach 时先以当前朝向抵达目标位置，再保持针尖位置调姿。
    这是可供现场核对的折线路径，不含环境碰撞模型。
    """
    start, end = transform(start_tcp), transform(target_tcp)
    # 先沿当前探针轴回退，随后抬高；避免针尖可能在孔内时先横移。
    retreat = start.copy()
    retreat[:3, 3] -= start[:3, 2] * clearance_m
    entry = end.copy()
    # 退针/抬升距离不应同时强制提前那么远对准目标；远处对准可能顶住腕部限位。
    # 最后至多 20 mm 保持目标姿态沿轴接近，之前允许针尖边移动边转向。
    entry[:3, 3] -= end[:3, 2] * min(clearance_m, .02)
    height = max(retreat[2, 3], entry[2, 3], start[2, 3]) + clearance_m
    raised = retreat.copy(); raised[2, 3] = height
    above = entry.copy(); above[2, 3] = height
    above[:3, :3] = start[:3, :3]
    if align_after_approach:
        near = end.copy(); near[:3, :3] = start[:3, :3]
        above[:2, 3] = near[:2, 3]
        return [retreat, raised, above, near, end]
    return [retreat, raised, above, entry, end]


def precision_problem(tcp):
    if tcp is None:
        return '请先载入已保存的 TCP 标定结果。'
    if tcp['rms_mm'] > .05 or tcp['max_mm'] > .10:
        return (f"TCP RMS {tcp['rms_mm']:.3f} mm / 最大 {tcp['max_mm']:.3f} mm，"
                '超过本任务的 0.05 / 0.10 mm 入口门槛。可预览和接近观察位，回到接触点与插孔暂不可执行。')
    return None
