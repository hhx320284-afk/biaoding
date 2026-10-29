"""真实位姿流的稳定采集与逐点审查；本模块不发送机械臂运动命令。"""

from collections import deque
import math
import time

import numpy as np

from 四点标定 import review_samples, rotation_matrix, validate_criteria


DEFAULT_STABILITY = {
    "window_s": 1.0,
    "max_translation_mm": 0.05,
    "max_rotation_deg": 0.05,
    "max_gap_s": 0.25,
    "max_age_s": 0.5,
    "min_samples": 6,
}


def _rotation_angle(rotation):
    sine = 0.5 * math.hypot(rotation[2, 1] - rotation[1, 2],
                            rotation[0, 2] - rotation[2, 0],
                            rotation[1, 0] - rotation[0, 1])
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine))


def matrix_to_pose(matrix, translation_unit="m"):
    """将 base_from_flange 刚性矩阵转为 [X,Y,Z,roll,pitch,yaw]。

    输出固定为毫米、度；旋转约定为 Rz(yaw) @ Ry(pitch) @ Rx(roll)。
    translation_unit 明确指定输入矩阵的平移单位，不猜测数据单位。
    奇异俯仰姿态将 roll 定为零，并求一个等价 yaw；不平均欧拉角。
    """
    if translation_unit not in ("m", "mm"):
        raise ValueError("矩阵平移单位必须为 m 或 mm。")
    try:
        transform = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("法兰位姿矩阵必须包含有效数字。") from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("法兰位姿必须是 4×4 有限数值矩阵。")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("法兰位姿矩阵的末行必须为 [0, 0, 0, 1]。")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("法兰位姿中的旋转不是有效刚性旋转，请检查数据来源和矩阵顺序。")
    cosine_pitch = math.hypot(rotation[0, 0], rotation[1, 0])
    pitch = math.atan2(-rotation[2, 0], cosine_pitch)
    if cosine_pitch > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    position = transform[:3, 3] * (1000.0 if translation_unit == "m" else 1.0)
    if not np.isfinite(position).all():
        raise ValueError("换算后的法兰位置无效，请检查矩阵平移单位。")
    return position.tolist() + np.rad2deg([roll, pitch, yaw]).tolist()


class StablePoseCollector:
    """只在持续收到新的、新鲜的读数时认定稳定，并返回最后一帧真实位姿。

    sampled_at 必须由读取设备数据的线程在每次真实新读数到达时用
    time.monotonic() 记录；不能由 UI 重读缓存时生成。now 使用同一时钟。
    接触需用户独立确认：静止不等于针尖已接触正确的固定点。
    """

    def __init__(self, window_s=1.0, max_translation_mm=0.05,
                 max_rotation_deg=0.05, max_gap_s=0.25, max_age_s=0.5,
                 min_samples=6):
        for name, value in (("window_s", window_s), ("max_translation_mm", max_translation_mm),
                            ("max_rotation_deg", max_rotation_deg), ("max_gap_s", max_gap_s),
                            ("max_age_s", max_age_s)):
            try:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("稳定采集的时间与抖动阈值必须为有限正数。") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError("稳定采集的时间与抖动阈值必须为有限正数。")
            setattr(self, name, value)
        try:
            valid_count = (not isinstance(min_samples, (bool, np.bool_))
                           and math.isfinite(float(min_samples))
                           and int(min_samples) == min_samples and min_samples >= 2)
        except (TypeError, ValueError, OverflowError):
            valid_count = False
        if not valid_count:
            raise ValueError("稳定采集至少需要 2 个连续新样本。")
        self.min_samples = int(min_samples)
        self.reset()

    def reset(self):
        self._samples = deque()
        self._last_timestamp = None

    def _status(self, reason, message, translation_range=0.0, rotation_range=0.0):
        duration = self._samples[-1][0] - self._samples[0][0] if len(self._samples) > 1 else 0.0
        stable = (reason == "stable")
        progress = min(1.0, duration / self.window_s, len(self._samples) / self.min_samples)
        return {
            "stable": stable,
            "progress": progress,
            "duration_s": duration,
            "count": len(self._samples),
            "translation_range_mm": float(translation_range),
            "rotation_range_deg": float(rotation_range),
            "pose": self._samples[-1][1].tolist() if stable else None,
            "reason": reason,
            "message": message,
        }

    def update(self, pose_mm_rpy, sampled_at, now=None):
        """加入一帧真实新样本；稳定返回 pose，等待或拒绝返回 pose=None。"""
        try:
            timestamp = float(sampled_at)
            now = time.monotonic() if now is None else float(now)
            pose = np.asarray(pose_mm_rpy, dtype=float)
            if (not math.isfinite(timestamp) or not math.isfinite(now)
                    or pose.shape != (6,) or not np.isfinite(pose).all()):
                raise ValueError
            rotation = rotation_matrix(pose[3:], "rpy_deg")
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return self._status("invalid", "位姿或读数时间无效，请检查机械臂连接；未采点。")
        age = now - timestamp
        if age > self.max_age_s or age < -0.01:
            self.reset()
            return self._status("stale", "机械臂读数已过期或时间异常，等待新的实时位姿；未采点。")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            # 保留最后时间边界，避免反复提交同一缓存包重新累积稳定窗口。
            self._samples.clear()
            return self._status("duplicate", "尚未收到新的机械臂读数，稳定计时已重置；未采点。")
        gap = self._last_timestamp is not None and timestamp - self._last_timestamp > self.max_gap_s
        self._last_timestamp = timestamp
        if gap:
            self._samples.clear()
        self._samples.append((timestamp, pose.copy(), rotation))
        # 保留跨越窗口起点的样本，稳定判定实际覆盖不少于 window_s。
        while len(self._samples) > 2 and self._samples[1][0] <= timestamp - self.window_s:
            self._samples.popleft()
        translation_range = 0.0
        rotation_range = 0.0
        samples = list(self._samples)
        for first_index, first in enumerate(samples):
            for second in samples[first_index + 1:]:
                translation_range = max(translation_range, float(np.linalg.norm(first[1][:3] - second[1][:3])))
                rotation_range = max(rotation_range, _rotation_angle(first[2].T @ second[2]))
        if translation_range > self.max_translation_mm + 1e-9 or rotation_range > self.max_rotation_deg + 1e-9:
            self._samples.clear()
            self._samples.append((timestamp, pose.copy(), rotation))
            return self._status("moving", "位姿仍在变化，请停止调整、保持针尖在同一固定点，等待稳定后自动读数。",
                                translation_range, rotation_range)
        duration = timestamp - self._samples[0][0]
        if duration >= self.window_s - 1e-9 and len(self._samples) >= self.min_samples:
            return self._status("stable", "已连续稳定，使用最后一帧真实法兰位姿。", translation_range, rotation_range)
        message = ("读数中断后已重新开始计时，请保持针尖在同一固定点。" if gap
                   else f"正在检查稳定性，请保持不动（需连续 {self.window_s:g} 秒新读数）。")
        return self._status("gap" if gap else "waiting", message, translation_range, rotation_range)


def _axis_hint(rotations):
    """选择与当前主要转轴差异较大的法兰 X/Y 参考方向，仅作人工引导。"""
    if len(rotations) < 2:
        return "X"
    reference = rotations[0]
    # (R-I) 的最弱方向近似共同转轴；选择更不平行的 X/Y 倾斜方向。
    differences = np.vstack([reference.T @ rotation - np.eye(3) for rotation in rotations[1:]])
    _, _, vectors = np.linalg.svd(differences)
    axis = vectors[-1]
    return "Y" if abs(axis[0]) >= abs(axis[1]) else "X"


def review_partial_samples(samples, mode="rpy_deg", criteria=None):
    """逐点审查 1～4 组采点；前三点不因整体尚未满秩而拒收。

    编号对应调用者传入的列表，UI 重采中间行时须自行映射行号。
    guidance 不包含采点编号；closest_pair / suggested_point 均从 1 开始。
    四点时直接复用完整审查，不降低最后的秩和条件数要求。
    """
    limits = validate_criteria(criteria)
    review = {"passed": False, "complete": False, "checks": [], "criteria": limits,
              "min_angle_deg": None, "closest_pair": None, "angle_pairs": [],
              "rank": None, "condition_number": None, "suggested_axis": "X",
              "suggested_point": None, "guidance": "", "warnings": []}
    try:
        poses = np.asarray(samples, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 6 or not 1 <= len(poses) <= 4 or not np.isfinite(poses).all():
            raise ValueError("需要 1～4 组有效法兰位姿，每组包含 6 个有限数值。")
        rotations = [rotation_matrix(pose[3:], mode) for pose in poses]
    except (TypeError, ValueError, OverflowError) as exc:
        review["checks"] = [{"id": "input", "label": "采点输入有效", "passed": False,
                             "actual": None, "limit": "1～4 组有效位姿", "detail": str(exc)}]
        review["guidance"] = "未接收该采点，请检查机械臂位姿来源、姿态格式和位置单位。"
        return review
    count = len(poses)
    axis = _axis_hint(rotations)
    review["suggested_axis"] = axis
    adjustment = (f"以第一个已采姿态的法兰轴方向为参照，增加绕 {axis} 轴的倾角，"
                  f"与已有每个姿态的角差达到 {limits['min_angle_deg']:g}°；"
                  "调整倾角时配合平移，让针尖回到同一固定点，再确认触点。"
                  "不要只绕法兰原点旋转来代替针尖对点。")
    if count == 4:
        review.update(review_samples(poses, mode, limits))
        review["complete"] = True
        review["suggested_axis"] = axis
        if review["passed"]:
            review["guidance"] = "四点姿态审查通过，可自动计算并继续检查拟合残差。"
        elif any(check["id"] == "angle" and not check["passed"] for check in review["checks"]):
            review["suggested_point"] = review["closest_pair"][1]
            review["guidance"] = "存在过于相近的姿态，请重采提示的采点。" + adjustment
        else:
            review["suggested_point"] = count
            review["guidance"] = "当前姿态分布不足，请重采一个姿态，避免继续沿同一轴改变。" + adjustment
        return review
    checks = review["checks"]
    checks.append({"id": "input", "label": "采点输入有效", "passed": True,
                   "actual": count, "limit": "1～4 组有效位姿", "detail": "已读取有效法兰位姿。"})
    for first in range(count):
        for second in range(first + 1, count):
            review["angle_pairs"].append({"points": [first + 1, second + 1],
                                           "angle_deg": _rotation_angle(rotations[first].T @ rotations[second])})
    if review["angle_pairs"]:
        closest = min(review["angle_pairs"], key=lambda pair: pair["angle_deg"])
        review["closest_pair"] = closest["points"]
        review["min_angle_deg"] = closest["angle_deg"]
        angle_passed = closest["angle_deg"] >= limits["min_angle_deg"] - 1e-9
        checks.append({"id": "angle", "label": "与已有采点的姿态角差", "passed": angle_passed,
                       "actual": closest["angle_deg"], "limit": limits["min_angle_deg"],
                       "detail": "两两姿态角差满足门槛。" if angle_passed else "姿态与已有采点过于相近，需要改变倾角。"})
    matrix = np.concatenate([np.vstack(rotations), -np.tile(np.eye(3), (count, 1))], axis=1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular_values[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    review["rank"] = rank
    if rank == 6:
        review["condition_number"] = float(singular_values[0] / singular_values[-1])
    if count >= 3 and rank < 6:
        review["warnings"].append("当前采点主要沿同一转轴变化；下一点需要增加另一方向的倾斜。前三点暂不要求满秩。")
    elif count >= 3 and review["condition_number"] > limits["max_condition_number"]:
        review["warnings"].append("当前倾角跨度偏小；下一点增加不同方向的倾斜，第四点再检查整体条件数。")
    review["passed"] = all(check["passed"] for check in checks)
    if review["passed"]:
        review["guidance"] = "本点审查通过。" + ("".join(review["warnings"]) or "请调整到下一种姿态。") + adjustment
    else:
        review["suggested_point"] = review["closest_pair"][1]
        review["guidance"] = "该组采点存在过于相近的姿态，请调整后重采。" + adjustment
    return review
