"""PiPER X 探针路径计算与显式执行；不提供碰撞检测或接触判定。"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
import threading
import time
import traceback

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from 四点标定 import rotation_matrix, validate_criteria
from 机械臂采集 import MOTION_CONTROL_HOLD_REASON


class MotionError(RuntimeError):
    """供 UI 显示的规划或执行错误。"""


# 现场出现点击执行后立即向下的异常；执行链核实前禁止再次使能或下发目标。
# 此暂停只影响自动执行，保留只读反馈、预览和停止功能；不能由界面勾选绕过。
MOTION_EXECUTION_HOLD_REASON = MOTION_CONTROL_HOLD_REASON
BOARD_MIN_GAP_M = .005


def _transform(value, label="位姿"):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise MotionError(f"{label}必须是有限的 4×4 刚性变换。")
    return matrix.copy()


def _errors(actual, target):
    distance = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    angle = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()))
    return distance, angle


def load_tcp(path):
    """读取显式保存的验收结果；通过自定门槛不等于独立精度验证。"""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        audit = payload["audit"]
        result = payload["result"]
        if (payload.get("kind") != "accepted_tcp_calibration" or audit.get("passed") is not True
                or payload.get("position_unit") != "mm"
                or payload.get("pose_frame") != "base_from_flange"
                or payload.get("rotation_mode") != "rpy_deg"
                or payload.get("transform_direction") != "flange_from_tcp"
                or result.get("transform_direction") != "flange_from_tcp"
                or payload.get("robot", {}).get("model") != "piper_x"
                or payload["robot"].get("channel") != "can1"):
            raise MotionError("需要右臂 can1 已验收的 TCP 结果，不能使用未通过的采集报告。")
        tool = _transform(result["flange_to_tcp"], "TCP 偏移")
        offset = np.asarray(result["tcp_offset_mm"], dtype=float)
        pivot = np.asarray(result["reference_point_base_mm"], dtype=float)
        if (offset.shape != (3,) or pivot.shape != (3,) or not np.isfinite(pivot).all()
                or not np.isfinite(offset).all() or not np.allclose(tool[:3, 3], offset, atol=1e-6, rtol=0)):
            raise MotionError("TCP 偏移与矩阵不一致，或固定点坐标无效。")
        rms, maximum = float(result["rms_mm"]), float(result["max_error_mm"])
        limits = validate_criteria(audit["criteria"])
        if (not all(math.isfinite(v) and v >= 0 for v in (rms, maximum)) or rms > maximum + 1e-9
                or rms > float(limits["max_rms_mm"]) + 1e-9
                or maximum > float(limits["max_error_mm"]) + 1e-9):
            raise MotionError("保存结果的残差未通过其实际验收门槛。")
        eligible = audit["eligible_indices"]
        records = payload["records"]
        if not eligible or any(type(i) is not int or i < 0 or i >= len(records) for i in eligible):
            raise MotionError("保存结果缺少可信参考姿态。")
        reference = records[eligible[0]]
        source = reference.get("source", {})
        if (reference.get("status") != "合格" or source.get("channel") != "can1"
                or source.get("model") != "piper_x"):
            raise MotionError("参考采样未通过验收，或不是右臂 PiPER X。")
        pose = np.asarray(reference["pose"], dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise MotionError("参考采样位姿无效。")
        tool[:3, 3] *= .001
        pivot *= .001
        reference_tcp = np.eye(4)
        reference_tcp[:3, :3] = rotation_matrix(pose[3:], "rpy_deg") @ tool[:3, :3]
        reference_tcp[:3, 3] = pivot
        return {"T_flange_tcp": tool, "pivot_base": pivot, "reference_T_base_tcp": reference_tcp,
                "rms_mm": rms, "max_mm": maximum, "payload": payload, "path": str(Path(path))}
    except MotionError:
        raise
    except (OSError, KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise MotionError(f"TCP 结果格式无效：{exc}") from exc


def _snapshot(snapshot, fresh=False):
    if (not isinstance(snapshot, dict) or snapshot.get("model") != "piper_x"
            or snapshot.get("channel") != "can1" or snapshot.get("frame") != "flange"
            or snapshot.get("translation_unit") != "m"
            or snapshot.get("transform_direction") != "base_from_flange"
            or snapshot.get("joint_angle_unit") != "rad"):
        raise MotionError("需要右臂 can1 的实时法兰位姿及弧度关节反馈。")
    pose = _transform(snapshot["T_base_flange"], "实时法兰位姿")
    joints = np.asarray(snapshot.get("joint_angles"), dtype=float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise MotionError("缺少完整六关节反馈，不能预检查可达性。")
    stamp = float(snapshot["monotonic"])
    if not math.isfinite(stamp) or (fresh and not 0 <= time.monotonic() - stamp <= .3):
        raise MotionError("右臂反馈超过 0.3 秒或时间异常。")
    return pose, joints, stamp


def plan_tcp_path(start_snapshot, T_flange_tcp, targets_base_tcp, arm_fk=None, *,
                  free_orientation_segments=0, align_free_endpoint=True):
    """按针尖最多 5 mm / 2° 采样，不检测碰撞。

    free_orientation_segments 为允许途中调姿的前 N 段，默认全部严格保持目标姿态。
    第 N 段终点必须严格对准，后续段全程严格；至少保留最后一段严格接近。
    自由段仅将朝向作为偏好，针尖位置、关节限位和相邻姿态连续性仍须通过验收。
    align_free_endpoint=False 时，自由段末端不强制对准；由后续独立调姿段对准。
    """
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    tool = _transform(T_flange_tcp, "TCP 偏移")
    flange_start, joints, _ = _snapshot(start_snapshot)
    limits = ROBOT_JOINT_LIMIT_PRESET["piper_x"]
    lower = np.array([limits[f"joint{i}"][0] for i in range(1, 7)])
    upper = np.array([limits[f"joint{i}"][1] for i in range(1, 7)])
    if np.any(joints < lower - 1e-7) or np.any(joints > upper + 1e-7):
        raise MotionError("当前关节超出 PiPER X 模型限位，停止规划并核实反馈与型号。")
    mdh = get_mdh("piper_x")
    fk = arm_fk or (lambda values: fk_from_mdh(mdh, values))

    def matrix_fk(values):
        output = np.asarray(fk(list(values)), dtype=float)
        if output.shape == (4, 4):
            return _transform(output, "FK 输出")
        if output.shape != (6,) or not np.isfinite(output).all():
            raise MotionError("FK 必须返回米/弧度六元组或 4×4 变换。")
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("xyz", output[3:]).as_matrix()
        result[:3, 3] = output[:3]
        return result

    distance, angle = _errors(matrix_fk(joints), flange_start)
    if distance > .0002 or angle > math.radians(.1):
        raise MotionError("实时法兰位姿与所选 FK 不一致，拒绝使用不匹配的运动模型。")
    targets = [_transform(target, "针尖目标") for target in targets_base_tcp]
    if not targets:
        raise MotionError("没有可执行的针尖目标。")
    if (type(free_orientation_segments) is not int
            or not 0 <= free_orientation_segments < len(targets)):
        raise MotionError("途中调姿段数必须为非负整数，并至少保留最后一段严格对准接近。")
    if type(align_free_endpoint) is not bool:
        raise MotionError("自由段末端对准选项必须为布尔值。")
    current = flange_start @ tool
    tool_inverse = np.linalg.inv(tool)
    steps = []
    maximum_position_error = maximum_angle_error = 0.
    maximum_preferred_angle_error = 0.
    position_tolerance, angle_tolerance = .0001, math.radians(.05)
    joint_step_limit = math.radians(12)

    def solve_tcp(tcp, previous_joints, relaxed=False, previous_tcp=None):
        # 机械限位和相邻关节连续性一起约束，失败候选不能成为下一点的起始解。
        local_lower = np.maximum(lower, previous_joints - joint_step_limit)
        local_upper = np.minimum(upper, previous_joints + joint_step_limit)

        def residual(values):
            actual = matrix_fk(values) @ tool
            orientation = Rotation.from_matrix(tcp[:3, :3].T @ actual[:3, :3]).as_rotvec()
            if relaxed:
                previous_orientation = Rotation.from_matrix(
                    previous_tcp[:3, :3].T @ actual[:3, :3]).as_rotvec()
                # 位置为硬验收；朝向与关节变化只用来选择连续、接近偏好的候选解。
                # 最终仍单独检验实际相邻旋转，不能把惩罚项视为已经满足约束。
                return np.r_[(actual[:3, 3] - tcp[:3, 3]) / position_tolerance,
                             orientation / math.radians(5),
                             (values - previous_joints) / joint_step_limit * .05,
                             max(0., np.linalg.norm(previous_orientation) / math.radians(2) - 1.) * 100]
            return np.r_[(actual[:3, 3] - tcp[:3, 3]) / position_tolerance,
                         orientation / angle_tolerance]

        best = None
        accepted = []
        seeds = [previous_joints.copy()]
        for axis in range(6):
            for direction in (-1, 1):
                seed = previous_joints.copy()
                seed[axis] += direction * math.radians(5)
                seeds.append(seed)
        for index, seed in enumerate(seeds):
            seed = np.clip(seed, local_lower + 1e-10, local_upper - 1e-10)
            solved = least_squares(residual, seed, bounds=(local_lower, local_upper), max_nfev=250,
                                   xtol=1e-11, ftol=1e-11, gtol=1e-11)
            position_error, angle_error = _errors(matrix_fk(solved.x) @ tool, tcp)
            if not (math.isfinite(position_error) and math.isfinite(angle_error)):
                raise MotionError("逆解回算产生非有限误差，已拒绝该路径。")
            if relaxed:
                _, step_angle = _errors(matrix_fk(solved.x) @ tool, previous_tcp)
                score = max(position_error / position_tolerance, step_angle / math.radians(2))
            else:
                score = max(position_error / position_tolerance, angle_error / angle_tolerance)
            candidate = (solved.x.copy(), position_error, angle_error,
                         {"停止状态": int(solved.status), "求值次数": int(solved.nfev),
                          "梯度最优性": float(solved.optimality),
                          "局部限位触发": solved.active_mask.astype(int).tolist()})
            if best is None or score < best[0]:
                best = (score, candidate)
            # 优化器停止不代表到位；以实际 TCP 正向回算误差作为验收依据。
            if score <= 1.:
                if index == 0:
                    return candidate
                accepted.append(candidate)
        if accepted:
            return min(accepted, key=lambda item: np.linalg.norm(item[0] - previous_joints))
        return best[1]

    def append_sample(previous_tcp, tcp, segment_index, number, depth=0, orientation_required=True):
        nonlocal joints, maximum_position_error, maximum_angle_error, maximum_preferred_angle_error
        if len(steps) >= 1000:
            raise MotionError("路径超过 1000 个小段，请缩短本次目标范围。")
        solution, position_error, angle_error, solver_info = solve_tcp(tcp, joints)
        actual_flange = matrix_fk(solution)
        actual_tcp = actual_flange @ tool
        step_distance, step_angle = _errors(actual_tcp, previous_tcp)
        hard_accepted = position_error <= position_tolerance and angle_error <= angle_tolerance
        if free_orientation_segments and not orientation_required and (
                not hard_accepted or step_angle > math.radians(2)):
            solution, position_error, angle_error, solver_info = solve_tcp(
                tcp, joints, relaxed=True, previous_tcp=previous_tcp)
            actual_flange = matrix_fk(solution)
            actual_tcp = actual_flange @ tool
            step_distance, step_angle = _errors(actual_tcp, previous_tcp)
        accepted = position_error <= position_tolerance and (
            not orientation_required or angle_error <= angle_tolerance)
        if free_orientation_segments:
            accepted = (accepted and step_distance <= .005 + 1e-10
                        and step_angle <= math.radians(2) + 1e-10
                        and np.max(np.abs(solution - joints)) <= joint_step_limit + 1e-10)
        if not accepted:
            if depth < 3:
                midpoint = np.eye(4)
                midpoint[:3, 3] = (previous_tcp[:3, 3] + tcp[:3, 3]) / 2
                rotation = Slerp([0., 1.], Rotation.from_matrix(
                    np.stack([previous_tcp[:3, :3], tcp[:3, :3]])))
                midpoint[:3, :3] = rotation([.5]).as_matrix()[0]
                reached = append_sample(previous_tcp, midpoint, segment_index, number, depth + 1,
                                        orientation_required=segment_index >= free_orientation_segments)
                return append_sample(reached, tcp, segment_index, number, depth + 1,
                                     orientation_required=orientation_required)
            # 仅在失败时写入复现所需的运动参数；复用应用现有轮转日志。
            diagnostic = {
                "模型": "piper_x", "段": segment_index + 1, "点": number, "细分层数": depth,
                "起点法兰矩阵_m": flange_start.tolist(),
                "起点关节角_rad": np.asarray(start_snapshot["joint_angles"], dtype=float).tolist(),
                "TCP偏移矩阵_m": tool.tolist(),
                "目标路径矩阵_m": [value.tolist() for value in targets],
                "失败目标矩阵_m": tcp.tolist(),
                "上一合格关节角_rad": joints.tolist(),
                "候选关节角_rad": solution.tolist(),
                "距机械下限_deg": np.rad2deg(solution - lower).tolist(),
                "距机械上限_deg": np.rad2deg(upper - solution).tolist(),
                "针尖位置误差_mm": position_error * 1000,
                "姿态误差_deg": math.degrees(angle_error), "求解器": solver_info,
                "姿态约束": "完整姿态" if orientation_required else "途中自由调姿",
                "自由调姿段数": free_orientation_segments,
                "相邻实际旋转_deg": math.degrees(step_angle),
                "相邻实际距离_mm": step_distance * 1000,
            }
            logging.getLogger("TCP标定").warning(
                "路径逆解诊断：%s", json.dumps(diagnostic, ensure_ascii=False, allow_nan=False))
            orientation_message = (f"姿态误差 {math.degrees(angle_error):.3f}°（须≤0.05°）"
                                   if orientation_required else
                                   f"朝向偏好偏差 {math.degrees(angle_error):.3f}°（不作硬约束）")
            continuity_message = (f"，相邻实际旋转 {math.degrees(step_angle):.3f}°（须≤2°）"
                                  if free_orientation_segments else "")
            raise MotionError(f"第 {segment_index + 1} 段第 {number} 点附近重试、细分后仍未找到合格逆解："
                              f"针尖位置误差 {position_error * 1000:.3f} mm（须≤0.1 mm），"
                              f"{orientation_message}{continuity_message}。")
        joints = solution
        maximum_position_error = max(maximum_position_error, position_error)
        if orientation_required:
            maximum_angle_error = max(maximum_angle_error, angle_error)
        else:
            maximum_preferred_angle_error = max(maximum_preferred_angle_error, angle_error)
        commanded_tcp = actual_tcp if free_orientation_segments else tcp
        commanded_flange = actual_flange if free_orientation_segments else tcp @ tool_inverse
        steps.append({"T_base_flange": commanded_flange.tolist(), "T_base_tcp": commanded_tcp.tolist(),
                      "joints_rad": joints.tolist(), "segment_index": segment_index,
                      "ik_position_error_mm": position_error * 1000,
                      "ik_angle_error_deg": math.degrees(angle_error) if orientation_required else None,
                      "orientation_constraint": "required" if orientation_required else "preferred",
                      "T_base_tcp_requested": tcp.tolist(),
                      "orientation_preference_error_deg": math.degrees(angle_error)})
        return commanded_tcp

    for segment_index, target in enumerate(targets):
        distance, angle = _errors(current, target)
        count = max(1, int(math.ceil(distance / .005)), int(math.ceil(angle / math.radians(2))))
        if len(steps) + count > 1000:
            raise MotionError("路径超过 1000 个小段，请缩短本次目标范围。")
        interpolator = Slerp([0., 1.], Rotation.from_matrix(np.stack([current[:3, :3], target[:3, :3]])))
        previous_tcp = current
        for number in range(1, count + 1):
            fraction = number / count
            tcp = np.eye(4)
            tcp[:3, :3] = interpolator([fraction]).as_matrix()[0]
            tcp[:3, 3] = current[:3, 3] + fraction * (target[:3, 3] - current[:3, 3])
            required = (segment_index >= free_orientation_segments
                        or (align_free_endpoint and segment_index == free_orientation_segments - 1
                            and number == count))
            previous_tcp = append_sample(previous_tcp, tcp, segment_index, number,
                                         orientation_required=required)
        current = previous_tcp if free_orientation_segments else target
    return {"steps": steps, "start_snapshot": copy.deepcopy(start_snapshot), "T_flange_tcp": tool.tolist(),
            "metadata": {"model": "piper_x", "channel": "can1", "step_count": len(steps),
                         "translation_step_mm": 5., "rotation_step_deg": 2.,
                         "ik_position_frame": "tcp", "ik_position_tolerance_mm": .1,
                         "ik_angle_tolerance_deg": .05,
                         "ik_max_position_error_mm": maximum_position_error * 1000,
                         "ik_max_angle_error_deg": math.degrees(maximum_angle_error),
                         "free_orientation_segments": free_orientation_segments,
                         "align_free_endpoint": align_free_endpoint,
                         "ik_angle_tolerance_scope": "仅完整姿态约束点；自由调姿点仅验收位置与连续性",
                         "orientation_preference_max_error_deg": math.degrees(maximum_preferred_angle_error),
                         "command_pose_source": "accepted_joint_fk" if free_orientation_segments else "requested_pose",
                         "collision_checked": False,
                         "scope": "仅检查采样位姿可达与关节连续性；未检查碰撞、接触或实际定位精度。"}}


class MotionExecutor:
    """将预检通过的同向直线样点合并执行；回调只传事件，不操作 UI。"""

    def __init__(self, reader, get_snapshot, on_event):
        self.reader, self.get_snapshot, self.on_event = reader, get_snapshot, on_event
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._worker = None
        self.locked = False
        self._stop_error = None
        self._direction_proof = None
        self._guarded_context = None
        self._direction_epoch = 0

    @property
    def running(self):
        return self._worker is not None and self._worker.is_alive()

    def _event(self, kind, **data):
        if kind == "done" and self._guarded_context is not None:
            data.update(self._finish_guarded_motion(data))
        elif kind in ("error", "stopped", "paused"):
            self.invalidate_direction_check()
        recorder = getattr(self.reader, "diagnostic_event", None)
        if callable(recorder):
            recorder("运动事件：" + kind, data, trigger=kind in ("error", "stopped"))
        try:
            self.on_event(kind, data)
        except Exception:
            # UI 回调不可使已启动的执行线程无声退出。
            if kind not in ("error", "stopped"):
                self.invalidate_direction_check()
                raise MotionError("运动事件无法送达界面，已终止执行。")

    def invalidate_direction_check(self):
        """目标、控制状态或连接改变时废弃本次方向验证，不发送设备指令。"""
        with self._guard:
            self._direction_proof = None
            self._direction_epoch += 1

    def direction_check_valid(self):
        """只读当前验证凭证，供界面控制观察位按钮；不向机械臂发送指令。"""
        if MOTION_EXECUTION_HOLD_REASON:
            return False
        proof = self._direction_proof
        if (proof is None or self.running or self.locked or self._stop.is_set()
                or proof["epoch"] != self._direction_epoch
                or not 0 <= time.monotonic() - proof["verified_at"] <= 300):
            return False
        try:
            pose, joints, _, state = self._guarded_live()
            if (proof["connection_id"] != state["connection_id"]
                    or proof["send_attempt_count"] != state["send_attempt_count"]):
                return False
            self._check_start(pose, joints, proof["pose"], proof["joints"], "方向验证后")
            if _errors(pose @ proof["tool"], proof["pose"] @ proof["tool"])[0] > .0002:
                return False
        except Exception:
            return False
        return proof is self._direction_proof and proof["epoch"] == self._direction_epoch

    @staticmethod
    def _unit_normal(value):
        normal = np.asarray(value, dtype=float)
        if (normal.shape != (3,) or not np.isfinite(normal).all()
                or abs(float(np.linalg.norm(normal)) - 1.) > 1e-6):
            raise MotionError("板外法线须为有限的三维单位向量。")
        return normal.copy()

    def _require_idle_before_target(self, status, pose, joints, stamp, *,
                                    scope=None, expected_state=None):
        """所有目标发送前均须确认到位，原位静止不构成例外。"""
        if status["motion_status"] != 0:
            raise MotionError("控制器未到位，静止不能替代到位确认；拒绝发送下一目标。")

    def _guarded_live(self, *, scope=None):
        pose, joints, stamp, status = self._live()
        state = self._send_state()
        if (state["connection_id"] is None or state["command_pending"]
                or self.locked or self._stop.is_set()):
            raise MotionError("运动仍被锁定、连接已变更或存在在途指令，请重新准备运动控制。")
        if (status["ctrl_mode"] != 1 or status.get("mode_feedback") != 2
                or len(status["enabled"]) != 6
                or not all(status["enabled"])):
            raise MotionError("请先准备运动控制，确认 CAN 直线模式且六关节全部使能。")
        self._require_idle_before_target(status, pose, joints, stamp, scope=scope, expected_state=state)
        return pose, joints, stamp, state

    @staticmethod
    def _validate_guarded_joints(plan):
        from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET
        limits = ROBOT_JOINT_LIMIT_PRESET["piper_x"]
        lower = np.asarray([limits[f"joint{i}"][0] for i in range(1, 7)])
        upper = np.asarray([limits[f"joint{i}"][1] for i in range(1, 7)])
        _, previous, _ = _snapshot(plan["start_snapshot"])
        for values in [previous, *(step.get("joints_rad") for step in plan["steps"])]:
            joints = np.asarray(values, dtype=float)
            if (joints.shape != (6,) or not np.isfinite(joints).all()
                    or np.any(joints < lower - 1e-7) or np.any(joints > upper + 1e-7)
                    or np.max(np.abs(joints - previous)) > math.radians(12) + 1e-8):
                raise MotionError("受控验证路径的关节限位或相邻关节变化未通过检查。")
            previous = joints

    @staticmethod
    def _check_board_clearance(plan, start_flange, point, normal, minimum_gap):
        """核对法兰 MoveL/Slerp 参考下的针尖板距；不检查实体碰撞或实机精度。"""
        tool = _transform(plan["T_flange_tcp"])
        previous = start_flange
        radius = float(np.linalg.norm(tool[:3, 3]))
        for step in plan["steps"]:
            target = _transform(step["T_base_flange"])
            distance, angle = _errors(previous, target)
            count = max(2, int(math.ceil(angle / math.radians(.5))),
                        int(math.ceil(distance / .002)))
            # 固定旋转轴的二阶导数不超过 radius * angle²。
            # 端点线性弦与真实参考曲线间的最大板距偏差据此保守留量。
            gap_margin = radius * (angle / count) ** 2 / 8.
            fractions = np.linspace(0., 1., count + 1)
            rotations = Slerp([0., 1.], Rotation.from_matrix(
                np.stack([previous[:3, :3], target[:3, :3]])))(fractions).as_matrix()
            positions = (previous[:3, 3][None, :]
                         + fractions[:, None] * (target[:3, 3] - previous[:3, 3])[None, :]
                         + rotations @ tool[:3, 3])
            if float(np.min((positions - point) @ normal)) - gap_margin < minimum_gap - 1e-9:
                raise MotionError(f"路径的法兰直线插补参考会使针尖距板面小于 {minimum_gap*1000:g} mm，请调整起点或路径后重新预览。")
            previous = target

    @classmethod
    def _board_noncontact_spec(cls, plan):
        rule = plan.get("metadata", {}).get("board_noncontact")
        if rule is None:
            return None
        point = np.asarray(rule["point_base_m"], dtype=float)
        normal = cls._unit_normal(rule["normal_base"])
        minimum = float(rule["minimum_gap_m"])
        if (point.shape != (3,) or not np.isfinite(point).all()
                or not math.isfinite(minimum) or minimum < BOARD_MIN_GAP_M):
            raise MotionError("标定板非接触约束无效；最小间距须至少为 5 mm。")
        return point, normal, minimum

    @classmethod
    def check_board_noncontact(cls, plan, start_flange=None):
        """仅对带标定板非接触约束的计划检查整条插补参考，不影响插孔计划。"""
        rule = cls._board_noncontact_spec(plan)
        if rule is not None:
            start = _transform(plan["start_snapshot"]["T_base_flange"]
                               if start_flange is None else start_flange)
            cls._check_board_clearance(plan, start, *rule)

    @classmethod
    def _check_board_noncontact_feedback(cls, plan, flange):
        rule = cls._board_noncontact_spec(plan)
        if rule is not None:
            point, normal, minimum = rule
            tcp = flange @ _transform(plan["T_flange_tcp"])
            gap = float((tcp[:3, 3] - point) @ normal)
            if gap < minimum - 1e-9:
                raise MotionError(f"反馈针尖距板面 {gap*1000:.3f} mm，小于非接触下限 {minimum*1000:g} mm，停止运动。")

    def _guarded_preflight(self, plan, scope, normal, point, operator_confirmed):
        tool = _transform(plan["T_flange_tcp"])
        planned_pose, planned_joints, _ = _snapshot(plan["start_snapshot"])
        pose, joints, _, send_state = self._guarded_live(scope=scope)
        self._check_start(pose, joints, planned_pose, planned_joints, "受控验证")
        start_tcp = pose @ tool
        if _errors(start_tcp, planned_pose @ tool)[0] > .0002:
            raise MotionError("针尖已偏离预览起点超过 0.2 mm，请重新预览。")
        self._validate_guarded_joints(plan)
        steps = plan["steps"]
        if scope == "direction_check":
            if len(steps) != 1:
                raise MotionError("方向试步仅允许一个 0.5 mm 目标，不能连续执行多段。")
            end = _transform(steps[0]["T_base_tcp"])
            planned_tcp = planned_pose @ tool
            if (np.linalg.norm(end[:3, 3] - planned_tcp[:3, 3] - normal * .0005) > .00002
                    or _errors(end, planned_tcp)[1] > math.radians(.01)):
                raise MotionError("方向试步必须保持针尖朝向，并沿板外法线退开 0.5 mm。")
            # 不允许预览后的起点漂移吞掉这次仅 0.5 mm 的方向试步。
            if np.linalg.norm(end[:3, 3] - start_tcp[:3, 3] - normal * .0005) > .00002:
                raise MotionError("试步起点已有变化，请以实时反馈重新生成 0.5 mm 试步。")
            self._direction_proof = None
        else:
            proof = self._direction_proof
            if operator_confirmed is not True:
                raise MotionError("请先人工确认刚才的 0.5 mm 试步确实远离板面。")
            if (proof is None or proof["epoch"] != self._direction_epoch
                    or not 0 <= time.monotonic() - proof["verified_at"] <= 300
                    or proof["connection_id"] != send_state["connection_id"]
                    or proof["send_attempt_count"] != send_state["send_attempt_count"]
                    or not np.allclose(proof["normal"], normal, atol=1e-6, rtol=0)
                    or not np.allclose(proof["tool"], tool, atol=1e-9, rtol=0)):
                raise MotionError("方向验证已失效，请重新试退开 0.5 mm 并人工确认方向。")
            self._check_start(pose, joints, proof["pose"], proof["joints"], "方向验证后")
            if _errors(start_tcp, proof["pose"] @ tool)[0] > .0002:
                raise MotionError("方向验证后针尖已移动超过 0.2 mm，请重新试步。")
            start_gap = float(np.dot(start_tcp[:3, 3] - point, normal))
            if start_gap < .01:
                raise MotionError("当前针尖距板面不足 10 mm，不能启动观察位验证。")
            minimum_gap = min(start_gap, .03) - .0001
            for step in steps:
                tcp = _transform(step["T_base_tcp"])
                if float(np.dot(tcp[:3, 3] - point, normal)) < minimum_gap:
                    raise MotionError("观察位路径有采样点过于靠近板面，请调整起点后重新预览。")
            self._check_board_clearance(plan, pose, point, normal, minimum_gap)
            end = _transform(steps[-1]["T_base_tcp"])
            offset = end[:3, 3] - point
            end_gap = float(np.dot(offset, normal))
            if (end_gap < .03 - 1e-9 or np.linalg.norm(offset - normal * end_gap) > .0001
                    or math.acos(float(np.clip(np.dot(end[:3, 2], -normal), -1., 1.)))
                    > math.radians(.05)):
                raise MotionError("仅开放板前至少 30 mm、针轴垂直板面并指向目标的观察位。")
            self._direction_proof = None  # 一次试步凭证只启动一次观察位路径。
        return {"scope": scope, "normal": normal.copy(), "tool": tool,
                "start_tcp": start_tcp, "connection_id": send_state["connection_id"],
                "epoch": self._direction_epoch, "count": len(steps)}

    def _finish_guarded_motion(self, data):
        context = self._guarded_context
        self._guarded_context = None
        if (context["epoch"] != self._direction_epoch or self._stop.is_set()
                or data.get("index") != context["count"] or data.get("count") != context["count"]):
            raise MotionError("受控验证已中止或路径未完整完成，不能确认验证通过。")
        pose, joints, _, send_state = self._guarded_live()
        if send_state["connection_id"] != context["connection_id"]:
            raise MotionError("验证期间机械臂连接已改变，请重新准备运动控制。")
        if context["scope"] != "direction_check":
            return {"guarded_scope": context["scope"]}
        actual = pose @ context["tool"]
        offset = actual[:3, 3] - context["start_tcp"][:3, 3]
        outward = float(np.dot(offset, context["normal"]))
        lateral = float(np.linalg.norm(offset - outward * context["normal"]))
        angle = _errors(actual, context["start_tcp"])[1]
        if (not .0003 <= outward <= .0007 or lateral > .0001
                or angle > math.radians(.15)):
            raise MotionError("实际试步未通过板外方向、位移或保姿检查，观察位执行仍锁定。")
        with self._guard:
            if context["epoch"] != self._direction_epoch or self._stop.is_set():
                raise MotionError("方向验证已被停止或作废，请重新验证。")
            self._direction_proof = {"epoch": self._direction_epoch,
                "verified_at": time.monotonic(), "pose": pose.copy(), "joints": joints.copy(),
                "normal": context["normal"].copy(), "tool": context["tool"].copy(),
                "connection_id": send_state["connection_id"],
                "send_attempt_count": send_state["send_attempt_count"]}
        return {"guarded_scope": "direction_check", "direction_verified": True,
                "outward_mm": outward * 1000, "lateral_mm": lateral * 1000,
                "orientation_change_deg": math.degrees(angle)}

    def start(self, plan, speed_percent=5, tolerance_mm=.1, tolerance_deg=.05, *,
              live_guard=None, allow_enable=True):
        if MOTION_EXECUTION_HOLD_REASON:
            logging.getLogger("TCP标定").warning(MOTION_EXECUTION_HOLD_REASON)
            raise MotionError(MOTION_EXECUTION_HOLD_REASON)
        self._start_plan(plan, speed_percent, tolerance_mm, tolerance_deg,
                         live_guard=live_guard, allow_enable=allow_enable)

    def start_direction_check(self, plan, outward_normal):
        """操作者单独点击后仅试退开 0.5 mm；不会自动接续观察路径。"""
        normal = self._unit_normal(outward_normal)
        self._start_plan(plan, 1, .1, .05, scope="direction_check", normal=normal)

    def start_observation(self, plan, point_base, normal, *, operator_confirmed=False):
        """凭本连接刚完成的方向验证，仅开放板前至少 30 mm 的观察位。"""
        normal = self._unit_normal(normal)
        point = np.asarray(point_base, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise MotionError("板面目标点须为有限的三维坐标。")
        self._start_plan(plan, 1, .1, .05, scope="observation", normal=normal,
                         point=point.copy(), operator_confirmed=operator_confirmed)

    def _start_plan(self, plan, speed_percent, tolerance_mm, tolerance_deg, *, scope=None,
                    normal=None, point=None, operator_confirmed=False,
                    live_guard=None, allow_enable=True):
        if MOTION_EXECUTION_HOLD_REASON:
            raise MotionError(MOTION_EXECUTION_HOLD_REASON)
        if live_guard is not None and not callable(live_guard):
            raise MotionError("实时运动检查必须可调用。")
        if live_guard is not None:
            live_guard()
        if type(speed_percent) is not int or not 1 <= speed_percent <= 10:
            raise MotionError("速度比例须为 1～10 的整数。")
        if (isinstance(tolerance_mm, bool) or isinstance(tolerance_deg, bool)
                or not math.isfinite(float(tolerance_mm)) or not .1 <= float(tolerance_mm) <= 5.
                or not math.isfinite(float(tolerance_deg)) or not .01 <= float(tolerance_deg) <= 2.):
            raise MotionError("到达门槛须为 0.1～5 mm、0.01～2°；末端默认使用 0.1 mm / 0.05°。")
        with self._guard:
            if self.locked or self.running:
                raise MotionError("运动正在执行或已锁定；请等待结束，故障后须人工处理并重新连接。")
            if not isinstance(plan, dict) or not plan.get("steps"):
                raise MotionError("没有已检查的运动步骤。")
            if plan.get("metadata", {}).get("channel") != "can1":
                raise MotionError("只能执行右臂 can1 的计划。")
            frozen = copy.deepcopy(plan)
            tool = _transform(frozen["T_flange_tcp"])
            for step in frozen["steps"]:
                flange = _transform(step["T_base_flange"])
                tcp = _transform(step["T_base_tcp"])
                if max(_errors(flange @ tool, tcp)) > 1e-6:
                    raise MotionError("运动计划的针尖与法兰变换不一致。")
            self.check_board_noncontact(frozen)
            if scope is not None:
                context = self._guarded_preflight(frozen, scope, normal, point, operator_confirmed)
                frozen["_guarded_motion_scope"] = scope
            else:
                context = None
            if scope is not None and (self._stop.is_set() or self.locked):
                raise MotionError("受控验证启动前已收到停止请求，未发送目标。")
            self._guarded_context = context
            if scope is None:
                self._stop.clear()
            self._pause.clear()
            self._stop_error = None
            self._worker = threading.Thread(target=self._run,
                                            args=(frozen, speed_percent, float(tolerance_mm), float(tolerance_deg)),
                                            kwargs={"allow_enable": scope is None and allow_enable,
                                                    "live_guard": live_guard},
                                            name="TCP右臂分段运动", daemon=True)
            self._worker.start()

    def pause_after_step(self):
        """当前已发小段到达后暂停，不以急停代替正常阶段切换。"""
        self._pause.set()

    def stop(self):
        """同步锁定并发送电子急停，返回发送结果；调用方随后才可断开 CAN。"""
        self._stop.set()
        self.locked = True
        error = None
        try:
            self.reader.stop_motion()
        except Exception as exc:
            error = str(exc)
        self._stop_error = error
        if not self.running:
            self._event("stopped", message=self._stop_error or "已请求电子急停，运动已锁定。")
        return {"requested": True, "sent": error is None, "error": error}

    def _live(self):
        # 控制线程直接读取接入层原子快照，不受 UI 刷新节拍限制。
        # 有此接口但读取失败时必须终止，不能退回旧的界面缓存。
        read_pose = getattr(self.reader, "read_pose", None)
        snapshot = read_pose() if callable(read_pose) else self.get_snapshot()
        pose, joints, stamp = _snapshot(snapshot, fresh=True)
        status = self.reader.motion_status()
        if status.get("locked") or status["arm_status"] != 0 or status["err_code"] != 0:
            raise MotionError(f"控制器异常或锁定：状态 {status['arm_status']}，故障码 {status['err_code']}。")
        if status["ctrl_mode"] not in (0, 1):
            raise MotionError("控制器已退出 CAN/待机模式。")
        if status.get("teach_status") not in (0, 2, 6):
            raise MotionError("控制器仍在拖动示教或轨迹执行状态，请先退出并确认反馈后再运动。")
        return pose, joints, stamp, status

    def _send_state(self):
        getter = getattr(self.reader, "motion_send_state", None)
        if not callable(getter):
            raise MotionError("机械臂接入缺少发送状态确认，拒绝执行。")
        state = getter()
        if (not isinstance(state, dict) or not isinstance(state.get("send_attempt_count"), int)
                or not isinstance(state.get("command_pending"), bool)
                or "connection_id" not in state):
            raise MotionError("机械臂发送状态不完整，拒绝执行。")
        return state

    @staticmethod
    def _check_start(pose, joints, reference_pose, reference_joints, stage, tool=None):
        distance, angle = _errors(pose, reference_pose)
        if tool is not None:
            distance = max(distance, _errors(pose @ tool, reference_pose @ tool)[0])
        if (distance > .0002 or angle > math.radians(.1)
                or np.max(np.abs(joints - reference_joints)) > math.radians(.1)):
            raise MotionError(f"{stage}起点变化超过 0.2 mm / 0.1° 或关节 0.1°，未发送下一目标，请重新预览。")

    def _fresh_live_after(self, stamp, live_guard=None):
        deadline = time.monotonic() + .5
        while not self._stop.is_set():
            if live_guard is not None:
                live_guard()
            live = self._live()
            if live[2] > stamp:
                return live
            if time.monotonic() >= deadline:
                raise MotionError("使能检查后 0.5 秒内未收到新鲜位姿反馈，未发送运动目标。")
            self._stop.wait(.02)
        return None

    @staticmethod
    def _check_flange_corridor(start, target, actual, lateral_tolerance=.0005):
        """按 MoveL 法兰直线及球面插值拦截异常；阈值不是实机精度承诺。"""
        delta = target[:3, 3] - start[:3, 3]
        length = float(np.linalg.norm(delta))
        offset = actual[:3, 3] - start[:3, 3]
        if length > 1e-8:
            progress = float(np.dot(offset, delta) / length)
            lateral = float(np.linalg.norm(offset - progress * delta / length))
            if progress < -.0002:
                raise MotionError("实际法兰沿本段反方向移动超过 0.2 mm，已偏离预览方向。")
            if lateral > lateral_tolerance:
                raise MotionError(f"实际法兰偏出本段直线超过 {lateral_tolerance * 1000:.1f} mm。")
            if progress > length + .0005:
                raise MotionError("实际法兰越过本段目标超过 0.5 mm。")
            fraction = float(np.clip(progress / length, 0., 1.))
        else:
            if np.linalg.norm(offset) > lateral_tolerance:
                raise MotionError(f"原地调姿时实际法兰平移超过 {lateral_tolerance * 1000:.1f} mm。")
            expected_rotation = Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).as_rotvec()
            actual_rotation = Rotation.from_matrix(start[:3, :3].T @ actual[:3, :3]).as_rotvec()
            norm_squared = float(np.dot(expected_rotation, expected_rotation))
            fraction = (float(np.clip(np.dot(actual_rotation, expected_rotation) / norm_squared, 0., 1.))
                        if norm_squared > 1e-12 else 0.)
        expected = Slerp([0., 1.], Rotation.from_matrix(
            np.stack([start[:3, :3], target[:3, :3]])))([fraction]).as_matrix()[0]
        if Rotation.from_matrix(expected.T @ actual[:3, :3]).magnitude() > math.radians(1):
            raise MotionError("实际法兰朝向偏离本段插值参考超过 1°。")

    @staticmethod
    def _execution_steps(plan):
        """保留完整预检样点，仅压缩严格同姿态、同向共线段的发送端点。

        返回的序号是原 steps 中的末样点序号，供现有进度、阶段判断使用。
        不跨段、转角、调姿或自由姿态段；每条合并指令最多 50 mm / 12° 关节跨度。
        """
        steps = plan["steps"]
        if plan.get("_guarded_motion_scope") == "direction_check":
            return list(enumerate(steps, 1))
        start_flange, start_joints, _ = _snapshot(plan["start_snapshot"])
        tool = _transform(plan["T_flange_tcp"])
        poses = [(_transform(step["T_base_flange"]), _transform(step["T_base_tcp"]))
                 for step in steps]
        joints = [np.asarray(step["joints_rad"], dtype=float) for step in steps]
        if any(values.shape != (6,) or not np.isfinite(values).all() for values in joints):
            raise MotionError("运动计划缺少完整且有限的预检关节解。")
        free_segments = plan.get("metadata", {}).get("free_orientation_segments", 0)
        commands = []
        first = 0
        while first < len(steps):
            origin = poses[first - 1] if first else (start_flange, start_flange @ tool)
            origin_joints = joints[first - 1] if first else start_joints
            segment = steps[first]["segment_index"]
            directions = [pose[:3, 3] - base[:3, 3] for base, pose in zip(origin, poses[first])]
            lengths = [float(np.linalg.norm(direction)) for direction in directions]
            joint_min = np.minimum(origin_joints, joints[first])
            joint_max = np.maximum(origin_joints, joints[first])

            def same_line(index):
                if (steps[index]["segment_index"] != segment or segment < free_segments
                        or steps[index].get("orientation_constraint", "required") != "required"):
                    return False
                for base, pose, direction, length in zip(origin, poses[index], directions, lengths):
                    if (length <= 1e-10 or not np.allclose(base[:3, :3], pose[:3, :3],
                                                        atol=1e-10, rtol=0)):
                        return False
                    offset = pose[:3, 3] - base[:3, 3]
                    progress = float(np.dot(offset, direction) / length)
                    if (progress < -1e-10 or progress > .05 + 1e-10
                            or np.linalg.norm(offset - progress * direction / length) > 1e-9):
                        return False
                return True

            last = first
            if same_line(first) and np.max(joint_max - joint_min) <= math.radians(12) + 1e-10:
                while last + 1 < len(steps) and same_line(last + 1):
                    candidate = last + 1
                    if any(np.dot(poses[candidate][axis][:3, 3] - poses[last][axis][:3, 3], direction) / length
                           < -1e-10 for axis, (direction, length) in enumerate(zip(directions, lengths))):
                        break
                    next_min = np.minimum(joint_min, joints[candidate])
                    next_max = np.maximum(joint_max, joints[candidate])
                    if np.max(next_max - next_min) > math.radians(12) + 1e-10:
                        break
                    joint_min, joint_max = next_min, next_max
                    last = candidate
            commands.append((last + 1, steps[last]))
            first = last + 1
        return commands

    def _run(self, plan, speed, tolerance_mm, tolerance_deg, *, allow_enable=True,
             live_guard=None):
        last_feedback = None
        target_flange = None
        guarded = plan.get("_guarded_motion_scope") in ("direction_check", "observation")
        if guarded:
            allow_enable = False
        lateral_tolerance = .0002 if plan.get("_guarded_motion_scope") == "direction_check" else .0005

        def check_control_mode(status):
            if guarded and (status["ctrl_mode"] != 1 or status.get("mode_feedback") != 2):
                raise MotionError("受控执行中控制器已离开 CAN / MoveL 模式，拒绝继续发送目标。")

        def read_live():
            nonlocal last_feedback
            if live_guard is not None:
                live_guard()
            live = self._live()
            last_feedback = {"T_base_flange": live[0], "joints_rad": live[1],
                             "monotonic": live[2], "status": live[3]}
            check_control_mode(live[3])
            self._check_board_noncontact_feedback(plan, live[0])
            return live

        completed = 0
        motion_requested = False
        send_baseline = None
        last_arrival = None
        logger = logging.getLogger("TCP标定")
        try:
            count = len(plan["steps"])
            commands = self._execution_steps(plan)
            command_count = len(commands)
            plan["metadata"].update(execution_command_count=command_count,
                                    merged_sample_count=count - command_count,
                                    linear_command_max_mm=50., joint_span_max_deg=12.,
                                    feedback_poll_ms=20)
            send_baseline = self._send_state()
            if send_baseline["connection_id"] is None:
                raise MotionError("机械臂连接已断开，拒绝执行。")
            if send_baseline["command_pending"]:
                raise MotionError("机械臂仍有未确认完成的控制指令，拒绝开始新路径。")
            start_pose, start_joints, start_stamp, start_status = read_live()
            self._require_idle_before_target(start_status, start_pose, start_joints, start_stamp,
                                             scope=plan.get("_guarded_motion_scope"),
                                             expected_state=send_baseline)
            planned_pose, planned_joints, _ = _snapshot(plan["start_snapshot"])
            tool = _transform(plan["T_flange_tcp"])
            self._check_start(start_pose, start_joints, planned_pose, planned_joints, "执行前", tool)
            logger.info("执行前核对通过：使能=%s，示教状态=%s，起点法兰_m=%s，关节_rad=%s",
                        start_status["enabled"], start_status["teach_status"],
                        start_pose[:3, 3].tolist(), start_joints.tolist())
            if self._stop.is_set():
                return
            if live_guard is not None:
                live_guard()
            if allow_enable:
                self.reader.enable_for_motion()
            elif not all(start_status["enabled"]):
                raise MotionError("受控执行要求机械臂已全部使能；本入口不会自动使能。")
            if self._stop.is_set():
                return
            if live_guard is not None:
                live_guard()
            enable_checked_at = time.monotonic()
            enabled_live = self._fresh_live_after(enable_checked_at, live_guard)
            if enabled_live is None:
                return
            enabled_pose, enabled_joints, _, enabled_status = enabled_live
            check_control_mode(enabled_status)
            self._check_board_noncontact_feedback(plan, enabled_pose)
            self._check_start(enabled_pose, enabled_joints, planned_pose, planned_joints, "使能后", tool)
            self._check_start(enabled_pose, enabled_joints, start_pose, start_joints, "使能前后", tool)
            if not all(enabled_status["enabled"]):
                raise MotionError("使能检查后关节未全部使能，未发送运动目标。")
            logger.info("使能后新鲜反馈与预览起点一致：发送状态=%s", self._send_state())
            self._event("started", count=count, speed_percent=speed, metadata=plan["metadata"],
                        command_count=command_count,
                        tolerance_mm=tolerance_mm, tolerance_deg=tolerance_deg)
            for command_index, (index, step) in enumerate(commands, 1):
                if self._stop.is_set():
                    break
                if self._pause.is_set():
                    self._event("paused", index=completed, count=count)
                    return
                before_pose, before_joints, previous_stamp, status = read_live()
                if not all(status["enabled"]):
                    raise MotionError("关节使能丢失，停止执行。")
                self._require_idle_before_target(
                    status, before_pose, before_joints, previous_stamp,
                    scope=plan.get("_guarded_motion_scope") if command_index == 1 else None,
                    expected_state=send_baseline)
                if command_index == 1:
                    self._check_start(before_pose, before_joints, planned_pose, planned_joints, "首目标发送前", tool)
                    self._check_start(before_pose, before_joints, enabled_pose, enabled_joints, "使能检查后", tool)
                elif last_arrival is not None:
                    self._check_start(before_pose, before_joints, *last_arrival, "下一小段发送前", tool=tool)
                target_flange = _transform(step["T_base_flange"])
                self.check_board_noncontact({**plan, "steps": [step]}, before_pose)
                if command_index == 1:
                    logger.info("首目标准备发送：法兰起点_m=%s，法兰目标_m=%s，针尖目标_m=%s，速度比例=%s",
                                before_pose[:3, 3].tolist(), target_flange[:3, 3].tolist(),
                                np.asarray(step["T_base_tcp"])[:3, 3].tolist(), speed)
                    recorder = getattr(self.reader, "diagnostic_event", None)
                    if callable(recorder):
                        recorder("首目标上下文", {"start_flange": before_pose, "start_joints_rad": before_joints,
                                                    "target_flange": step["T_base_flange"],
                                                    "target_tcp": step["T_base_tcp"], "T_flange_tcp": tool,
                                                    "status": status, "metadata": plan["metadata"],
                                                    "speed_percent": speed, "send_state": self._send_state()})
                if live_guard is not None:
                    live_guard()
                if self._stop.is_set():
                    break
                self.reader.move_flange_linear(step["T_base_flange"], speed)
                command_completed_at = time.monotonic()
                if command_index == 1:
                    logger.info("首目标发送调用完成：发送状态=%s", self._send_state())
                target = _transform(step["T_base_tcp"])
                deadline = time.monotonic() + 15.
                arrived = 0
                first_feedback_logged = False
                while not self._stop.wait(.02):
                    actual, actual_joints, stamp, status = read_live()
                    if not all(status["enabled"]):
                        raise MotionError("运动过程中关节失能。")
                    if time.monotonic() > deadline:
                        raise MotionError(f"第 {index} 小段 15 秒内未到达，已请求停止。")
                    if np.max(np.abs(actual_joints - before_joints)) > math.radians(20):
                        raise MotionError("小段执行中的关节变化超过 20°，可能偏离预检解支。")
                    if stamp <= previous_stamp:
                        continue
                    previous_stamp = stamp
                    if command_index == 1 and not first_feedback_logged and stamp > command_completed_at:
                        logger.info("首目标后首份新鲜反馈：法兰位移_mm=%s，针尖位移_mm=%s，关节变化_deg=%s",
                                    ((actual[:3, 3] - before_pose[:3, 3]) * 1000).tolist(),
                                    (((actual @ tool)[:3, 3] - (before_pose @ tool)[:3, 3]) * 1000).tolist(),
                                    np.rad2deg(actual_joints - before_joints).tolist())
                        first_feedback_logged = True
                    self._check_flange_corridor(before_pose, target_flange, actual, lateral_tolerance)
                    distance, angle = _errors(actual @ tool, target)
                    arrived = arrived + 1 if distance <= tolerance_mm / 1000. and angle <= math.radians(tolerance_deg) else 0
                    if arrived >= 3 and status["motion_status"] == 0:
                        if np.max(np.abs(actual_joints - np.asarray(step["joints_rad"]))) > math.radians(3):
                            raise MotionError("实际关节与预检解相差超过 3°，停止后续路径。")
                        if callable(getattr(self.reader, "mark_motion_arrived", None)):
                            self.reader.mark_motion_arrived()
                        last_arrival = (actual.copy(), actual_joints.copy())
                        break
                if self._stop.is_set():
                    break
                completed = index
                self._event("step", index=index, count=count, segment_index=step["segment_index"],
                            T_base_tcp=step["T_base_tcp"], command_index=command_index,
                            command_count=command_count)
                if self._pause.is_set():
                    self._event("paused", index=index, count=count)
                    return
            if not self._stop.is_set():
                logger.info("路径反馈到位：已完成小段=%s，总小段=%s", completed, count)
                self._event("done", index=completed, count=count)
        except Exception as exc:
            failure_traceback = traceback.format_exc()
            try:
                send_state = self._send_state() if send_baseline is not None else None
            except Exception as state_exc:
                send_state = None
                self.locked = True
                self._stop_error = f"无法确认指令发送阶段：{state_exc}；请检查控制器实际状态。"
            motion_requested = bool(send_baseline is not None and send_state is not None
                                    and send_state["connection_id"] == send_baseline["connection_id"]
                                    and send_state["send_attempt_count"] > send_baseline["send_attempt_count"])
            pending = motion_requested and send_state["command_pending"]
            logger.exception("运动执行失败：本次已尝试真实发送=%s，指令待完成=%s，已完成小段=%d，发送状态=%s",
                             motion_requested, pending, completed, send_state)
            if pending and not self._stop.is_set():
                self.locked = True
                try:
                    self.reader.stop_motion()
                except Exception as stop_exc:
                    self._stop_error = str(stop_exc)
            self._event("error", message=str(exc), stop_error=self._stop_error, index=completed,
                        motion_requested=motion_requested, send_state=send_state,
                        start_snapshot=plan.get("start_snapshot"), target_flange=target_flange,
                        T_flange_tcp=plan.get("T_flange_tcp"), last_feedback=last_feedback,
                        traceback=failure_traceback)
        finally:
            if self._stop.is_set():
                self._event("stopped", message=self._stop_error or "已请求电子急停，运动已锁定。", index=completed)
