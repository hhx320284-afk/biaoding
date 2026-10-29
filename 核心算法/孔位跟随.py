"""孔外跟随：原始视觉守卫、自适应平滑及稳定目标的较长直线接近。"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from 探针运动 import MotionError, MotionExecutor, _errors, _snapshot, _transform, plan_tcp_path
from 目标变换 import travel_targets


class HoleFollower:
    """视觉目标只保留最新一份；失效后锁定，不随视觉恢复自动重启。

    target 的 monotonic 必须是相机曝光时间；T_base_hole 的 +Z 指向孔内。
    接口不提供实体碰撞检测。5 mm 为目标间距，2 mm 为反馈及插补参考的停机下限。
    """

    IMAGE_MAX_AGE = .75
    HEARTBEAT_MAX_AGE = .5
    PLAN_MAX_TIME = .75
    STEP_MAX_TIME = 2.
    MAX_STEP_M = .0005
    STABLE_STEP_M = .005
    MAX_STEP_RAD = math.radians(.5)
    FILTER_STATIC_TIME_S = .25
    FILTER_MOVING_TIME_S = .06
    FILTER_WINDOW_S = .25
    FILTER_MIN_SPAN_S = .10
    FILTER_LINEAR_SPEED_RANGE = (.0005, .003)
    FILTER_ANGULAR_SPEED_RANGE = (math.radians(.5), math.radians(3.))
    STABLE_TIME_S = .35
    HOLD_ENTER_M = .00015
    HOLD_EXIT_M = .00035
    HOLD_ENTER_RAD = math.radians(.1)
    HOLD_EXIT_RAD = math.radians(.3)
    MIN_GAP_M = .002
    RGB_REQUIRED_DISTANCE_M = .080
    RGB_WAIT_MAX_TIME = 3.
    STAGES = ("退针", "抬升", "移向孔上方", "靠近孔口", "垂直对准", "自动跟随")

    def __init__(self, reader, get_snapshot, on_event):
        self.reader, self.on_event = reader, on_event
        # 控制与规划直接读取接收缓存；不依赖界面 80 ms 刷新。
        read_pose = getattr(reader, 'read_pose', None)
        self.get_snapshot = read_pose if callable(read_pose) else get_snapshot
        self.executor = MotionExecutor(reader, self.get_snapshot, self._motion_event)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._stop_done = threading.Event()
        self._stop_result = None
        self._worker = self._watchdog = None
        self._target = None
        self._filtered_target = self._stationary_anchor = None
        self._stationary = self._holding = False
        self._filter_alpha = 1.
        self._filter_history = deque(maxlen=128)
        self._filter_time_s = self.FILTER_STATIC_TIME_S
        self._filter_linear_speed = self._filter_angular_speed = self._filter_motion_weight = 0.
        self._step_timeout = self.STEP_MAX_TIME
        self._heartbeat = 0.
        self._planning_at = self._step_at = None
        self._active_plan = self._step_target = None
        self._step_result = None
        self._connection_id = None
        self._tool = None
        self._stage = 0
        self._failure = None
        self._vision_wait_at = None
        self._vision_wait_state_at = 0.
        self.locked = False

    @property
    def running(self):
        return (self.executor.running
                or any(worker is not None and worker.is_alive()
                       for worker in (self._worker, self._watchdog)))

    @property
    def latest_target(self):
        with self._lock:
            return copy.deepcopy(self._target)

    def _event(self, kind, **data):
        self.on_event(kind, data)

    @staticmethod
    def _validated_target(value, gap):
        if not isinstance(value, dict) or value.get("hole_id") is None:
            raise MotionError("跟随目标缺少所选孔编号。")
        quality = value.get("quality")
        if (value.get("valid") is False or quality is False
                or (isinstance(quality, dict) and quality.get("accepted") is False)):
            raise MotionError("工件定位质量失效，自动跟随已停止。")
        target = copy.deepcopy(value)
        target["T_base_hole"] = _transform(target["T_base_hole"], "孔口目标")
        target["T_base_tcp"] = _transform(target["T_base_tcp"], "跟随目标")
        stamp = float(target["monotonic"])
        if not math.isfinite(stamp) or not 0 <= time.monotonic() - stamp <= HoleFollower.IMAGE_MAX_AGE:
            raise MotionError("孔位图像超过 0.75 秒或曝光时间异常，自动跟随已停止。")
        target["monotonic"] = stamp
        hole, tcp = target["T_base_hole"], target["T_base_tcp"]
        expected = hole.copy()
        expected[:3, 3] -= gap * hole[:3, 2]
        distance, angle = _errors(expected, tcp)
        if distance > 1e-6 or angle > 1e-6:
            raise MotionError("跟随目标必须位于同一孔外 5 mm 并垂直对准孔口。")
        return target

    def start(self, T_flange_tcp, initial_target, gap_m=.005, clearance_m=.05, speed_percent=5):
        if self.locked or self.running or (self._worker is not None and self._worker.is_alive()):
            raise MotionError("自动跟随正在运行或已停止锁定；故障后须处理原因并重新连接。")
        if not math.isclose(float(gap_m), .005, abs_tol=1e-9):
            raise MotionError("自动跟随固定保持孔外 5 mm，不执行接触或插孔。")
        if not math.isfinite(float(clearance_m)) or not .005 <= clearance_m <= .15:
            raise MotionError("退针和抬升距离须为 5～150 mm。")
        if type(speed_percent) is not int or not 1 <= speed_percent <= 5:
            raise MotionError("自动跟随速度比例须为 1～5 的整数。")
        self._tool = _transform(T_flange_tcp, "TCP 偏移")
        target = self._validated_target(initial_target, gap_m)
        pose, _, _, status = self.executor._live()
        send = self.executor._send_state()
        if send["connection_id"] is None or send["command_pending"] or status["motion_status"] != 0:
            raise MotionError("机械臂未连接或上一运动尚未到位，不能启动自动跟随。")
        self._connection_id = send["connection_id"]
        self._initial_tcp = pose @ self._tool
        self._gap, self._clearance, self._speed = float(gap_m), float(clearance_m), speed_percent
        with self._lock:
            self._target = target
            self._filtered_target = copy.deepcopy(target)
            self._stationary_anchor = copy.deepcopy(target)
            self._filter_history.clear()
            self._filter_history.append((target['monotonic'], target['T_base_hole'].copy()))
            self._filter_time_s = self.FILTER_STATIC_TIME_S
            self._filter_linear_speed = self._filter_angular_speed = self._filter_motion_weight = 0.
            self._filter_alpha = 1.
            self._stationary = self._holding = False
            self._heartbeat = time.monotonic()
            self._stage = 0
            self._vision_wait_at = None
            self._stop.clear()
        self._guard()
        self._event("started", stage=self.STAGES[0], target=target,
                    gap_mm=gap_m * 1000, speed_percent=speed_percent,
                    minimum_gap_mm=self.MIN_GAP_M * 1000,
                    image_max_age_s=self.IMAGE_MAX_AGE,
                    translation_step_mm=self.MAX_STEP_M * 1000,
                    stable_translation_step_mm=self.STABLE_STEP_M * 1000,
                    rgb_required_distance_mm=self.RGB_REQUIRED_DISTANCE_M * 1000,
                    rgb_wait_max_time_s=self.RGB_WAIT_MAX_TIME,
                    filter_static_time_s=self.FILTER_STATIC_TIME_S,
                    filter_moving_time_s=self.FILTER_MOVING_TIME_S,
                    filter_window_s=self.FILTER_WINDOW_S,
                    filter_min_span_s=self.FILTER_MIN_SPAN_S,
                    filter_linear_speed_range_mm_s=[v*1000 for v in self.FILTER_LINEAR_SPEED_RANGE],
                    filter_angular_speed_range_deg_s=[math.degrees(v) for v in self.FILTER_ANGULAR_SPEED_RANGE],
                    hold_enter_mm=self.HOLD_ENTER_M * 1000, hold_exit_mm=self.HOLD_EXIT_M * 1000,
                    rotation_step_deg=math.degrees(self.MAX_STEP_RAD))
        self._worker = threading.Thread(target=self._run, name="孔外自动跟随", daemon=True)
        self._watchdog = threading.Thread(target=self._watch, name="孔位跟随实时监护", daemon=True)
        self._worker.start()
        self._watchdog.start()

    def update_heartbeat(self):
        with self._lock:
            self._heartbeat = time.monotonic()

    def update(self, target):
        """同一曝光帧仅续界面心跳，不续图像寿命；旧曝光结果不覆盖新结果。"""
        self.update_heartbeat()
        if self._stop.is_set() or self.locked:
            return False
        try:
            latest = self._validated_target(target, self._gap)
            with self._lock:
                old = self._target
                if latest["hole_id"] != old["hole_id"]:
                    raise MotionError("跟随中的孔编号已改变，自动跟随已停止。")
                elapsed = latest["monotonic"] - old["monotonic"]
                if elapsed <= 0:
                    return False
                distance, angle = _errors(old["T_base_hole"], latest["T_base_hole"])
                # 给视觉亚毫米抖动留量，但拒绝突变与明显超过分段跟随能力的运动。
                if distance > max(.001, elapsed * .010) or angle > max(math.radians(.5), elapsed * math.radians(10)):
                    raise MotionError("工件移动过快或孔位发生跳变，自动跟随已停止。")
                self._target = latest
                self._filter_target(latest, elapsed)
            return True
        except Exception as exc:
            self._fail(str(exc))
            return False

    def _adaptive_filter_time(self, latest):
        """只用基坐标系新曝光的短窗趋势；相机随臂运动不直接参与运动判定。"""
        stamp = latest['monotonic']
        history = self._filter_history
        history.append((stamp, latest['T_base_hole'].copy()))
        # 使用真实秒数窗口并保留一个边界样本，低帧率也能估计趋势。
        while len(history) > 2 and history[1][0] <= stamp - self.FILTER_WINDOW_S:
            history.popleft()
        linear_speed = angular_speed = weight = 0.
        span = stamp - history[0][0]
        # 高频起步的两帧不判运动；较低帧率的两帧须至少相隔 0.2 秒。
        enough_samples = len(history) >= 3 or (len(history) == 2 and span >= .2 - 1e-9)
        if enough_samples and span >= self.FILTER_MIN_SPAN_S - 1e-9:
            times = np.array([item[0] - stamp for item in history])
            centered = times - times.mean()
            denominator = float(centered @ centered)
            positions = np.stack([item[1][:3, 3] for item in history])
            positions -= positions[0]
            reference = history[0][1][:3, :3]
            rotations = np.stack([reference.T @ item[1][:3, :3] for item in history])
            vectors = Rotation.from_matrix(rotations).as_rotvec()
            # 多帧有符号趋势抑制正负来回的噪声；不能把逐帧位移绝对值当速度。
            linear_speed = float(np.linalg.norm(centered @ positions / denominator))
            angular_speed = float(np.linalg.norm(centered @ vectors / denominator))
            linear_low, linear_high = self.FILTER_LINEAR_SPEED_RANGE
            angular_low, angular_high = self.FILTER_ANGULAR_SPEED_RANGE
            ratio = float(np.clip(max((linear_speed-linear_low)/(linear_high-linear_low),
                                      (angular_speed-angular_low)/(angular_high-angular_low)), 0., 1.))
            weight = ratio*ratio*(3.-2.*ratio)
        self._filter_linear_speed, self._filter_angular_speed = linear_speed, angular_speed
        self._filter_motion_weight = weight
        self._filter_time_s = self.FILTER_STATIC_TIME_S + weight*(self.FILTER_MOVING_TIME_S-self.FILTER_STATIC_TIME_S)
        return self._filter_time_s

    def _filter_target(self, latest, elapsed):
        """只在新曝光到达时平滑；原始目标始终独立用于所有安全判据。"""
        anchor = self._stationary_anchor
        distance, angle = _errors(anchor['T_base_hole'], latest['T_base_hole'])
        if distance > .00025 or angle > math.radians(.2):
            self._stationary_anchor = copy.deepcopy(latest)
            self._stationary = False
        else:
            self._stationary = latest['monotonic'] - anchor['monotonic'] >= self.STABLE_TIME_S
        previous = self._filtered_target['T_base_hole']
        raw = latest['T_base_hole']
        alpha = -math.expm1(-elapsed / self._adaptive_filter_time(latest))
        filtered = previous.copy()
        filtered[:3, 3] += alpha * (raw[:3, 3] - previous[:3, 3])
        rotation = Rotation.from_matrix(previous[:3, :3].T @ raw[:3, :3]).as_rotvec()
        filtered[:3, :3] = previous[:3, :3] @ Rotation.from_rotvec(alpha * rotation).as_matrix()
        # 限制平滑滞后，不能把真实移动长时间积压在滤波器中。
        offset = filtered[:3, 3] - raw[:3, 3]
        norm = float(np.linalg.norm(offset))
        if norm > .0005:
            filtered[:3, 3] = raw[:3, 3] + offset * (.0005 / norm)
        rotation = Rotation.from_matrix(raw[:3, :3].T @ filtered[:3, :3]).as_rotvec()
        norm = float(np.linalg.norm(rotation))
        if norm > math.radians(.3):
            filtered[:3, :3] = raw[:3, :3] @ Rotation.from_rotvec(rotation * (math.radians(.3) / norm)).as_matrix()
        self._filtered_target = copy.deepcopy(latest)
        self._filtered_target['T_base_hole'] = filtered
        tcp = filtered.copy()
        tcp[:3, 3] -= self._gap * filtered[:3, 2]
        self._filtered_target['T_base_tcp'] = tcp
        self._filtered_target['filter'] = dict(
            time_constant_s=self._filter_time_s, alpha=alpha, motion_weight=self._filter_motion_weight,
            window_span_s=latest['monotonic']-self._filter_history[0][0],
            linear_speed_mm_s=self._filter_linear_speed*1000,
            angular_speed_deg_s=math.degrees(self._filter_angular_speed),
            translation_lag_mm=float(np.linalg.norm(filtered[:3, 3]-raw[:3, 3]))*1000,
            rotation_lag_deg=math.degrees(_errors(raw, filtered)[1]))
        self._filter_alpha = alpha

    def _hold_at_target(self, distance, angle):
        if self._holding:
            self._holding = distance <= self.HOLD_EXIT_M and angle <= self.HOLD_EXIT_RAD
        else:
            self._holding = distance <= self.HOLD_ENTER_M and angle <= self.HOLD_ENTER_RAD
        return self._holding

    def invalidate(self, reason):
        return self._fail(str(reason))

    def stop(self, reason="自动跟随已停止。"):
        return self._fail(str(reason), kind="stopped")

    def _fail(self, reason, kind="error"):
        with self._lock:
            first = not self._stop.is_set()
            if first:
                self._failure = reason
                self.locked = True
                self._stop.set()
        if not first:
            self._stop_done.wait(.75)
            with self._lock:
                return dict(self._stop_result or {"requested": True, "sent": False,
                                                  "error": "电子急停仍在发送中，请检查控制器实际状态。"})
        result = self.executor.stop()
        with self._lock:
            self._stop_result = dict(result)
            self._stop_done.set()
        try:
            self._event(kind, message=reason, stop_result=result, stage=self.STAGES[self._stage],
                        target=self.latest_target)
        except Exception:
            pass
        return result

    def _guard(self):
        now = time.monotonic()
        with self._lock:
            if self._stop.is_set() or self.locked:
                raise MotionError(self._failure or "自动跟随已停止。")
            target = self._target
            if target is None or not 0 <= now - target["monotonic"] <= self.IMAGE_MAX_AGE:
                raise MotionError("孔位图像超过 0.75 秒，自动跟随已停止。")
            if not 0 <= now - self._heartbeat <= self.HEARTBEAT_MAX_AGE:
                raise MotionError("界面监护心跳中断，自动跟随已停止。")
            if self._planning_at is not None and now - self._planning_at > self.PLAN_MAX_TIME:
                raise MotionError("跟随逆解计算超过 0.75 秒，拒绝执行旧目标。")
            if self._step_at is not None and now - self._step_at > self._step_timeout:
                raise MotionError(f"跟随运动段超过 {self._step_timeout:g} 秒未完成，自动跟随已停止。")
            plan, planned_target = self._active_plan, self._step_target
        state = self.executor._send_state()
        if state["connection_id"] != self._connection_id:
            raise MotionError("机械臂连接已断开或更换，自动跟随已停止。")
        pose, _, _, _ = self.executor._live()
        hole = target["T_base_hole"]
        normal, point = -hole[:3, 2], hole[:3, 3]
        tcp = pose @ self._tool
        gap = float((tcp[:3, 3] - point) @ normal)
        if gap < self.MIN_GAP_M:
            raise MotionError(f"针尖距孔平面仅 {gap*1000:.2f} mm，低于 2 mm 停机下限。")
        if planned_target is not None:
            distance, angle = _errors(planned_target["T_base_hole"], hole)
            if distance > .002 or angle > math.radians(2):
                raise MotionError("当前小段执行期间工件偏移过大，自动跟随已停止。")
        if plan is not None:
            if not self._rgb_ready(target):
                previous_tcp = tcp
                for step in plan['steps']:
                    next_tcp = _transform(step['T_base_flange']) @ self._tool
                    if self._requires_rgb(target, previous_tcp, next_tcp):
                        raise MotionError('近孔运动缺少所选孔的连续 RGB 实测确认，自动跟随已停止。')
                    previous_tcp = next_tcp
            # 按最新孔平面重新核对 MoveL 法兰插补曲线，不能只检查旧目标端点。
            MotionExecutor._check_board_clearance(plan, pose, point, normal, self.MIN_GAP_M)
        return target, tcp

    @staticmethod
    def _rgb_ready(target):
        quality = target.get('quality')
        if not isinstance(quality, dict):
            return False
        support = quality.get('selected_hole_support')
        return (target.get('vision_ready') is True
                and quality.get('accepted') is True
                and quality.get('tracking_method') == 'rgb_holes'
                and quality.get('measured_this_frame') is True
                and isinstance(support, dict)
                and str(support.get('id')) == str(target['hole_id'])
                and support.get('detected') is True and support.get('accepted') is True)

    def _requires_rgb(self, target, start_tcp, end_tcp=None):
        """按原始孔位检查整段针尖参考曲线，含法兰插补引起的工具旋转偏差。"""
        point = target['T_base_hole'][:3, 3]
        start = start_tcp[:3, 3]
        if end_tcp is None:
            distance = float(np.linalg.norm(start - point))
            margin = 0.
        else:
            delta = end_tcp[:3, 3] - start
            square = float(delta @ delta)
            fraction = float(np.clip((point - start) @ delta / square, 0., 1.)) if square > 1e-18 else 0.
            distance = float(np.linalg.norm(start + fraction * delta - point))
            angle = _errors(start_tcp, end_tcp)[1]
            # MoveL 的针尖曲线与端点弦之间偏差不超过 radius * angle² / 8。
            margin = float(np.linalg.norm(self._tool[:3, 3])) * angle ** 2 / 8.
        return distance - margin <= self.RGB_REQUIRED_DISTANCE_M + 1e-9

    def _wait_for_rgb(self, target, tcp):
        now = time.monotonic()
        with self._lock:
            first = self._vision_wait_at is None
            if first:
                self._vision_wait_at = now
            waited = now - self._vision_wait_at
        if waited > self.RGB_WAIT_MAX_TIME:
            raise MotionError('等待所选孔的连续 RGB 实测确认超过 3 秒，自动跟随已停止。')
        if first or now - self._vision_wait_state_at >= .5:
            self._event('state', **self._state(target, tcp))
            self._vision_wait_state_at = now
        self._stop.wait(.03)

    def _watch(self):
        while not self._stop.wait(.03):
            try:
                self._guard()
            except Exception as exc:
                self._fail(str(exc))
                return

    def _motion_event(self, kind, data):
        if kind in ("done", "error", "stopped", "paused"):
            with self._lock:
                self._step_result = (kind, data)

    def _waypoint(self, target):
        observation = target["T_base_tcp"].copy()
        observation[:3, 3] -= .005 * observation[:3, 2]
        route = travel_targets(self._initial_tcp, observation, self._clearance,
                               align_after_approach=True)
        return route[self._stage] if self._stage < 5 else target["T_base_tcp"]

    @classmethod
    def _small_step(cls, current, target, max_step_m=None):
        distance, angle = _errors(current, target)
        fraction = min(1., (cls.MAX_STEP_M if max_step_m is None else max_step_m) / max(distance, 1e-12),
                       cls.MAX_STEP_RAD / max(angle, 1e-12))
        result = current.copy()
        result[:3, 3] += fraction * (target[:3, 3] - current[:3, 3])
        rotation = Rotation.from_matrix(current[:3, :3].T @ target[:3, :3]).as_rotvec()
        result[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(rotation * fraction).as_matrix()
        return result

    def _state(self, target, tcp):
        hole = target["T_base_hole"]
        normal = -hole[:3, 2]
        delta = tcp[:3, 3] - hole[:3, 3]
        gap = float(delta @ normal)
        with self._lock:
            filtered = copy.deepcopy(self._filtered_target)
            stationary, alpha = self._stationary, self._filter_alpha
            waiting = self._vision_wait_at is not None
        quality = target.get('quality') or {}
        stage = '正在核对孔位' if waiting else self.STAGES[self._stage] + (' · 稳定保持' if self._holding else '')
        return {"stage": stage, "hole_id": target["hole_id"],
                "gap_mm": gap * 1000,
                "projection_error_mm": float(np.linalg.norm(delta - gap * normal)) * 1000,
                "angle_error_deg": math.degrees(_errors(tcp, target["T_base_tcp"])[1]),
                "target_monotonic": target["monotonic"],
                "tracking_method": quality.get('tracking_method') if isinstance(quality, dict) else None,
                "vision_ready": self._rgb_ready(target), "waiting_for_rgb": waiting,
                "T_base_tcp": tcp.tolist(), "target": target, "filtered_target": filtered,
                "stationary_target": stationary, "filter_alpha": alpha, "holding": self._holding}

    def _run(self):
        enabled_once = False
        state_at = 0.
        lag_started = None
        reached = False
        follow_started_at = None
        started_at = time.monotonic()
        try:
            while not self._stop.is_set():
                target, current = self._guard()
                quality = target.get('quality')
                source = quality.get('tracking_method') if isinstance(quality, dict) else None
                with self._lock:
                    command_target = copy.deepcopy(self._filtered_target)
                    stationary = self._stationary
                waypoint = self._waypoint(command_target)
                distance, angle = _errors(current, waypoint)
                if not self._rgb_ready(target) and self._requires_rgb(target, current):
                    self._wait_for_rgb(target, current)
                    continue
                if self._rgb_ready(target):
                    self._vision_wait_at = None
                if self._stage < 5 and distance <= .0001 and angle <= math.radians(.05):
                    self._stage += 1
                    if self._stage == 5:
                        follow_started_at = time.monotonic()
                    self._event("state", **self._state(target, current))
                    continue
                now = time.monotonic()
                if not reached and now - started_at > 240:
                    raise MotionError("初次接近超过 4 分钟，自动跟随已停止。")
                if self._stage == 5:
                    raw_distance, raw_angle = _errors(current, target['T_base_tcp'])
                    reached = reached or (raw_distance <= .0005 and raw_angle <= math.radians(.5))
                    if raw_distance > .010 or raw_angle > math.radians(5):
                        raise MotionError("针尖已跟不上孔位变化，自动跟随已停止。")
                    if not reached and now - follow_started_at > 10.:
                        raise MotionError("最终接近 10 秒内未能跟近孔位，自动跟随已停止。")
                    if reached and raw_distance > .002:
                        lag_started = now if lag_started is None else lag_started
                        if now - lag_started > 2.:
                            raise MotionError("针尖连续 2 秒未能跟近孔位，自动跟随已停止。")
                    else:
                        lag_started = None
                    self._hold_at_target(distance, angle)
                if now - state_at >= .5:
                    self._event("state", **self._state(target, current))
                    state_at = now
                if (self._stage == 5 and self._holding) or (distance <= .0001 and angle <= math.radians(.05)):
                    self._stop.wait(.03)
                    continue
                snapshot = copy.deepcopy(self.get_snapshot())
                pose, _, _ = _snapshot(snapshot, fresh=True)
                # 目标持续稳定且无需调姿时适当延长直线；动态目标及调姿维持原限长。
                actual_tcp = pose @ self._tool
                _, actual_angle = _errors(actual_tcp, waypoint)
                step_limit = (self.STABLE_STEP_M if stationary and actual_angle <= math.radians(.05)
                              else self.MAX_STEP_M)
                step_target = self._small_step(actual_tcp, waypoint, step_limit)
                if step_limit > self.MAX_STEP_M:
                    # 朝向已在原验收范围内时保持当前朝向，避免把反馈微差当成调姿命令。
                    step_target[:3, :3] = actual_tcp[:3, :3]
                if not self._rgb_ready(target) and self._requires_rgb(target, actual_tcp, step_target):
                    self._wait_for_rgb(target, actual_tcp)
                    continue
                self._vision_wait_at = None
                with self._lock:
                    self._planning_at = time.monotonic()
                    self._step_target = target
                self._event("planning", stage=self.STAGES[self._stage], target=target,
                            filtered_target=command_target, stationary_target=stationary,
                            translation_limit_mm=step_limit * 1000,
                            tracking_method=source,
                            vision_ready=self._rgb_ready(target),
                            start_snapshot=snapshot, T_base_tcp=step_target.tolist())
                plan = plan_tcp_path(snapshot, self._tool, [step_target])
                self._guard()
                plan["metadata"]["hole_follow"] = {
                    "hole_id": target["hole_id"], "nominal_gap_mm": 5.,
                    "minimum_gap_mm": self.MIN_GAP_M * 1000,
                    "tracking_method": source,
                    "vision_ready": self._rgb_ready(target),
                    "target_monotonic": target["monotonic"], "stage": self.STAGES[self._stage]}
                with self._lock:
                    self._planning_at = None
                    self._active_plan = plan
                    self._step_result = None
                    self._step_at = time.monotonic()
                    self._step_timeout = 5. if step_limit > self.MAX_STEP_M else self.STEP_MAX_TIME
                self._guard()
                self.executor.start(plan, speed_percent=self._speed, live_guard=self._guard,
                                    allow_enable=not enabled_once)
                while self.executor.running and not self._stop.wait(.02):
                    pass
                if self._stop.is_set():
                    break
                with self._lock:
                    result = self._step_result
                    self._active_plan = self._step_target = self._step_at = None
                if result is None or result[0] != "done":
                    data = {} if result is None else result[1]
                    raise MotionError(data.get("message", "跟随小段未确认到位，已停止。"))
                enabled_once = True
                target, current = self._guard()
                self._event("step", **self._state(target, current))
        except Exception as exc:
            self._fail(str(exc))
