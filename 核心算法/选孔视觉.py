"""孔表与投影，以及只用内存管道通信的视觉工作进程客户端。"""

from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import pickle
import queue
import signal
import struct
import subprocess
import threading

import cv2
import numpy as np


ORIGIN = Path('/home/w/codes/foundationpose_origin')
HOLES_PATH = ORIGIN / 'FoundationPose/P61123024-0010-005-001-A_hole_candidates.json'
MESH_PATH = ORIGIN / 'FoundationPose/my_data_online/mesh/P61123024-0010-005-001-A.obj'
FP_PYTHON = Path('/home/w/codes/foundationpose/.venv/bin/python')
SAM_PYTHON = ORIGIN / 'sam3/.venv/bin/python'
WORKER = Path(__file__).resolve().parents[1] / '运行支持/定位工作进程.py'
MAX_PACKET_BYTES = 128 * 1024 * 1024


class VisionError(RuntimeError):
    pass


class FollowVisionGate:
    """自动跟随的远近滞回及同孔新帧确认；不读取设备、不续旧帧时间。"""

    ENTER_RGB_M = .100
    EXIT_RGB_M = .140
    CONFIRM_FRAMES = 3

    def __init__(self, hole_id, generation):
        self.hole_id = str(hole_id)
        self.generation = int(generation)
        self.epoch = 0
        self.near = False
        self.confirmed = 0
        self.last_stamp = None

    def update_distance(self, distance):
        distance = float(distance)
        if not math.isfinite(distance) or distance < 0:
            raise VisionError('针尖与所选孔的距离无效。')
        near = distance < self.EXIT_RGB_M if self.near else distance <= self.ENTER_RGB_M
        if near != self.near:
            self.near = near
            self.epoch += 1
            self.confirmed = 0
            self.last_stamp = None
        return self.request()

    def request(self):
        return dict(selected_hole_id=self.hole_id, follow_generation=self.generation,
                    follow_epoch=self.epoch, near_rgb=self.near)

    def matches(self, result):
        return all(result.get(key) == value for key, value in self.request().items())

    def observe(self, result):
        """旧任务/旧模式/重复曝光不计数，近区检测失败由调用者停止运动。"""
        if not self.matches(result):
            return False
        stamp = float(result['monotonic'])
        if not math.isfinite(stamp):
            raise VisionError('孔位测量缺少有效曝光时间。')
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return False
        self.last_stamp = stamp
        quality = result.get('quality', {})
        support = quality.get('selected_hole_support') or {}
        if self.near:
            good = (quality.get('accepted') is True and quality.get('measured_this_frame') is True
                    and quality.get('tracking_method') == 'rgb_holes'
                    and str(support.get('id', '')) == self.hole_id
                    and support.get('detected') is True and support.get('accepted') is True)
            self.confirmed = min(self.confirmed + 1, self.CONFIRM_FRAMES) if good else 0
            if not good:
                reason = str(support.get('reason') or quality.get('reason', ''))
                if 'neighbor' in reason or 'ambiguous' in reason or 'duplicate' in reason:
                    detail = '所选孔与邻孔的对应不明确'
                elif 'too_few' in reason or 'not_measured' in reason or 'not_independently' in reason:
                    detail = '画面中可靠实测的孔不足或所选孔被遮挡'
                elif 'residual' in reason:
                    detail = '孔轮廓与定位结果的偏差过大'
                elif 'grazing' in reason or 'spread' in reason:
                    detail = '当前视角或可见孔分布不足以核对位置'
                elif quality.get('depth') is not None and not quality.get('accepted'):
                    detail = '当前图像与深度未共同通过定位检查'
                else:
                    detail = '当前画面未能可靠确认同一个目标孔'
                raise VisionError(detail + '，自动跟随已停止。')
        elif not quality.get('accepted'):
            raise VisionError('整件跟踪未通过检查：' + str(quality.get('reason', '定位不可信')))
        return True

    @property
    def ready(self):
        return not self.near or self.confirmed >= self.CONFIRM_FRAMES


def load_holes(path=HOLES_PATH):
    """读取原 CAD 毫米孔表；返回米单位的名义孔口及孔轴。"""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('unit_assumption') != 'mm':
        raise VisionError('孔表必须明确使用毫米单位。')
    holes, seen = [], set()
    for source in data['holes']:
        identity = str(source['hole_id'])
        point = np.array([source[f'center_{axis}_cad_mm'] for axis in 'xyz'], float) * .001
        normal = np.array([source[f'axis_{axis}_cad'] for axis in 'xyz'], float)
        radius = float(source['radius_mm']) * .001
        if (identity in seen or not np.isfinite(point).all() or not np.isfinite(normal).all()
                or np.linalg.norm(normal) < 1e-8 or not math.isfinite(radius) or radius <= 0):
            raise VisionError('孔表包含重复编号或无效几何。')
        seen.add(identity)
        holes.append(dict(id=identity, center_cad_m=point.tolist(),
                          normal_cad=(normal / np.linalg.norm(normal)).tolist(), radius_m=radius,
                          diameter_mm=radius * 2000., group=source.get('group', '')))
    if not holes:
        raise VisionError('孔表没有可选孔。')
    return holes


def validate_pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4, rtol=0)
            or not math.isclose(float(np.linalg.det(pose[:3, :3])), 1., abs_tol=1e-4)):
        raise VisionError('CAD 到相机的位姿矩阵无效。')
    return pose


def project_holes(T_cam_cad, K, image_shape, dist_coeffs=None, holes=None):
    """当前 12 mm 工件的可见面投影；visible 仅表示几何可见，不代表实测到孔。"""
    pose = validate_pose(T_cam_cad)
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise VisionError('相机内参无效。')
    height, width = map(int, image_shape[:2])
    if min(height, width) <= 0:
        raise VisionError('画面尺寸无效。')
    coefficients = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if coefficients.size == 0:
        coefficients = np.zeros(5)
    if coefficients.size not in (4, 5, 8, 12, 14) or not np.isfinite(coefficients).all():
        raise VisionError('畸变参数必须是 OpenCV 格式的有限数值。')
    camera_cad = -pose[:3, :3].T @ pose[:3, 3]
    side = 1. if camera_cad[1] >= 0 else -1.
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    result = []
    for hole in load_holes() if holes is None else holes:
        center = np.array(hole['center_cad_m'], dtype=float)
        if not np.allclose(hole['normal_cad'], [0, 1, 0], atol=1e-6):
            raise VisionError('当前孔投影仅适用于已核对的 Y 轴孔工件。')
        center[1] = side * .006
        radius = float(hole['radius_m'])
        mouth_radius = radius
        if side > 0:
            if math.isclose(radius, .004, abs_tol=1e-8):
                mouth_radius = .0045
            elif math.isclose(radius, .0033, abs_tol=1e-8):
                mouth_radius = .0055
        ring = center + mouth_radius * np.column_stack((np.cos(angles), np.zeros(32), np.sin(angles)))
        points = np.vstack((center, ring))
        cam = points @ pose[:3, :3].T + pose[:3, 3]
        uv = cv2.projectPoints(points, rvec, pose[:3, 3], K, coefficients)[0].reshape(-1, 2)
        finite = bool(np.isfinite(uv).all() and (cam[:, 2] > .001).all())
        diameter = float(min(cv2.fitEllipse(uv[1:].astype(np.float32))[1])) if finite else 0.
        in_frame = finite and bool((uv[:, 0] >= 0).all() and (uv[:, 0] < width).all()
                                   and (uv[:, 1] >= 0).all() and (uv[:, 1] < height).all())
        result.append(dict(id=hole['id'], uv=uv[0].tolist() if finite else None,
                           center_cad_m=center.tolist(), normal_cad=[0., side, 0.], radius_m=radius,
                           diameter_mm=radius * 2000., visible=in_frame, diameter_px=diameter,
                           measured=False))
    return result


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError('视觉工作进程已关闭管道。')
        data.extend(chunk)
    return bytes(data)


def _read_packet(stream):
    size = struct.unpack('!Q', _read_exact(stream, 8))[0]
    if size > MAX_PACKET_BYTES:
        raise VisionError('视觉通信数据包过大。')
    # 仅接受本机自行启动的子进程管道，不接收网络或用户提供的 pickle 文件。
    return _decode_arrays(pickle.loads(_read_exact(stream, size)))


def _encode_arrays(value):
    # 三个既有 Python 环境的 NumPy 主版本不同；不传递 NumPy 自身的 pickle reducer。
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'buifc':
            raise VisionError('视觉通信只支持数值数组。')
        array = np.ascontiguousarray(value)
        return dict(__tcp_array__=True, dtype=array.dtype.str, shape=array.shape, data=array.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_arrays(item) for item in value]
    return value


def _decode_arrays(value):
    if isinstance(value, dict):
        if value.get('__tcp_array__') is True:
            dtype = np.dtype(value['dtype'])
            shape = tuple(value['shape'])
            data = value['data']
            if (dtype.kind not in 'buifc' or len(shape) > 8 or any(not isinstance(n, int) or n < 0 for n in shape)
                    or math.prod(shape) * dtype.itemsize != len(data)):
                raise VisionError('视觉通信数组格式无效。')
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {key: _decode_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode_arrays(item) for item in value]
    return value


def _packet_bytes(value):
    payload = pickle.dumps(_encode_arrays(value), protocol=5)
    if len(payload) > MAX_PACKET_BYTES:
        raise VisionError('视觉帧超过通信大小限制。')
    return struct.pack('!Q', len(payload)) + payload


def _write_packet(stream, value):
    payload = memoryview(_packet_bytes(value))
    while payload:
        written = stream.write(payload)
        if not written:
            raise EOFError('视觉工作进程已关闭写入管道。')
        payload = payload[written:]
    stream.flush()


class VisionClient:
    """start/request 在后台线程调用；close 不等待正在运行的推理。"""

    def __init__(self, logger=None):
        self.logger = logger
        self._process = None
        self._state_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._ready = None
        self._closed = False
        self._events = None
        self._stderr = deque(maxlen=40)

    def _spawn(self):
        if not FP_PYTHON.is_file() or not WORKER.is_file():
            raise VisionError('找不到现有 FoundationPose Python 环境或定位工作进程。')
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4')
        events, errors = queue.Queue(), deque(maxlen=40)
        with self._state_lock:
            if self._closed:
                raise VisionError('视觉会话已关闭；重新定位请创建新会话。')
            try:
                process = subprocess.Popen([str(FP_PYTHON), '-B', str(WORKER)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           bufsize=0, env=environment, start_new_session=True)
            except OSError as exc:
                raise VisionError(f'无法启动视觉工作进程：{exc}') from exc
            self._process, self._events, self._stderr = process, events, errors

        def read_results():
            try:
                while True:
                    events.put(_read_packet(process.stdout))
            except Exception as exc:
                events.put(dict(ok=False, error=str(exc)))

        def read_errors():
            for line in iter(process.stderr.readline, b''):
                errors.append(line.decode('utf-8', errors='replace').rstrip())

        threading.Thread(target=read_results, name='定位结果接收', daemon=True).start()
        threading.Thread(target=read_errors, name='定位错误接收', daemon=True).start()
        return process

    def _wait(self, timeout):
        try:
            response = self._events.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise VisionError(f'视觉处理超过 {timeout:g} 秒，进程已停止；请重新开始定位。') from exc
        with self._state_lock:
            if self._closed:
                raise VisionError('视觉会话已关闭。')
        if not isinstance(response, dict) or not response.get('ok'):
            message = response.get('error', '视觉工作进程返回无效结果。') if isinstance(response, dict) else '视觉协议无效。'
            detail = '\n'.join(list(self._stderr)[-8:])
            if self.logger:
                self.logger.warning('视觉处理失败：%s\n%s', message, detail)
            raise VisionError(message)
        return response

    def _ensure_started(self):
        with self._state_lock:
            if self._closed:
                raise VisionError('视觉会话已关闭；重新定位请创建新会话。')
            if self._process is not None and self._process.poll() is None and self._ready is not None:
                return self._ready
        self._spawn()
        try:
            ready = self._wait(60.)
            with self._state_lock:
                if self._closed:
                    raise VisionError('视觉会话已关闭。')
                self._ready = ready
        except Exception:
            self.close()
            raise
        return self._ready

    def start(self):
        with self._rpc_lock:
            return self._ensure_started()

    def request(self, op, frame=None, **kwargs):
        with self._rpc_lock:
            self._ensure_started()
            timeout = float(kwargs.pop('timeout', {'init': 300., 'warmup': 120., 'track': 45., 'refine': 30.}.get(op, 10.)))
            if not math.isfinite(timeout) or timeout <= 0:
                raise VisionError('视觉处理超时必须为正数。')
            try:
                _write_packet(self._process.stdin, dict(op=op, frame=frame, **kwargs))
            except (OSError, ValueError, AttributeError) as exc:
                self.close()
                raise VisionError('视觉工作进程已结束，请重新开始定位。') from exc
            return self._wait(timeout)

    def close(self):
        with self._state_lock:
            self._closed = True
            process, self._process = self._process, None
            self._ready = None
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        def reap():
            try:
                process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

        threading.Thread(target=reap, name='结束视觉进程', daemon=True).start()
