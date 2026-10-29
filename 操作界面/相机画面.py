"""RealSense 实时预览与同源 RGB-D 快照；不保存图像或判断接触。"""

import copy
import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from 界面主题 import TK_COLORS, Tk按钮


def _color_rectification(intrinsics):
    """用当前彩色流内参建立去畸变映射，保持原分辨率和 K。"""
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    width, height = int(intrinsics.width), int(intrinsics.height)
    K = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                  [0.0, intrinsics.fy, intrinsics.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    coefficients = np.asarray(intrinsics.coeffs, dtype=np.float64)
    if (width <= 0 or height <= 0 or not np.isfinite(K).all() or
            K[0, 0] <= 0 or K[1, 1] <= 0 or coefficients.shape != (5,) or
            not np.isfinite(coefficients).all()):
        raise ValueError("当前彩色流内参无效，定位快照不可用")
    source = {
        "source": "当前 RealSense 彩色流的出厂内参",
        "width": width, "height": height,
        "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx), "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coeffs": coefficients.tolist(),
    }
    supported = (rs.distortion.none, rs.distortion.brown_conrady,
                 rs.distortion.modified_brown_conrady, rs.distortion.inverse_brown_conrady)
    if intrinsics.model not in supported:
        raise ValueError(f"不支持彩色流畸变模型 {intrinsics.model}，定位快照不可用")
    if intrinsics.model == rs.distortion.none and np.any(coefficients != 0.0):
        raise ValueError("彩色流声明无畸变，但系数非零，定位快照不可用")
    if not np.any(coefficients != 0.0):
        return K, source, None
    if intrinsics.model == rs.distortion.brown_conrady:
        maps = cv2.initUndistortRectifyMap(K, coefficients, None, K,
                                         (width, height), cv2.CV_32FC1)
    else:
        # modified/inverse Brown 的投影顺序与 OpenCV 标准 Brown 不同。
        # 按 librealsense rs2_project_point_to_pixel 的正向投影建立逆采样表。
        u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                           np.arange(height, dtype=np.float64))
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        radius2 = x * x + y * y
        k1, k2, p1, p2, k3 = coefficients
        radial = 1.0 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
        radial_x, radial_y = x * radial, y * radial
        distorted_x = radial_x + 2.0 * p1 * radial_x * radial_y + p2 * (radius2 + 2.0 * radial_x * radial_x)
        distorted_y = radial_y + 2.0 * p2 * radial_x * radial_y + p1 * (radius2 + 2.0 * radial_y * radial_y)
        maps = ((distorted_x * K[0, 0] + K[0, 2]).astype(np.float32),
                (distorted_y * K[1, 1] + K[1, 2]).astype(np.float32))
    if not all(np.isfinite(mapping).all() for mapping in maps):
        raise ValueError("彩色流去畸变映射无效，定位快照不可用")
    return K, source, maps


class CameraPanel(tk.Frame):
    """独立预览面板；on_status(message) 仅在 Tk 主线程调用。"""

    def __init__(self, parent, logger, on_status=None):
        self.colors = getattr(parent.winfo_toplevel(), "colors", TK_COLORS)
        super().__init__(parent, bg=self.colors["panel"])
        self.logger = logger
        self.on_status = on_status
        self.last_frame_at = 0.0
        self._closed = False
        self._state = "idle"
        self._generation = 0
        self._events = queue.Queue()
        self._stop = threading.Event()
        self._scan_thread = None
        self._stream_thread = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_snapshot = None
        self._snapshot_generation = None
        self._snapshot_error = "相机尚未打开"
        self._image = None
        self._photo = None
        self._devices = []
        self._selected = tk.StringVar()
        self._status_text = tk.StringVar(value="连接 D435i 后打开相机，观察针尖与固定参考点。")

        tk.Label(self, text="实时相机画面", bg=self.colors["panel"], fg=self.colors["text"],
                 font=("Noto Sans CJK SC", 12, "bold"), anchor="w").pack(fill="x", pady=(0, 8))
        devices = tk.Frame(self, bg=self.colors["panel"])
        devices.pack(fill="x")
        self.device_combo = ttk.Combobox(devices, textvariable=self._selected,
                                        state="readonly", width=25)
        self.device_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.refresh_button = self._button(devices, "刷新设备", self.refresh_devices)
        self.refresh_button.pack(side="left")
        actions = tk.Frame(self, bg=self.colors["panel"])
        actions.pack(fill="x", pady=(8, 8))
        self.open_button = self._button(actions, "打开相机", self.start, primary=True)
        self.open_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.close_button = self._button(actions, "关闭相机", self.stop)
        self.close_button.pack(side="left", fill="x", expand=True)
        self.canvas = tk.Canvas(self, width=440, height=230, bg=self.colors["selected"],
                                highlightthickness=1, highlightbackground=self.colors["muted"])
        self.canvas.pack(fill="x")
        self.canvas.bind("<Configure>", self._resize)
        self.status_label = tk.Label(self, textvariable=self._status_text,
                                     bg=self.colors["panel"], fg=self.colors["muted"],
                                     font=("Noto Sans CJK SC", 10), anchor="w",
                                     justify="left", wraplength=440)
        self.status_label.pack(fill="x", pady=(8, 4))
        tk.Label(self, text="画面辅助对点；每次接触固定点后，请点击“确认触点并采集”。",
                 bg=self.colors["panel"], fg=self.colors["muted"], font=("Noto Sans CJK SC", 9),
                 anchor="w", justify="left", wraplength=440).pack(fill="x")
        self._placeholder("尚未打开相机")
        self._controls()
        self._poll_id = self.after(60, self._poll)
        self._scan_id = self.after_idle(self.refresh_devices)

    def _button(self, parent, text, command, primary=False):
        return Tk按钮(parent, text, command, primary=primary, compact=True)

    def _controls(self):
        available = self._state in ("idle", "error") and not self._closed
        self.device_combo.configure(state="readonly" if available else "disabled")
        self.refresh_button.configure(state="normal" if available else "disabled")
        enabled = bool(available and self._devices)
        self.open_button.configure(state="normal" if enabled else "disabled")
        self.close_button.configure(state="normal" if self._state in ("starting", "live") else "disabled")

    def _status(self, message, error=False):
        self._status_text.set(message)
        self.status_label.configure(fg=self.colors["error"] if error else self.colors["muted"])
        if self.on_status is not None:
            self.on_status(message)

    def _invalidate_snapshot(self, reason, generation=None):
        with self._frame_lock:
            if generation is None or generation == self._generation:
                self._frame_snapshot = None
                self._snapshot_generation = None
                self._snapshot_error = reason

    def _snapshot_invalid_reason(self, max_age_s):
        """调用者持有 _frame_lock；不读取 Tk 控件。"""
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            return "快照有效期必须为正的有限秒数"
        if self._closed or self._stop.is_set() or self._state not in ("starting", "live"):
            return self._snapshot_error or "相机未在取流"
        if self._frame_snapshot is None or self._snapshot_generation != self._generation:
            return self._snapshot_error or "尚未收到有效 RGB-D 帧"
        age = time.monotonic() - self._frame_snapshot["monotonic"]
        if age < 0.0 or age > max_age_s:
            return "RGB-D 快照已过期，等待新帧"
        return ""

    def get_frame_snapshot(self, max_age_s=0.5):
        """返回独立 RGB-D 副本；停流、错误或过期时返回 None。

        rgb 为 uint8 RGB，depth 为 float32 米（0 表示无深度）。两者位于
        同一 undistorted_color 像素空间，K 为当前彩色流 K，dist_coeffs 为 0。
        raw_rgb 是去畸变前的 uint8 RGB，须与 source_intrinsics 配合使用，
        不可套用零畸变参数。raw_rgb 与 rgb 的数组副本互不共享内存。
        monotonic 是主机收到帧的时刻，不是曝光时刻；device_timestamp_ms
        和 frame_id 取自该彩色帧。source_intrinsics 保留去畸变前的出厂内参。
        预览消费 _latest_frame 不会消费这里的快照。无效原因可读 frame_snapshot_error。
        """
        with self._frame_lock:
            if self._snapshot_invalid_reason(max_age_s):
                return None
            return copy.deepcopy(self._frame_snapshot)

    @property
    def frame_snapshot_error(self):
        """默认 0.5 秒有效期内的快照无效原因；有效时为空字符串。"""
        with self._frame_lock:
            return self._snapshot_invalid_reason(0.5)

    def refresh_devices(self):
        if self._closed or self._state not in ("idle", "error"):
            return
        self._state = "scanning"
        self._generation += 1
        generation = self._generation
        self._status("正在查找 RealSense 相机…")
        self._controls()

        def scan():
            try:
                import pyrealsense2 as rs
                devices = []
                context = rs.context()
                for device in context.query_devices():
                    name = device.get_info(rs.camera_info.name)
                    serial = device.get_info(rs.camera_info.serial_number)
                    devices.append((name, serial))
                devices.sort(key=lambda item: ("D435" not in item[0].upper(), item[1]))
                self._events.put((generation, "devices", devices))
            except Exception as exc:
                self.logger.exception("查询 TCP 预览相机失败")
                self._events.put((generation, "error", "无法枚举相机：" + str(exc)))

        # RealSense 持有 C++ 设备对象；禁止解释器越过仍在释放设备的线程退出。
        self._scan_thread = threading.Thread(target=scan, daemon=False, name="TCP相机枚举")
        self._scan_thread.start()

    def start(self):
        if self._closed or self._state not in ("idle", "error"):
            return
        index = self.device_combo.current()
        if not 0 <= index < len(self._devices):
            self._status("请选择一个 RealSense 设备，再打开相机。", error=True)
            return
        name, serial = self._devices[index]
        self._state = "starting"
        self._generation += 1
        generation = self._generation
        stop = self._stop = threading.Event()
        self.last_frame_at = 0.0
        self._invalidate_snapshot("正在等待 RGB-D 首帧")
        self._status(f"正在打开 {name} · {serial}…")
        self._placeholder("正在连接相机…")
        self._controls()

        def stream():
            pipeline = None
            config = profile = align = frames = aligned = color = depth = None
            started = False
            failure = None
            try:
                import cv2
                import numpy as np
                import pyrealsense2 as rs
                if stop.is_set():
                    return
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                profile = pipeline.start(config)
                started = True
                align = rs.align(rs.stream.color)
                color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                color_serial = profile.get_device().get_info(rs.camera_info.serial_number)
                depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
                if not math.isfinite(depth_scale) or depth_scale <= 0.0:
                    raise RuntimeError("当前设备返回了无效的深度单位")
                rectification_error = ""
                try:
                    K, source_intrinsics, maps = _color_rectification(color_intrinsics)
                    source_intrinsics["color_serial"] = color_serial
                    source_intrinsics["depth_scale_m"] = depth_scale
                except ValueError as exc:
                    rectification_error = str(exc)
                    self._invalidate_snapshot(rectification_error, generation)
                    self.logger.error("TCP 相机保留原始 RGB 预览，定位快照禁用：%s", rectification_error)
                self.logger.info("打开 TCP 相机 RGB-D 预览，彩色/深度 640×480@30，对齐至彩色流；设备 %s", color_serial)
                last_received = time.monotonic()
                while not stop.is_set():
                    received, frames = pipeline.try_wait_for_frames(300)
                    now = time.monotonic()
                    if stop.is_set():
                        break
                    if not received:
                        if time.monotonic() - last_received > 3.0:
                            raise RuntimeError("超过 3 秒未收到新画面")
                        continue
                    aligned = align.process(frames)
                    color = aligned.get_color_frame()
                    depth = aligned.get_depth_frame()
                    if not color:
                        self._invalidate_snapshot("当前帧缺少彩色图像", generation)
                        continue
                    last_received = now
                    pixels = np.asanyarray(color.get_data()).copy()
                    raw_rgb = pixels.copy()
                    snapshot = None
                    snapshot_error = rectification_error
                    if not rectification_error:
                        if pixels.shape != (color_intrinsics.height, color_intrinsics.width, 3):
                            raise RuntimeError("当前彩色帧尺寸与流内参不一致")
                        if maps is not None:
                            pixels = cv2.remap(pixels, maps[0], maps[1], cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        if not depth:
                            snapshot_error = "当前帧缺少对齐的深度图像"
                        else:
                            meters = np.asarray(depth.get_data(), dtype=np.float32) * np.float32(depth_scale)
                            if meters.shape != pixels.shape[:2]:
                                raise RuntimeError("对齐深度与彩色图像的尺寸不一致")
                            if maps is not None:
                                meters = cv2.remap(meters, maps[0], maps[1], cv2.INTER_NEAREST,
                                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            snapshot = {
                                "rgb": pixels, "raw_rgb": raw_rgb, "depth": meters, "K": K,
                                "dist_coeffs": np.zeros(5, dtype=np.float64),
                                "color_serial": color_serial,
                                "width": int(color_intrinsics.width),
                                "height": int(color_intrinsics.height),
                                "frame_id": int(color.get_frame_number()),
                                "monotonic": now,
                                "device_timestamp_ms": float(color.get_timestamp()),
                                "image_space": "undistorted_color",
                                "source_intrinsics": source_intrinsics,
                            }
                    with self._frame_lock:
                        if generation == self._generation and not stop.is_set() and not self._closed:
                            self._latest_frame = (generation, now, pixels)
                            self._frame_snapshot = snapshot
                            self._snapshot_generation = generation if snapshot is not None else None
                            self._snapshot_error = snapshot_error
            except Exception as exc:
                if not stop.is_set():
                    self.logger.exception("TCP 相机预览取流失败")
                    failure = str(exc)
            finally:
                self._invalidate_snapshot("相机取流失败：" + failure if failure else "相机已停止取流", generation)
                # 先放掉帧句柄，再停流；所有 RealSense 对象在工作线程内释放完毕。
                depth = color = aligned = frames = align = None
                if started:
                    try:
                        pipeline.stop()
                    except Exception:
                        self.logger.exception("停止 TCP 相机预览失败")
                profile = config = pipeline = None
                self.logger.info("TCP 相机 RGB-D 预览已停止")
                self._events.put((generation, "stopped", failure))

        self._stream_thread = threading.Thread(target=stream, daemon=False, name="TCP相机预览")
        self._stream_thread.start()

    def stop(self):
        if self._state not in ("starting", "live", "stopping"):
            return
        self._state = "stopping"
        self._stop.set()
        self._invalidate_snapshot("相机正在停止取流")
        self.last_frame_at = 0.0
        self._image = None
        self._photo = None
        with self._frame_lock:
            self._latest_frame = None
        self._placeholder("正在关闭相机…")
        self._status("正在释放相机…")
        self._controls()

    def is_live(self):
        return (not self._closed and self._state == "live" and
                self.last_frame_at > 0.0 and time.monotonic() - self.last_frame_at < 1.5)

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                generation, kind, data = self._events.get_nowait()
                if generation != self._generation:
                    continue
                if kind == "devices":
                    previous = self._selected.get()
                    self._devices = data
                    labels = [f"{name} · {serial}" for name, serial in data]
                    self.device_combo.configure(values=labels)
                    self._selected.set(previous if previous in labels else (labels[0] if labels else ""))
                    self._state = "idle"
                    self._status("已找到相机，点击“打开相机”查看实时画面。" if data else
                                 "未发现 RealSense：检查 USB 连接后点击“刷新设备”。", error=not bool(data))
                elif kind == "error":
                    self._state = "error"
                    self._invalidate_snapshot(data)
                    self._status(data, error=True)
                elif kind == "stopped":
                    self._state = "error" if data else "idle"
                    self._invalidate_snapshot("相机取流失败：" + str(data) if data else "相机已关闭")
                    self.last_frame_at = 0.0
                    self._image = None
                    self._photo = None
                    with self._frame_lock:
                        self._latest_frame = None
                    self._placeholder("相机不可用" if data else "相机已关闭")
                    self._status(("相机断开或被占用：请检查 USB，关闭其他占用相机的程序，再重新打开。\n" +
                                  str(data)[:160]) if data else "相机已关闭，已采集的位姿仍保留。", error=bool(data))
                self._controls()
        except queue.Empty:
            pass
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        if frame is not None and frame[0] == self._generation and self._state in ("starting", "live"):
            first_frame = self._state != "live"
            self._state = "live"
            self.last_frame_at = frame[1]
            self._image = Image.fromarray(frame[2])
            self._paint()
            snapshot_error = self.frame_snapshot_error
            message = ("实时 RGB · 定位快照不可用：" + snapshot_error if snapshot_error else
                       "实时 RGB-D · 640 × 480 · 已对齐、去畸变；请观察针尖与固定参考点。")
            if self._status_text.get() != message:
                self._status(message, error=bool(snapshot_error))
            if first_frame:
                self._controls()
        if self._state == "live" and not self.is_live():
            self._image = None
            self._photo = None
            self._placeholder("暂未收到新画面")
            if self._status_text.get() != "画面已暂停，请检查相机连接。":
                self._status("画面已暂停，请检查相机连接。", error=True)
        elif self._state == "live" and self._status_text.get() == "画面已暂停，请检查相机连接。":
            snapshot_error = self.frame_snapshot_error
            self._status("实时 RGB · 定位快照不可用：" + snapshot_error if snapshot_error else
                         "实时 RGB-D · 640 × 480 · 已对齐、去畸变；请观察针尖与固定参考点。",
                         error=bool(snapshot_error))
        self._poll_id = self.after(60, self._poll)

    def _placeholder(self, text):
        self.canvas.delete("all")
        self.canvas.create_text(max(1, self.canvas.winfo_width()) / 2,
                                max(1, self.canvas.winfo_height()) / 2,
                                text=text, fill=self.colors["muted"], font=("Noto Sans CJK SC", 11),
                                tags="placeholder")

    def _画框高度(self, width):
        """画框高度跟随宽度按画面比例伸展；无画面时按 4∶3 预留。"""
        width = max(1, width - 2)
        if self._image is not None:
            image_width, image_height = self._image.size
            ratio = min(1.6, max(0.4, image_height / max(1, image_width)))
        else:
            ratio = 0.75
        return round(width * ratio) + 2

    def _适配画框(self):
        height = self._画框高度(self.canvas.winfo_width())
        if abs(self.canvas.winfo_height() - height) > 1:
            self.canvas.configure(height=height)

    def _paint(self):
        if self._image is None:
            return
        self._适配画框()
        width = max(1, self.canvas.winfo_width() - 2)
        height = max(1, self.canvas.winfo_height() - 2)
        image = self._image.copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image, master=self)
        self.canvas.delete("all")
        self.canvas.create_image((width + 2) / 2, (height + 2) / 2,
                                 image=self._photo, anchor="center")

    def _resize(self, event):
        self.status_label.configure(wraplength=max(180, event.width - 8))
        self._适配画框()
        if self._image is not None:
            self._paint()
        else:
            self.canvas.coords("placeholder", event.width / 2, event.height / 2)

    def close(self):
        """停止取流并有界等待设备释放；返回 True 表示后台线程已全部结束。"""
        if self._closed:
            return not any(thread and thread.is_alive()
                           for thread in (self._stream_thread, self._scan_thread))
        self._closed = True
        self._stop.set()
        self._invalidate_snapshot("相机面板已关闭")
        self.last_frame_at = 0.0
        for callback in (self._poll_id, self._scan_id):
            try:
                self.after_cancel(callback)
            except tk.TclError:
                pass
        self._image = self._photo = None
        with self._frame_lock:
            self._latest_frame = None
        # 正常停止最多等待当前 300 ms 取帧及设备 stop；共享 5 秒关闭期限。
        # 极端驱动超时由非守护线程继续完成释放，不能强杀线程导致 C++ 析构崩溃。
        deadline = time.monotonic() + 5.0
        workers = (self._stream_thread, self._scan_thread)
        for worker in workers:
            if worker and worker.is_alive() and worker is not threading.current_thread():
                worker.join(max(0.0, deadline - time.monotonic()))
        pending = [worker.name for worker in workers if worker and worker.is_alive()]
        if pending:
            self.logger.warning("相机关闭等待超时，后台仍在释放设备：%s", "、".join(pending))
        return not pending
