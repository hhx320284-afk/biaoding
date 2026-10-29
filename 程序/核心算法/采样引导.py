"""只在内存中分析实时标定帧，指导稳定取样与分布覆盖。

清晰度、曝光和仿射倾斜量都是采样启发指标，不能替代最后的重投影检验。
不使用单应矩阵残差过滤图像，以免把鱼眼镜头的正常弯曲当作坏照片。
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np


class Guide:
    """evaluate 只给取样建议；调用者成功收下内存样本后再调用 accept。"""

    def __init__(self, min_samples=20, max_samples=45, sharpness=60,
                 stable_seconds=0.8, target_rms=1.0):
        if not 3 <= int(min_samples) <= int(max_samples):
            raise ValueError("最少样本必须不少于 3，且不能超过采样上限。")
        if not all(math.isfinite(float(v)) and float(v) > 0
                   for v in (sharpness, stable_seconds, target_rms)):
            raise ValueError("清晰度、稳定时长和目标重投影误差必须为正数。")
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        self.sharpness = float(sharpness)
        self.stable_seconds = float(stable_seconds)
        self.target_rms = float(target_rms)
        self.reset()

    def reset(self):
        self.samples = []
        self.coverage = [0] * 9
        self._stable_feature = None
        self._stable_since = None

    @staticmethod
    def _feature(detection, frame_shape):
        h, w = frame_shape[:2]
        points = np.asarray(detection.image_points, dtype=np.float64).reshape(-1, 2)
        obj = np.asarray(detection.object_points, dtype=np.float64).reshape(-1, 3)[:, :2]
        if len(points) < 8 or len(points) != len(obj) or not np.isfinite(points).all() or not np.isfinite(obj).all():
            raise ValueError("有效角点不足或坐标无效。")
        outline = getattr(detection, "outline", None)
        outline = points if outline is None else np.asarray(outline, dtype=np.float64).reshape(-1, 2)
        if len(outline) < 3 or not np.isfinite(outline).all():
            outline = points
        hull = cv2.convexHull(outline.astype(np.float32))
        area = float(cv2.contourArea(hull)) / (w * h)
        if area <= 0:
            raise ValueError("检测角点没有形成可用区域。")
        moments = cv2.moments(hull)
        center = np.array([moments["m10"] / moments["m00"] / w,
                           moments["m01"] / moments["m00"] / h])
        design = np.column_stack((obj - obj.mean(axis=0), np.ones(len(obj))))
        mapping, _, rank, _ = np.linalg.lstsq(design, points - points.mean(axis=0), rcond=None)
        if rank < 3:
            raise ValueError("角点分布近似共线，请展开显示整块标定板。")
        linear = mapping[:2].T
        singular = np.linalg.svd(linear, compute_uv=False)
        ratio = float(singular[-1] / max(singular[0], 1e-12))
        gram = linear.T @ linear / max(singular[0] ** 2, 1e-12)
        tilt = np.array([1 - gram[0, 0], 1 - gram[1, 1], gram[0, 1]])
        angle = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
        cell = min(2, max(0, int(center[1] * 3))) * 3 + min(2, max(0, int(center[0] * 3)))
        return {"center": center, "scale": math.sqrt(area), "area": area,
                "tilt": tilt, "ratio": ratio, "angle": angle, "cell": cell,
                "hull": hull, "points": points, "bbox": (outline.min(0), outline.max(0))}

    @staticmethod
    def _angle_distance(a, b):
        return abs((a - b + 180) % 360 - 180)

    def _near(self, a, b, *, stability=False):
        if stability:
            return (np.linalg.norm(a["center"] - b["center"]) < 0.008
                    and abs(math.log(a["scale"] / b["scale"])) < 0.035
                    and np.linalg.norm(a["tilt"] - b["tilt"]) < 0.055
                    and self._angle_distance(a["angle"], b["angle"]) < 2.5)
        return (np.linalg.norm(a["center"] - b["center"]) < 0.075
                and abs(math.log(a["scale"] / b["scale"])) < 0.14
                and np.linalg.norm(a["tilt"] - b["tilt"]) < 0.18
                and self._angle_distance(a["angle"], b["angle"]) < 14)

    def _coverage_state(self):
        cells = sum(value > 0 for value in self.coverage)
        rows = sum(any(self.coverage[row * 3:row * 3 + 3]) for row in range(3))
        columns = sum(any(self.coverage[column::3]) for column in range(3))
        if self.samples:
            scales = [s["scale"] for s in self.samples]
            scale_range = math.log(max(scales) / min(scales))
            tilt_range = max(np.linalg.norm(a["tilt"] - b["tilt"])
                             for a in self.samples for b in self.samples)
        else:
            scale_range = tilt_range = 0.0
        complete = (len(self.samples) >= self.min_samples and cells >= 6 and rows == 3 and columns == 3
                    and scale_range >= 0.26 and tilt_range >= 0.22)
        progress = (0.55 * min(1, len(self.samples) / self.min_samples)
                    + 0.25 * min(1, cells / 6)
                    + 0.10 * min(1, scale_range / 0.26)
                    + 0.10 * min(1, tilt_range / 0.22))
        return {"sample_count": len(self.samples), "coverage_cells": cells,
                "coverage_rows": rows, "coverage_columns": columns,
                "scale_range": scale_range, "tilt_range": float(tilt_range),
                "complete": complete, "progress": 1.0 if complete else min(0.99, progress),
                "paused": len(self.samples) >= self.max_samples and not complete}

    def _next_instruction(self, state):
        names = ["左上", "上方", "右上", "左侧", "中央", "右侧", "左下", "下方", "右下"]
        if state["coverage_cells"] < 6 or state["coverage_rows"] < 3 or state["coverage_columns"] < 3:
            order = (4, 0, 2, 6, 8, 1, 3, 5, 7)
            cell = min(order, key=lambda n: (self.coverage[n],
                       sum(self.coverage[n // 3 * 3:n // 3 * 3 + 3]) + sum(self.coverage[n % 3::3])))
            return (f"把标定板移到画面{names[cell]}目标区域，完整入镜后保持不动。",
                    ((cell % 3 + 0.5) / 3, (cell // 3 + 0.5) / 3))
        if state["scale_range"] < 0.26:
            return "保持完整入镜，把标定板靠近或远离相机，改变约三分之一的画面大小。", None
        if state["tilt_range"] < 0.22:
            return "围绕水平轴或竖直轴适度倾斜标定板（约 25°～40°），再保持不动。", None
        return "换到尚未充分采集的位置，改变远近或倾斜方向，保持标定板完整清晰。", None

    def accept(self, detection, frame_shape):
        """记录已由调用者收下的样本姿态；不保存任何照片或设置。"""
        if len(self.samples) >= self.max_samples:
            return False
        feature = self._feature(detection, frame_shape)
        if any(self._near(feature, old) for old in self.samples):
            return False
        self.samples.append({key: value for key, value in feature.items()
                             if key not in ("hull", "points", "bbox")})
        self.coverage[feature["cell"]] += 1
        self._stable_feature = None
        self._stable_since = None
        return True

    def evaluate(self, frame, detection, now=None):
        now = time.monotonic() if now is None else float(now)
        state = self._coverage_state()
        instruction, target = self._next_instruction(state)
        metrics = dict(state)
        metrics.update({"sharpness_threshold": self.sharpness,
                        "target_rms": self.target_rms,
                        "quality_note": "清晰度与倾斜为启发指标；最终以重投影检验为准。"})
        result = {"accepted": False, "ready": False, "status": "等待标定板",
                  "instruction": instruction, "color": "red", "progress": state["progress"],
                  "metrics": metrics, "target": target, "complete": state["complete"],
                  "coverage": list(self.coverage)}

        def reject(status, advice, color="red", next_target=None):
            self._stable_feature = None
            self._stable_since = None
            result.update(status=status, instruction=advice, color=color, target=next_target)
            return result

        if state["complete"]:
            result.update(status="采样覆盖已满足", instruction="可开始标定；求解后请查看重投影误差和逐张误差。", color="green", target=None)
            return result
        if state["paused"]:
            return reject("已到采样上限，覆盖仍不足", "自动采样已暂停。" + instruction + " 可重置重采，或提高采样上限后继续。", "amber", target)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return reject("尚无相机图像", "打开所选相机并等待实时画面。")
        if detection is None:
            return reject("未识别到标定板", "让完整标定板朝向相机；检查板类型、字典与行列数，改善光照并避开反光。", next_target=(0.5, 0.5))
        try:
            feature = self._feature(detection, frame.shape)
        except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error):
            return reject("角点不足或分布异常", "完整展示标定板，避免遮挡、模糊和过度倾斜。", next_target=(0.5, 0.5))
        h, w = frame.shape[:2]
        found = int(getattr(detection, "found_count", len(feature["points"])))
        expected = int(getattr(detection, "expected_count", found))
        metrics.update({"found_count": found, "expected_count": expected,
                        "board_area": feature["area"], "center": feature["center"].tolist(),
                        "tilt_proxy": float(1 - feature["ratio"]), "rotation_deg": feature["angle"]})
        metrics["detected_fraction"] = found / expected if expected > 0 else 1.0
        if expected > 0 and found < expected:
            metrics["detection_hint"] = "当前只识别到部分角点；尽量完整入镜以改善覆盖统计。"
        low, high = feature["bbox"]
        margin_x, margin_y = max(6, w * 0.015), max(6, h * 0.015)
        if low[0] < margin_x or high[0] > w - margin_x or low[1] < margin_y or high[1] > h - margin_y:
            directions = []
            if low[0] < margin_x:
                directions.append("向右")
            if high[0] > w - margin_x:
                directions.append("向左")
            if low[1] < margin_y:
                directions.append("向下")
            if high[1] > h - margin_y:
                directions.append("向上")
            return reject("标定板靠近画面边缘", "、".join(directions) + "移动标定板，必要时远离相机，保留少量边距。", next_target=(0.5, 0.5))
        if feature["area"] < 0.035 or min(cv2.minAreaRect(feature["hull"])[1]) < 60:
            return reject("标定板太小", "将标定板靠近相机，让角点更清晰；保持完整入镜。")
        if feature["area"] > 0.80:
            return reject("标定板太大", "将标定板远离相机，为移动和倾斜留出边距。")
        if feature["ratio"] < 0.12:
            return reject("标定板投影过窄", "减小倾斜角度，让板面更朝向相机，避免侧着拍。")

        # 与既有离线质量检查一致：曝光与清晰度只统计标定板凸包，
        # 腐蚀掩膜避免板外背景和板外轮廓边缘虚增清晰度。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        x0, y0 = np.maximum(np.floor(feature["hull"].reshape(-1, 2).min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(feature["hull"].reshape(-1, 2).max(0)).astype(int) + 1, [w, h])
        roi = gray[y0:y1, x0:x1]
        mask = np.zeros(roi.shape, np.uint8)
        local_hull = np.rint(feature["hull"] - [x0, y0]).astype(np.int32)
        cv2.fillConvexPoly(mask, local_hull, 255)
        inner = cv2.erode(mask, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if not inner.any():
            return reject("标定板可用区域太小", "靠近相机并减小倾斜，确保标定板完整可见。")
        pixels = roi[inner]
        p05, p50, p95 = np.percentile(pixels, [5, 50, 95])
        clarity = float(cv2.Laplacian(roi, cv2.CV_64F)[inner].var())
        metrics.update({"sharpness": clarity, "brightness_p05": float(p05),
                        "brightness_p50": float(p50), "brightness_p95": float(p95)})
        if p95 < 80:
            return reject("标定板偏暗", "增加均匀光照或适度提高曝光，直到黑白格清楚可分。")
        if p05 > 170 or p95 - p05 < 55:
            return reject("标定板对比度不足", "避开反光，调整曝光或光照，让黑色标记和白底分明。")
        if clarity < self.sharpness:
            return reject("标定板清晰度不足", "保持标定板不动，调整对焦与光照；必要时靠近。清晰度阈值可在采样参数中调整。")
        if any(self._near(feature, old) for old in self.samples):
            return reject("这个姿态已经采集过", instruction, "amber", target)
        if (self._stable_feature is None or self._stable_since is None
                or now < self._stable_since or not self._near(feature, self._stable_feature, stability=True)):
            self._stable_feature = feature
            self._stable_since = now
        elapsed = now - self._stable_since
        remaining = max(0.0, self.stable_seconds - elapsed)
        metrics["stable_remaining"] = remaining
        result.update(ready=True, color="green", target=None)
        if elapsed >= self.stable_seconds:
            result.update(accepted=True, status="姿态合格，可以自动采集",
                          instruction="保持不动，正在采集本姿态；采集后按提示换位置。")
        else:
            result.update(status=f"姿态合格，保持稳定 {remaining:.1f} 秒",
                          instruction="保持相机与标定板不动，稳定后会自动采集。")
        return result
