"""多板型检测、针孔/鱼眼内参标定及真实像素重投影误差。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from 核心算法.标定板 import BoardSpec, _dictionary, opencv_board


@dataclass
class Detection:
    object_points: np.ndarray
    image_points: np.ndarray
    ids: np.ndarray
    outline: np.ndarray
    found_count: int
    expected_count: int


@lru_cache(maxsize=24)
def _detector(spec: BoardSpec):
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.015
    if spec.kind == "charuco":
        # ChArUco 的棋盘角点插值自行细化；标签边缘亚像素可能受邻近黑格影响。
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        return cv2.aruco.CharucoDetector(opencv_board(spec), detectorParams=params)
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(_dictionary(spec.dictionary_name), params)


def detect_board(frame: np.ndarray, spec: BoardSpec) -> Detection | None:
    """识别至少 16 个非共线角点；角点坐标单位分别为米和原图像素。"""
    spec.validate()
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("检测画面必须为灰度、BGR 或 BGRA 图像。")
    if gray.dtype != np.uint8:
        raise ValueError("检测画面必须为 8 位图像；请先由相机驱动转换。")

    if spec.kind == "chessboard":
        pattern = (spec.columns - 1, spec.rows - 1)
        found, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        xy = np.array([(col, row) for row in range(1, spec.rows)
                       for col in range(1, spec.columns)], dtype=np.float64) * spec.square_mm / 1000
        objects = np.column_stack((xy, np.zeros(len(xy))))
        pixels = corners.reshape(-1, 2)
        ids = np.arange(len(objects), dtype=np.int32)
    elif spec.kind == "charuco":
        corners, ids, _, _ = _detector(spec).detectBoard(gray)
        if corners is None or ids is None or len(ids) < 16:
            return None
        ids = ids.reshape(-1)
        objects = np.asarray(opencv_board(spec).getChessboardCorners())[ids]
        pixels = corners.reshape(-1, 2)
    else:
        corners, marker_ids, _ = _detector(spec).detectMarkers(gray)
        if marker_ids is None:
            return None
        board = opencv_board(spec)
        object_map = {int(mid): obj for mid, obj in zip(board.getIds().reshape(-1), board.getObjPoints())}
        detected_ids = marker_ids.reshape(-1)
        unique, counts = np.unique(detected_ids, return_counts=True)
        duplicates = set(unique[counts > 1].tolist())
        pairs = [(int(mid), corner) for mid, corner in zip(detected_ids, corners)
                 if int(mid) in object_map and int(mid) not in duplicates]
        if not pairs:
            return None
        objects = np.concatenate([object_map[mid] for mid, _ in pairs])
        pixels = np.concatenate([corner.reshape(-1, 2) for _, corner in pairs])
        ids = np.concatenate([np.arange(mid * 4, mid * 4 + 4, dtype=np.int32) for mid, _ in pairs])

    objects = np.ascontiguousarray(objects, dtype=np.float64).reshape(-1, 3)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64).reshape(-1, 2)
    if (len(objects) < 16 or not np.isfinite(pixels).all()
            or np.linalg.matrix_rank(objects[:, :2] - objects[:, :2].mean(axis=0)) < 2
            or np.linalg.matrix_rank(pixels - pixels.mean(axis=0)) < 2):
        return None
    order = np.argsort(ids)
    return Detection(objects[order], pixels[order], ids[order].reshape(-1, 1),
                     cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).astype(np.float64),
                     len(pixels), spec.expected_count)


def _samples(samples: list[dict], image_size=None):
    objects, pixels = [], []
    for index, sample in enumerate(samples):
        try:
            obj = np.asarray(sample["object_points"], np.float64).reshape(-1, 3)
            img = np.asarray(sample["image_points"], np.float64).reshape(-1, 2)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"第 {index + 1} 张样本的角点格式不正确。") from exc
        if len(obj) != len(img) or len(obj) < 16:
            raise ValueError(f"第 {index + 1} 张样本需要至少 16 对对应角点。")
        if not np.isfinite(obj).all() or not np.isfinite(img).all():
            raise ValueError(f"第 {index + 1} 张样本存在无效坐标。")
        if (np.linalg.matrix_rank(obj - obj.mean(axis=0)) < 2
                or np.linalg.matrix_rank(img - img.mean(axis=0)) < 2):
            raise ValueError(f"第 {index + 1} 张样本角点共线，无法标定。")
        if np.ptp(obj[:, 2]) > 1e-6:
            raise ValueError("当前标定流程要求平面标定板。")
        if image_size is not None:
            width, height = image_size
            if np.any(img[:, 0] < 0) or np.any(img[:, 0] >= width) or np.any(img[:, 1] < 0) or np.any(img[:, 1] >= height):
                raise ValueError(f"第 {index + 1} 张样本角点超出画面，请核对采样分辨率。")
            if sample.get("image_size") is not None and tuple(sample["image_size"]) != tuple(image_size):
                raise ValueError("样本分辨率不一致，请清空后使用同一分辨率重新采样。")
        objects.append(np.ascontiguousarray(obj.reshape(-1, 1, 3)))
        pixels.append(np.ascontiguousarray(img.reshape(-1, 1, 2)))
    return objects, pixels


def _project(objects, rvec, tvec, K, D, model):
    if model == "fisheye":
        return cv2.fisheye.projectPoints(objects, rvec, tvec, K, D)[0].reshape(-1, 2)
    return cv2.projectPoints(objects, rvec, tvec, K, D)[0].reshape(-1, 2)


def _assess(objects, pixels, K, D, rvecs, tvecs, model, image_size):
    width, height = image_size
    expected_coefficients = 4 if model == "fisheye" else 5
    if K.shape != (3, 3) or D.size != expected_coefficients:
        raise ValueError("求解的内参矩阵或畸变系数数量与镜头模型不匹配。")
    if (not np.isfinite(K).all() or not np.isfinite(D).all()
            or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-9)):
        raise ValueError("求解产生无效焦距或畸变参数，请增加不同倾角及不同位置的样本。")
    # 全为正面或重复姿态可得到极小 RMS，却出现几十亿像素的虚假焦距。
    # 此宽界限只拒绝数值退化；较温和的异常仍在下方作为可见警告报告。
    if (min(K[0, 0], K[1, 1]) < 1e-4 * min(width, height)
            or max(K[0, 0], K[1, 1]) > 1e3 * max(width, height)
            or abs(K[0, 2]) > 1e3 * width or abs(K[1, 2]) > 1e3 * height):
        raise ValueError("内参出现极端焦距或主点，当前姿态无法可靠约束标定；请增加明显倾斜及远近变化的样本。")
    per_view, all_squared, reprojected, max_radius, max_angle = [], [], [], 0.0, 0.0
    for index, (obj, img, rvec, tvec) in enumerate(zip(objects, pixels, rvecs, tvecs)):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            raise ValueError(f"第 {index + 1} 张样本的位姿无效。")
        camera = obj.reshape(-1, 3) @ cv2.Rodrigues(rvec)[0].T + np.asarray(tvec).reshape(1, 3)
        if np.any(camera[:, 2] <= 0):
            raise ValueError(f"第 {index + 1} 张样本被求解到相机后方，请检查板参数与姿态多样性。")
        radius = np.linalg.norm(camera[:, :2], axis=1) / camera[:, 2]
        max_radius = max(max_radius, float(np.max(radius)))
        max_angle = max(max_angle, float(np.max(np.arctan(radius))))
        projected = _project(obj, rvec, tvec, K, D, model)
        squared = np.sum((projected - img.reshape(-1, 2)) ** 2, axis=1)
        if not np.isfinite(squared).all():
            raise ValueError("重投影坐标包含无效数值。")
        per_view.append(float(np.sqrt(np.mean(squared))))
        all_squared.append(squared)
        reprojected.append(projected.tolist())

    coeffs = D.reshape(-1)
    if model == "fisheye":
        angles = np.linspace(0, max_angle, 512)
        derivative = np.ones_like(angles)
        for degree, value in enumerate(coeffs, 1):
            derivative += (2 * degree + 1) * value * angles ** (2 * degree)
    else:
        radii = np.linspace(0, max_radius, 512)
        derivative = 1 + 3 * coeffs[0] * radii ** 2 + 5 * coeffs[1] * radii ** 4 + 7 * coeffs[4] * radii ** 6
    if np.any(derivative <= 0):
        raise ValueError("畸变模型在已观测范围内发生折叠，参数不可靠；请补充边缘及不同倾角样本。")

    warnings = []
    if not (0 <= K[0, 2] < width and 0 <= K[1, 2] < height):
        warnings.append("主点位于画面外，请核对板型、镜头模型和画面裁切。")
    if not 0.1 * min(width, height) <= min(K[0, 0], K[1, 1]) <= 10 * max(width, height):
        warnings.append("焦距偏离常见范围，建议补充近远距离和明显倾斜的样本。")
    if max(K[0, 0], K[1, 1]) / min(K[0, 0], K[1, 1]) > 1.5:
        warnings.append("水平和垂直焦距差异较大，请确认画面未被拉伸、方格确为正方形。")
    if np.max(np.abs(coeffs)) > (20 if model == "fisheye" else 100):
        warnings.append("畸变系数较大，可能存在过拟合；请增加视野覆盖后重新计算。")
    combined = np.concatenate(all_squared)
    rms = float(np.sqrt(np.mean(combined)))
    if rms > 1.0:
        warnings.append("总体重投影 RMS 超过 1 像素，请检查逐张误差、清晰度、板平整度与模型选择。")
    threshold = max(1.0, float(np.median(per_view)) * 2.5)
    outliers = [i + 1 for i, value in enumerate(per_view) if value > threshold]
    if outliers:
        warnings.append("建议检查高误差样本：" + "、".join(map(str, outliers)) + "；程序未自动剔除。")
    warnings.append("当前 RMS 为参与拟合样本的重投影误差，不能替代独立验证或保证机器人定位精度。")
    return {"rms": rms, "per_view_rms": per_view,
            "mean_error": float(np.mean(np.sqrt(combined))),
            "per_view_mean": [float(np.mean(np.sqrt(values))) for values in all_squared],
            "per_view_point_errors": [np.sqrt(values).tolist() for values in all_squared],
            "max_error": float(np.sqrt(combined.max())),
            "p95_error": float(np.sqrt(np.percentile(combined, 95))),
            "reprojected_points": reprojected, "warnings": warnings,
            "max_observed_angle_deg": float(np.degrees(max_angle)),
            "point_count": int(combined.size)}


def calibrate(samples: list[dict], image_size: tuple[int, int], model: str = "pinhole") -> dict:
    """使用全部输入样本拟合，不静默删点/删图；结果可直接 JSON 序列化。"""
    if model not in ("pinhole", "fisheye"):
        raise ValueError("镜头模型请选择针孔或鱼眼。")
    if len(image_size) != 2 or any(not isinstance(v, (int, np.integer)) or v < 16 for v in image_size):
        raise ValueError("画面分辨率必须为有效的宽、高整数。")
    if len(samples) < 8:
        raise ValueError("至少需要 8 个不同姿态；建议按引导完成更充分的视野覆盖。")
    image_size = tuple(int(v) for v in image_size)
    objects, pixels = _samples(samples, image_size)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 120, 1e-8)
    if model == "pinhole":
        try:
            _, K, D, rvecs, tvecs = cv2.calibrateCamera(
                [a.astype(np.float32) for a in objects], [a.astype(np.float32) for a in pixels],
                image_size, None, None, criteria=criteria)
            assessment = _assess(objects, pixels, K, D, rvecs, tvecs, model, image_size)
        except cv2.error as exc:
            raise ValueError("针孔标定未收敛，请增加不同倾角及不同位置的清晰样本。" + str(exc)) from exc
    else:
        width, height = image_size
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
                 | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_CHECK_COND)
        attempt_errors = []
        # 强畸变鱼眼对初值敏感；补充较短焦距候选，仍逐次执行全部有效性检查。
        for focal_scale in (0.5, 0.31, 0.8, 0.25, 0.2):
            initial = np.array([[width * focal_scale, 0, width / 2],
                                [0, width * focal_scale, height / 2], [0, 0, 1.0]])
            try:
                _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    objects, pixels, image_size, initial, np.zeros((4, 1)), flags=flags, criteria=criteria)
                assessment = _assess(objects, pixels, K, D, rvecs, tvecs, model, image_size)
                break
            except (cv2.error, ValueError) as exc:
                attempt_errors.append(f"初始焦距 = 图像宽度 × {focal_scale:g}：{exc}")
        else:
            raise ValueError("鱼眼标定未收敛或参数无效。请避免全为正面/相同姿态，补充边缘、远近和倾角。"
                             "各初值尝试的具体原因（输入编号仅对应该次尝试，不代表样本必然无效）：\n"
                             + "\n".join(attempt_errors))
    return {"model": model, "image_size": list(image_size), "K": K.tolist(),
            "D": D.reshape(-1).tolist(), "rvecs": [np.asarray(v).reshape(3).tolist() for v in rvecs],
            "tvecs": [np.asarray(v).reshape(3).tolist() for v in tvecs],
            "translation_unit": "m", "sample_count": len(samples),
            "error_definition": "sqrt(sum((u-u_projected)^2+(v-v_projected)^2)/N)，单位：像素",
            "validation": {"method": "training_only", "label": "拟合样本重投影；未进行独立留出验证"},
            **assessment}


def _image_size(value) -> tuple[int, int]:
    try:
        valid = (len(value) == 2 and all(isinstance(v, (int, np.integer))
                 and not isinstance(v, (bool, np.bool_)) and v >= 16 for v in value))
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("画面分辨率必须为有效的宽、高整数。")
    return tuple(int(v) for v in value)


def normalize_calibration(result: dict) -> dict:
    """读取本程序导出包或内参字典；不保留历史误差，以免误当作当前帧验证结果。"""
    if not isinstance(result, dict):
        raise ValueError("标定参数必须为 JSON 对象。")
    parameters = result.get("calibration", result)
    if not isinstance(parameters, dict):
        raise ValueError("文件中没有有效的标定参数。")
    model = parameters.get("model")
    if model not in ("pinhole", "fisheye"):
        raise ValueError("验证结果的镜头模型无效，请使用本程序导出的针孔或鱼眼标定结果。")
    size = _image_size(parameters.get("image_size"))
    try:
        K = np.asarray(parameters["K"], np.float64)
        D = np.asarray(parameters["D"], np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("标定参数缺少有效的 K 或 D。") from exc
    if (K.shape != (3, 3) or D.size != (4 if model == "fisheye" else 5)
            or D.ndim not in (1, 2) or (D.ndim == 2 and 1 not in D.shape)):
        raise ValueError("标定内参 K 应为 3×3；鱼眼 D 应为 4 项，针孔 D 应为 5 项。")
    if (not np.isfinite(K).all() or not np.isfinite(D).all()
            or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-9)):
        raise ValueError("标定参数包含无效数值、焦距或内参矩阵末行。")
    if abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9:
        raise ValueError("当前验证仅支持本程序使用的零倾斜内参矩阵。")
    width, height = size
    if (min(K[0, 0], K[1, 1]) < 1e-4 * min(size)
            or max(K[0, 0], K[1, 1]) > 1e3 * max(size)
            or abs(K[0, 2]) > 1e3 * width or abs(K[1, 2]) > 1e3 * height):
        raise ValueError("标定内参的焦距或主点数值异常，请检查参数文件。")
    return {"model": model, "image_size": list(size), "K": K.tolist(), "D": D.reshape(-1).tolist()}


def adapt_calibration_resolution(result: dict, image_size: tuple[int, int], *,
                                 allow_uniform_scale: bool = False) -> dict:
    """仅在确认视场、裁剪不变的等比例缩放时换算 K；D 保持不变。"""
    parameters = normalize_calibration(result)
    target = _image_size(image_size)
    source = tuple(parameters["image_size"])
    if source == target:
        return parameters
    if target[0] * source[1] != target[1] * source[0]:
        raise ValueError("当前分辨率与标定分辨率宽高比不同，不能按等比例缩放换算；请使用原分辨率或重新标定。")
    if allow_uniform_scale is not True:
        raise ValueError("分辨率与标定结果不同；仅在确认视场、裁剪不变且画面只做等比例缩放后才能换算 K。")
    K = np.asarray(parameters["K"], np.float64)
    K[0, :] *= target[0] / source[0]
    K[1, :] *= target[1] / source[1]
    parameters.update(K=K.tolist(), image_size=list(target))
    return parameters


def validate_result(samples: list[dict], result: dict, *, held_out: bool = False) -> dict:
    """固定内参，仅重估每张图的位姿；调用方必须明确这些是否为独立留出样本。"""
    if not samples:
        raise ValueError("请提供至少一张验证样本。")
    result = normalize_calibration(result)
    model = result["model"]
    objects, pixels = _samples(samples, result["image_size"])
    K, D = np.asarray(result["K"], np.float64), np.asarray(result["D"], np.float64)
    rvecs, tvecs = [], []
    for index, (obj, img) in enumerate(zip(objects, pixels)):
        solver = cv2.fisheye.solvePnP if model == "fisheye" else cv2.solvePnP
        try:
            ok, rvec, tvec = solver(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as exc:
            raise ValueError(f"第 {index + 1} 张验证样本无法求解位姿，请调整标定板位置或检查 K、D。") from exc
        if not ok:
            raise ValueError(f"第 {index + 1} 张验证样本无法求解位姿。")
        rvecs.append(rvec)
        tvecs.append(tvec)
    assessed = _assess(objects, pixels, K, D, rvecs, tvecs, model, result["image_size"])
    assessed["warnings"] = [v for v in assessed["warnings"] if not v.startswith("当前 RMS")]
    assessed["method"] = "held_out" if held_out else "fixed_intrinsics_reprojection"
    assessed["label"] = ("独立留出样本：固定内参重投影" if held_out
                         else "固定内参重投影核验；未确认样本独立性，不称为留出验证")
    assessed["rvecs"] = [np.asarray(v).reshape(3).tolist() for v in rvecs]
    assessed["tvecs"] = [np.asarray(v).reshape(3).tolist() for v in tvecs]
    return assessed


def validate_frame(detection: Detection, result: dict, image_size: tuple[int, int], *,
                   allow_uniform_scale: bool = False) -> dict:
    """固定已有 K、D，仅估计本帧板位姿，返回原始画面像素误差与连线端点。"""
    if detection is None:
        raise ValueError("未识别到足够标定板角点，当前帧无法验证。")
    parameters = adapt_calibration_resolution(result, image_size,
                                               allow_uniform_scale=allow_uniform_scale)
    sample = {"object_points": detection.object_points, "image_points": detection.image_points,
              "image_size": image_size}
    assessment = validate_result([sample], parameters)
    projected = np.asarray(assessment["reprojected_points"][0], np.float64)
    pixels = np.asarray(detection.image_points, np.float64).reshape(-1, 2)
    return {"mean_px": assessment["mean_error"], "rms_px": assessment["rms"],
            "max_px": assessment["max_error"], "point_count": assessment["point_count"],
            "projected_points": projected.tolist(), "observed_points": pixels.tolist(),
            "residual_vectors": (projected - pixels).tolist(),
            "point_errors": assessment["per_view_point_errors"][0],
            "rvec": assessment["rvecs"][0], "tvec": assessment["tvecs"][0],
            "warnings": assessment["warnings"], "method": assessment["method"],
            "image_size": parameters["image_size"], "model": parameters["model"]}
