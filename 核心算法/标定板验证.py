"""标定板视觉与探针对点的纯 CPU 几何；读取参数，不保存图像或驱动设备。

T_a_b 将 b 坐标变换到 a 坐标，所有内部长度均为米。对点误差检查整个
视觉、手眼、机器人与探针 TCP 坐标链，不能仅凭一次对点把误差归因到某项标定。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from 目标变换 import load_handeye, transform


# 直接复用相机标定的板几何、检测和镜头模型，不导入其界面或采集控制器。
_CAMERA_PROGRAM = Path(__file__).resolve().parents[3] / "相机标定" / "程序"
if str(_CAMERA_PROGRAM) not in sys.path:
    sys.path.append(str(_CAMERA_PROGRAM))
from 核心算法.标定板 import BoardSpec, KIND_NAMES
from 核心算法.标定算法 import detect_board, normalize_calibration, validate_frame


BOARD_KINDS = {name: kind for kind, name in KIND_NAMES.items()}
DEFAULT_SPEC = BoardSpec("aprilgrid", 6, 6, 24.0, 20.0, 6.0).to_dict()
SYSTEM_SCOPE = "对点误差反映整条坐标链，不能单独归因于 TCP 或手眼标定；像素重投影合格不等于达到 0.1 mm。"


def _spec(value):
    if isinstance(value, BoardSpec):
        result = value
    elif isinstance(value, dict):
        values = dict(value)
        values["kind"] = BOARD_KINDS.get(values.get("kind"), values.get("kind"))
        result = BoardSpec(**values)
    else:
        raise ValueError("请提供实际标定板的类型、行列数和毫米尺寸。")
    result.validate()
    return result


def _serial(payload):
    for item in (payload, payload.get("camera", {}), payload.get("source_intrinsics", {})):
        if not isinstance(item, dict):
            continue
        for key in ("serial_number", "serial", "color_serial"):
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    # 本地相机标定的完整导出以 V4L 的 by-id 路径记录设备，RealSense 为 12 位序列号。
    for value in (payload.get("device", ""), payload.get("camera_name", "")):
        matches = re.findall(r"(?<!\d)(\d{12})(?!\d)", str(value))
        if len(set(matches)) == 1:
            return matches[0]
    return None


def load_intrinsics(path):
    """读取现有相机内参 JSON；保留设备身份及原包，绝不写回。"""
    source = Path(path).expanduser()
    if source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("内参文件过大，请选择相机标定导出的结果 JSON。")
    payload = json.loads(source.read_text(encoding="utf-8"))
    calibration = normalize_calibration(payload)
    return {"calibration": calibration, "normcalibration": calibration,
            "serial": _serial(payload), "payload": payload, "path": str(source)}


def board_geometry(spec):
    """图案矩形范围；尺寸不包含未知的纸张白边或板基材。"""
    spec = _spec(spec)
    width, height = np.asarray(spec.size_mm, dtype=float) / 1000.0
    # 复用 detect_board 的实际约定：棋盘格首内角点是 (square,square)，不是原点。
    note = ("图案外边界左上角为原点；首个内角点坐标为 (方格边长, 方格边长)。"
            if spec.kind == "chessboard" else "图案外边界左上角为原点，X、Y 沿图案两边；不包含纸张白边。")
    minimum, maximum = [0.0, 0.0, 0.0], [float(width), float(height), 0.0]
    return {"bounds_xy_m": [[0.0, 0.0], [float(width), float(height)]],
            "bounds_m": [minimum, maximum], "min": minimum, "max": maximum,
            "size_mm": list(spec.size_mm),
            "corners_m": [[0.0, 0.0, 0.0], [float(width), 0.0, 0.0],
                          [float(width), float(height), 0.0], [0.0, float(height), 0.0]],
            "center_m": [float(width / 2), float(height / 2), 0.0],
            "normal_board": [0.0, 0.0, 1.0], "origin_note": note,
            "scope": "已知尺寸的标定图案范围，不包含纸张白边"}


def _frame_parameters(frame, camera_record):
    if not isinstance(frame, dict):
        raise ValueError("请提供同一相机帧的图像、内参与时间信息。")
    if camera_record is None:
        image = frame.get("rgb")
        if frame.get("image_space", "undistorted_color") != "undistorted_color":
            raise ValueError("默认内参仅用于已经去畸变的彩色快照。")
        D = np.asarray(frame.get("dist_coeffs", np.zeros(5)), dtype=float).reshape(-1)
        if D.size != 5 or not np.isfinite(D).all() or np.any(np.abs(D) > 1e-12):
            raise ValueError("已去畸变快照的 D 必须为零；不能再次应用外部畸变系数。")
        space = "undistorted_color"
    else:
        if not isinstance(camera_record, dict):
            raise ValueError("外部内参必须先通过 load_intrinsics 读取。")
        image = frame.get("raw_rgb")
        if image is None:
            raise ValueError("当前快照没有原始 raw_rgb，不能把外部 D 应用到已去畸变画面。")
        calibration = normalize_calibration(camera_record.get("calibration", camera_record))
        serial = camera_record.get("serial") or _serial(camera_record.get("payload", camera_record))
        current = _serial(frame)
        if not serial or not current:
            raise ValueError("外部内参或当前帧缺少可核对的相机序列号，不能确认属于同一设备。")
        if str(serial) != str(current):
            raise ValueError(f"外部内参相机 {serial} 与当前彩色相机 {current} 不一致。")
        source = frame.get("source_intrinsics")
        if not isinstance(source, dict):
            raise ValueError("原始快照缺少 source_intrinsics，无法核对原始彩色流。")
        if _serial(source) and _serial(source) != current:
            raise ValueError("原始彩色流与当前快照的相机序列号不一致。")
        space = "raw_color"
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
            or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)):
        raise ValueError("标定板检测需要有效的 8 位 RGB 或灰度画面。")
    size = (int(image.shape[1]), int(image.shape[0]))
    if camera_record is None:
        calibration = normalize_calibration({"model": "pinhole", "image_size": list(size),
                                              "K": frame.get("K"), "D": D.tolist()})
    else:
        if tuple(calibration["image_size"]) != size:
            raise ValueError("原始画面与外部内参分辨率不同；此处不自动缩放或裁剪内参。")
        if (source.get("width"), source.get("height")) != size:
            raise ValueError("原始彩色画面与 source_intrinsics 的尺寸不一致。")
    return image, calibration, space


def estimate_board(frame, spec, camera_record=None):
    """从当前实际角点估计板→相机位姿；接受门槛不代表机器人绝对精度。"""
    spec = _spec(spec)
    geometry = board_geometry(spec)
    image, calibration, space = _frame_parameters(frame, camera_record)
    size = (int(image.shape[1]), int(image.shape[0]))
    warnings = [SYSTEM_SCOPE]
    if spec.kind == "chessboard":
        warnings.append("普通棋盘格存在 180° 排序歧义；自定义坐标须核对板原点和朝向，建议使用有唯一 ID 的板。")
    result = {"accepted": False, "accept": False, "reason": "尚未识别到标定板。",
              "T_cam_board": None, "pixels": [], "objects": [], "ids": [],
              "image_rgb": image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
              "normal_camera_outward": None,
              "projected_points": [], "point_count": 0, "reprojection_rms_px": None,
              "rms_px": None, "geometry": geometry, "center_point_m": geometry["center_m"],
              "board": spec.to_dict(), "image_size": list(size), "image_space": space,
              "K": calibration["K"], "D": calibration["D"], "model": calibration["model"],
              "frame_id": frame.get("frame_id"), "monotonic": frame.get("monotonic"),
              "color_serial": _serial(frame), "warnings": warnings,
              "source_intrinsics": frame.get("source_intrinsics"),
              "device_timestamp_ms": frame.get("device_timestamp_ms"),
              "orientation_ambiguous": spec.kind == "chessboard",
              "criteria": {"rms_max_px": 1.0, "min_points": 16,
                           "min_board_span_fraction": 0.25, "min_image_span_px": 24.0,
                           "min_image_area_fraction": 0.002}}
    # 相机快照是 RGB；复用检测器接受灰度/BGR，显式转换避免通道语义混淆。
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    detection = detect_board(gray, spec)
    if detection is None:
        result["reason"] = "未检测到至少 16 个有效、非共线的板角点；请核对板类型、尺寸、字典并改善清晰度或遮挡。"
        return result
    assessed = validate_frame(detection, calibration, size)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(assessed["rvec"], dtype=float))[0]
    T[:3, 3] = assessed["tvec"]
    T = transform(T)
    objects = np.asarray(detection.object_points, dtype=float)
    pixels = np.asarray(detection.image_points, dtype=float)
    board_span = np.ptp(objects[:, :2], axis=0) / (np.asarray(spec.size_mm) / 1000)
    image_span = np.ptp(pixels, axis=0)
    image_area = float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32)))) / (size[0] * size[1])
    depths = (objects @ T[:3, :3].T + T[:3, 3])[:, 2]
    center_camera = T[:3, :3] @ np.asarray(geometry["center_m"]) + T[:3, 3]
    outward = T[:3, 2].copy()
    if float(outward @ center_camera) > 0:
        outward *= -1
    rms = float(assessed["rms_px"])
    problems = []
    if not np.isfinite(rms) or rms > 1.0:
        problems.append(f"角点重投影 RMS {rms:.3f} px 超过 1.000 px")
    if np.any(depths <= 0):
        problems.append("估计的板角点位于相机后方")
    if np.any(board_span < 0.25):
        problems.append("角点分布过于集中，横纵跨度均需覆盖至少 25% 的板图案")
    if np.any(image_span < 24.0) or image_area < 0.002:
        problems.append("标定板在画面中太小或过于侧斜，请靠近并改善观察角度")
    result.update(T_cam_board=T.tolist(), pixels=pixels.tolist(), objects=objects.tolist(),
                  ids=detection.ids.reshape(-1).astype(int).tolist(),
                  projected_points=assessed["projected_points"],
                  point_errors=assessed["point_errors"], point_count=int(detection.found_count),
                  reprojection_rms_px=rms, rms_px=rms, max_px=float(assessed["max_px"]),
                  board_span_fraction=board_span.tolist(), image_area_fraction=image_area,
                  normal_camera_outward=outward.tolist(),
                  accepted=not problems, accept=not problems,
                  reason="；".join(problems) if problems else "当前实际角点重投影与分布检查通过，可选择板上验证点。")
    result["warnings"].extend(assessed["warnings"])
    return result


def _inside(point, geometry):
    point = np.asarray(point, dtype=float)
    bounds = np.asarray(geometry["bounds_xy_m"], dtype=float)
    if (point.shape != (3,) or not np.isfinite(point).all() or abs(point[2]) > 1e-8
            or bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])):
        raise ValueError("板上点或板的有效范围无效。")
    if np.any(point[:2] < bounds[0] - 1e-9) or np.any(point[:2] > bounds[1] + 1e-9):
        raise ValueError("目标点位于已知尺寸的标定图案范围外，不能把纸张白边当作已知板面。")
    return point.copy()


def target_point(result, mode="center", corner_id=None, xy_mm=None):
    """选择板图案中心、实际检测到的角点 ID，或自定义板坐标 XY（毫米）。"""
    if not result.get("accepted") or result.get("T_cam_board") is None:
        raise ValueError("请先取得通过角点审查的当前标定板位姿。")
    mode = {"板图案中心": "center", "检测角点": "corner", "自定义 XY": "custom"}.get(mode, mode)
    if mode == "center":
        point = result["geometry"]["center_m"]
    elif mode == "corner":
        ids = list(result["ids"])
        if corner_id is None or isinstance(corner_id, bool) or corner_id not in ids:
            raise ValueError("请选择本帧实际检测到的角点 ID。")
        point = result["objects"][ids.index(corner_id)]
    elif mode == "custom":
        xy = np.asarray(xy_mm, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("自定义板坐标需要两个有限的毫米数值 X、Y。")
        point = np.r_[xy / 1000.0, 0.0]
    else:
        raise ValueError("目标点类型应为板图案中心、检测角点或自定义 XY。")
    return _inside(point, result["geometry"])


def image_to_board(pixel, K, D, T_cam_board, geometry, model="pinhole"):
    """将同一图像空间的像素射线与板面 Z=0 相交，拒绝板范围外的点。"""
    pixel = np.asarray(pixel, dtype=float)
    K, D = np.asarray(K, dtype=float), np.asarray(D, dtype=float).reshape(-1)
    if (pixel.shape != (2,) or not np.isfinite(pixel).all() or K.shape != (3, 3)
            or not np.isfinite(K).all() or not np.isfinite(D).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or model not in ("pinhole", "fisheye")
            or D.size != (4 if model == "fisheye" else 5)):
        raise ValueError("像素坐标或针孔/鱼眼内参无效。")
    T = transform(T_cam_board)
    undistort = cv2.fisheye.undistortPoints if model == "fisheye" else cv2.undistortPoints
    xy = undistort(pixel.reshape(1, 1, 2), K, D).reshape(2)
    ray_camera = np.r_[xy, 1.0]
    R, t = T[:3, :3], T[:3, 3]
    origin = -R.T @ t
    ray = R.T @ ray_camera
    if not np.isfinite(ray).all() or abs(ray[2]) < 1e-8:
        raise ValueError("选点射线与板面近乎平行，无法可靠反投影。")
    distance = -origin[2] / ray[2]
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError("所选像素与板面的交点位于相机后方。")
    point = origin + distance * ray
    point[2] = 0.0
    return _inside(point, geometry)


def project_board_point(result, p_board):
    """用本次检测的镜头模型将板上点投影回其 image_rgb 画面。"""
    if result.get("T_cam_board") is None:
        raise ValueError("当前没有可投影的标定板位姿。")
    point = _inside(p_board, result["geometry"])
    T = transform(result["T_cam_board"])
    if (T[:3, :3] @ point + T[:3, 3])[2] <= 0:
        raise ValueError("板上点位于相机后方。")
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    K, D = np.asarray(result["K"], dtype=float), np.asarray(result["D"], dtype=float)
    projector = cv2.fisheye.projectPoints if result["model"] == "fisheye" else cv2.projectPoints
    pixel = projector(point.reshape(1, 1, 3), rvec, T[:3, 3], K, D)[0].reshape(2)
    if not np.isfinite(pixel).all():
        raise ValueError("板上点投影无效，请重新检测标定板。")
    return pixel


def base_board_chain(T_base_flange, handeye, T_flange_tcp, T_cam_board=None):
    """以法兰统一手眼参考，当前针尖 TCP 只参与针尖位姿，不再重复乘入相机链。"""
    if isinstance(handeye, (str, Path)):
        handeye = load_handeye(handeye)
    if not isinstance(handeye, dict) or "T_flange_camera" not in handeye:
        raise ValueError("请先用 load_handeye 读取带方向与 TCP 定义的手眼结果。")
    payload = handeye.get("payload")
    if payload is not None:
        if (payload.get("schema") != "handeye_result_v2"
                or payload.get("configuration") != "eye_in_hand"
                or payload.get("translation_unit") != "m"
                or payload.get("transform_direction") != "tcp_to_camera"):
            raise ValueError("手眼结果必须明确 eye_in_hand、米单位及 TCP→相机方向。")
        expected = (transform(payload["tcp_definition"]["T_flange_tcp"])
                    @ np.linalg.inv(transform(payload["result"]["T_cam_tcp"])))
        if not np.allclose(expected, transform(handeye["T_flange_camera"]), rtol=0, atol=1e-8):
            raise ValueError("手眼换算矩阵与文件中的参考 TCP 或方向不一致。")
    base_flange = transform(T_base_flange)
    flange_camera = transform(handeye["T_flange_camera"])
    flange_tcp = transform(T_flange_tcp)
    result = {"T_flange_camera": flange_camera, "T_base_camera": base_flange @ flange_camera,
              "T_base_tcp": base_flange @ flange_tcp, "scope": SYSTEM_SCOPE}
    if T_cam_board is not None:
        result["T_base_board"] = result["T_base_camera"] @ transform(T_cam_board)
    return result
