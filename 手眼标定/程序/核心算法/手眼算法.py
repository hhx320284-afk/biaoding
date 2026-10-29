"""Eye-in-hand：相机 → 末端固定变换；所有内部平移均使用米。

约定 T_a_b 将 b 坐标转换到 a 坐标：T_base_end @ T_end_cam @ T_cam_board。
相机模型、标定板检测与 PnP 直接复用相邻相机标定项目，无磁盘写入。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np


_CAMERA_PROGRAM = Path(__file__).resolve().parents[3] / "相机标定" / "程序"
if str(_CAMERA_PROGRAM) not in sys.path:
    sys.path.insert(0, str(_CAMERA_PROGRAM))
from 核心算法.标定板 import BoardSpec
from 核心算法.标定算法 import detect_board, normalize_calibration, validate_frame


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def validate_transform(value, name: str = "位姿") -> np.ndarray:
    """校验有限、右手、正交的 4×4 刚体变换，不静默修正无效输入。"""
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是数值 4×4 矩阵。") from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name}必须是有限数值组成的 4×4 矩阵。")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError(f"{name}的最后一行必须为 0、0、0、1。")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-4)
            or abs(np.linalg.det(rotation) - 1) > 1e-4):
        raise ValueError(f"{name}的旋转部分不是有效的右手正交矩阵；请检查格式、单位和精度。")
    return transform.copy()


def load_camera_record(raw: dict) -> dict:
    """读取相机标定导出包；保留实际板参数，禁止默认为另一块板。"""
    calibration = normalize_calibration(raw)
    board = raw.get("board")
    if not isinstance(board, dict):
        raise ValueError("内参文件缺少 board 标定板参数，请导入“相机标定”导出的完整结果 JSON。")
    try:
        spec = BoardSpec(**board)
        spec.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"标定板参数无效：{exc}") from exc
    warnings = []
    if spec.kind == "chessboard":
        warnings.append("普通棋盘格可能存在 180° 角点排序歧义；须始终保持板坐标原点与朝向一致。"
                        "手眼标定优先使用带唯一标签的 ChArUco、ArUco 或 AprilGrid。")
    return {"calibration": calibration, "board": spec.to_dict(),
            "camera_name": str(raw.get("camera_name", "")), "warnings": warnings}


def estimate_board(frame: np.ndarray, record: dict) -> dict:
    """从原始未去畸变图估计 T_cam_board；画面尺寸必须匹配原内参。"""
    record = load_camera_record(record)
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
        raise ValueError("请提供有效的原始相机图像。")
    size = (int(frame.shape[1]), int(frame.shape[0]))
    expected = tuple(record["calibration"]["image_size"])
    if size != expected:
        raise ValueError(f"图像分辨率 {size[0]}×{size[1]} 与内参分辨率 "
                         f"{expected[0]}×{expected[1]} 不一致；请使用原始分辨率和相同视场。")
    detection = detect_board(frame, BoardSpec(**record["board"]))
    if detection is None:
        raise ValueError("未识别到足够标定板角点；请检查板参数、清晰度、遮挡和画面覆盖。")
    assessed = validate_frame(detection, record["calibration"], size)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(assessed["rvec"], np.float64))[0]
    transform[:3, 3] = assessed["tvec"]
    validate_transform(transform, "板到相机位姿")
    return {"T_cam_board": transform.tolist(),
            "reprojection_rms_px": float(assessed["rms_px"]),
            "corners": detection.image_points.tolist(), "image_size": list(size),
            "point_count": int(detection.found_count),
            "warnings": record["warnings"] + assessed["warnings"]}


def parse_pose(text: str, representation: str = "matrix", translation_unit: str = "mm",
               angle_unit: str = "deg") -> np.ndarray:
    """解析末端 → 基座位姿。

    matrix: 逐行 16 项；rotvec: x y z rx ry rz（旋转向量）；
    rpy: x y z roll pitch yaw，固定轴 Rz(yaw) @ Ry(pitch) @ Rx(roll)；
    quaternion: x y z qx qy qz qw（单位四元数，顺序不可交换）。
    平移单位 m/mm，角度 rad/deg。返回平移为米的 numpy 4×4 矩阵。
    """
    if representation not in ("matrix", "rotvec", "rpy", "quaternion"):
        raise ValueError("位姿格式请选择 matrix、rotvec、rpy 或 quaternion。")
    if translation_unit not in ("m", "mm") or angle_unit not in ("rad", "deg"):
        raise ValueError("平移单位必须为 m/mm，角度单位必须为 rad/deg。")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入当前末端到基座的位姿。")
    parts = [part for part in re.split(r"[\s,，;；\[\]\(\)]+", text.strip()) if part]
    try:
        values = np.array([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("位姿只能包含数值，用空格、逗号或换行分隔；不要包含字段名。") from exc
    expected = {"matrix": 16, "rotvec": 6, "rpy": 6, "quaternion": 7}[representation]
    if values.size != expected or not np.isfinite(values).all():
        raise ValueError(f"当前格式要求 {expected} 个有限数值，实际读取 {values.size} 个。")
    scale = 0.001 if translation_unit == "mm" else 1.0
    if representation == "matrix":
        transform = values.reshape(4, 4).copy()
        transform[:3, 3] *= scale
        return validate_transform(transform, "机器人位姿")
    transform = np.eye(4)
    transform[:3, 3] = values[:3] * scale
    if representation == "quaternion":
        quaternion = values[3:]
        norm = float(np.linalg.norm(quaternion))
        if abs(norm - 1) > 0.02:
            raise ValueError("四元数必须接近单位长度；顺序为 qx qy qz qw，请核对机器人接口。")
        x, y, z, w = quaternion / norm
        transform[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    else:
        angles = values[3:] * (np.pi / 180 if angle_unit == "deg" else 1.0)
        if representation == "rotvec":
            transform[:3, :3] = cv2.Rodrigues(angles)[0]
        else:
            roll, pitch, yaw = angles
            cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
            transform[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                 [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                                 [-sp, cp*sr, cp*cr]]
    return validate_transform(transform, "机器人位姿")


def _sample_matrices(samples: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("请先添加配对样本。")
    result = []
    for index, sample in enumerate(samples, 1):
        if not isinstance(sample, dict):
            raise ValueError(f"第 {index} 个样本格式错误。")
        if sample.get("role", "calibration") not in ("calibration", "validation"):
            raise ValueError(f"第 {index} 个样本用途必须为标定或验证。")
        try:
            robot = validate_transform(sample["T_base_end"], f"第 {index} 个机器人位姿")
            board = validate_transform(sample["T_cam_board"], f"第 {index} 个标定板位姿")
        except KeyError as exc:
            raise ValueError(f"第 {index} 个样本缺少配对的机器人位姿或板位姿。") from exc
        result.append((robot, board))
    return result


def _mean_pose(transforms: list[np.ndarray]) -> np.ndarray:
    mean = np.eye(4)
    u, _, vt = np.linalg.svd(np.sum([value[:3, :3] for value in transforms], axis=0))
    mean[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    mean[:3, 3] = np.mean([value[:3, 3] for value in transforms], axis=0)
    return mean


def _errors(transforms: list[np.ndarray], reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.array([np.linalg.norm(value[:3, 3] - reference[:3, 3]) * 1000 for value in transforms])
    rotation = np.array([np.degrees(np.arccos(np.clip((np.trace(reference[:3, :3].T @ value[:3, :3])-1)/2,
                                                   -1.0, 1.0))) for value in transforms])
    return position, rotation


def _metrics(position: np.ndarray, rotation: np.ndarray) -> dict:
    return {"translation_rms_mm": float(np.sqrt(np.mean(position**2))),
            "translation_max_mm": float(np.max(position)),
            "rotation_rms_deg": float(np.sqrt(np.mean(rotation**2))),
            "rotation_max_deg": float(np.max(rotation))}


def evaluate_samples(samples: list[dict], T_end_cam, reference_board=None) -> dict:
    """固定板一致性。传入训练组 reference_board 时，验证偏移不会被自身均值抵消。"""
    pairs = _sample_matrices(samples)
    handeye = validate_transform(T_end_cam, "相机到末端变换")
    transforms = [robot @ handeye @ board for robot, board in pairs]
    mean = _mean_pose(transforms)
    reference = mean if reference_board is None else validate_transform(reference_board, "固定板参考位姿")
    position, rotation = _errors(transforms, reference)
    spread_position, spread_rotation = _errors(transforms, mean)
    return {**_metrics(position, rotation), "sample_count": len(samples),
            "reference_board": reference.tolist(), "mean_board": mean.tolist(),
            "reference_mode": "sample_mean" if reference_board is None else "fixed_reference",
            "spread": _metrics(spread_position, spread_rotation),
            "per_sample": [{"id": sample.get("id", index+1),
                            "translation_error_mm": float(position[index]),
                            "rotation_error_deg": float(rotation[index]),
                            "T_base_board": transforms[index].tolist()}
                           for index, sample in enumerate(samples)]}


def _motion_diagnostics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    rotations = []
    for first in range(len(pairs)):
        for second in range(first + 1, len(pairs)):
            # 所有相对旋转轴均表达在基座坐标系，避免混用不同末端坐标系。
            relative = pairs[second][0][:3, :3] @ pairs[first][0][:3, :3].T
            vector = cv2.Rodrigues(relative)[0].reshape(3)
            rotations.append(vector)
    magnitudes = np.linalg.norm(rotations, axis=1)
    meaningful = np.asarray(rotations)[magnitudes >= np.radians(5)]
    if len(meaningful) < 2:
        raise ValueError("机器人旋转变化不足：至少采集绕两个不同轴明显旋转的姿态，不能只平移或重复拍摄。")
    axes = meaningful / np.linalg.norm(meaningful, axis=1)[:, None]
    singular = np.linalg.svd(axes, compute_uv=False)
    ratio = float(singular[1] / singular[0])
    if ratio < 0.10:
        raise ValueError("机器人姿态接近单轴旋转，手眼变换无法可靠约束；请补充绕另一个轴的明显旋转。")
    translations = np.array([pair[0][:3, 3] for pair in pairs])
    return {"rotation_axis_ratio": ratio,
            "max_relative_rotation_deg": float(np.degrees(magnitudes.max())),
            "meaningful_rotation_pairs": len(meaningful),
            "translation_span_mm": ((translations.max(axis=0) - translations.min(axis=0)) * 1000).tolist()}


def solve_handeye(samples: list[dict], method: str = "PARK") -> dict:
    """仅使用标定样本计算 T_end_cam，拒绝单轴/纯平移退化数据。"""
    _sample_matrices(samples)
    method = str(method).upper()
    if method not in METHODS:
        raise ValueError("不支持的手眼标定方法。")
    training = [sample for sample in samples if sample.get("role", "calibration") == "calibration"]
    if len(training) < 5:
        raise ValueError("至少需要 5 组标定样本；建议采集 15～25 组有明显多轴旋转、远近和位置变化的样本。")
    pairs = _sample_matrices(training)
    diagnostics = _motion_diagnostics(pairs)
    try:
        rotation, translation = cv2.calibrateHandEye(
            [robot[:3, :3] for robot, _ in pairs],
            [robot[:3, 3].reshape(3, 1) for robot, _ in pairs],
            [board[:3, :3] for _, board in pairs],
            [board[:3, 3].reshape(3, 1) for _, board in pairs], method=METHODS[method])
        handeye = np.eye(4)
        handeye[:3, :3], handeye[:3, 3] = rotation, np.asarray(translation).reshape(3)
        handeye = validate_transform(handeye, "手眼求解结果")
    except (cv2.error, ValueError, TypeError) as exc:
        raise ValueError("手眼求解失败或结果退化，请检查配对、位姿方向、单位和多轴姿态覆盖，"
                         f"或更换求解方法。具体原因：{exc}") from exc
    warnings = []
    if len(training) < 15:
        warnings.append("标定样本少于建议的 15 组；请根据工作范围增加不同位置、距离与多轴姿态。")
    if diagnostics["max_relative_rotation_deg"] < 20 or diagnostics["rotation_axis_ratio"] < 0.25:
        warnings.append("姿态覆盖仍较弱，建议扩大旋转范围并平衡不同旋转轴。")
    return {"T_end_cam": handeye.tolist(), "method": method,
            "translation_unit": "m", "transform_direction": "camera_to_end",
            "calibration_count": len(training), "diagnostics": diagnostics, "warnings": warnings}
