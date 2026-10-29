"""手眼标定的图像准入与采集方向建议；只分析内存数据，不控制设备。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from 手眼算法 import BoardSpec, _motion_diagnostics, validate_transform


REGION_NAMES = ("左上", "上方", "右上", "左侧", "中央", "右侧", "左下", "下方", "右下")


def _image_geometry(measurement):
    size = np.asarray(measurement["image_size"], dtype=float)
    points = np.asarray(measurement["corners"], dtype=float).reshape(-1, 2)
    if (size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0)
            or len(points) < 3 or not np.isfinite(points).all()):
        raise ValueError("图像尺寸或角点坐标无效。")
    hull = cv2.convexHull(points.astype(np.float32))
    moments = cv2.moments(hull)
    area = float(moments["m00"])
    if not math.isfinite(area) or area <= 0:
        raise ValueError("角点没有形成可用区域，请完整展示标定板。")
    center = np.array([moments["m10"], moments["m01"]]) / area / size
    cell_xy = np.clip((center * 3).astype(int), 0, 2)
    return points, size, center, area / float(np.prod(size)), int(cell_xy[1] * 3 + cell_xy[0])


def assess_image(measurement, board_dict):
    """检查一帧已识别图像；左右上下始终指原始相机画面。"""
    result = {"accepted": False, "message": "", "region": "", "center": None, "area_ratio": 0.0}
    try:
        spec = BoardSpec(**board_dict)
        spec.validate()
        points, size, center, area, cell = _image_geometry(measurement)
        result.update(region=REGION_NAMES[cell], center=np.clip(center, 0, 1).tolist(), area_ratio=area)
        transform = validate_transform(measurement["T_cam_board"], "板到相机位姿")
        rms = float(measurement["reprojection_rms_px"])
        if not math.isfinite(rms) or rms < 0:
            raise ValueError("角点重投影误差无效。")
        width, height = np.asarray(spec.size_mm) / 1000
        vertices = np.array([[0, 0, 0], [width, 0, 0], [0, height, 0], [width, height, 0]])
        if np.any((vertices @ transform[:3, :3].T + transform[:3, 3])[:, 2] <= 0):
            result["message"] = "板位姿不在相机前方，本次不采集；请检查标定板参数、内参与识别结果。"
        elif rms > 2.0:
            result["message"] = f"角点重投影误差 {rms:.2f} px，需不超过 2 px；请停稳、改善光照和清晰度。"
        elif len(points) < max(12, math.ceil(spec.expected_count * 0.25)):
            required = max(12, math.ceil(spec.expected_count * 0.25))
            result["message"] = f"只识别到 {len(points)} 个角点，至少需 {required} 个；请让更多标定板区域入镜并避开遮挡。"
        else:
            low, high = points.min(axis=0), points.max(axis=0)
            margin = size * 0.01
            if np.any(low < margin) or np.any(high > size - margin):
                directions = []
                if low[0] < margin:
                    directions.append("右")
                if high[0] > size[0] - margin[0]:
                    directions.append("左")
                if low[1] < margin[1]:
                    directions.append("下")
                if high[1] > size[1] - margin[1]:
                    directions.append("上")
                opposite = (("左" in directions and "右" in directions)
                            or ("上" in directions and "下" in directions))
                adjustment = ("让相机稍远离标定板，使整板入镜" if opposite else
                              "让画面中的标定板向" + "、".join(directions) + "侧回到画面内")
                result["message"] = (f"角点离画面边缘不足 1%；{adjustment}，保留边距。"
                                     "只调整机械臂上的相机，标定板保持固定。")
            elif area < 0.01:
                result["message"] = ("角点区域不足画面的 1%；让相机适当靠近标定板，保持整板清晰入镜。"
                                     "只调整机械臂，标定板保持固定。")
            else:
                result.update(accepted=True, message=(f"已识别 {len(points)} 个角点，重投影 RMS {rms:.2f} px；"
                                                     f"标定板位于画面{REGION_NAMES[cell]}，图像检查通过。"))
    except (KeyError, TypeError, ValueError, OverflowError, cv2.error) as exc:
        result["message"] = f"图像检查未通过：{exc}"
    return result


def capture_guidance(samples, target_count=15):
    """分析标定样本的充分性及下一方向；区域和远近仅作建议。

    ready 只代表数量与多轴运动达到自动求解条件，不代表结果已验收。
    target 是标定板中心在原始相机画面中的目标，不是机械臂移动向量。
    """
    if (isinstance(target_count, bool) or not isinstance(target_count, (int, np.integer))
            or target_count < 5):
        raise ValueError("自动标定目标组数必须是至少为 5 的整数。")
    regions = dict.fromkeys(REGION_NAMES, 0)
    pairs, depths, invalid = [], [], 0
    for sample in samples:
        if not isinstance(sample, dict):
            invalid += 1
            continue
        if sample.get("role", "calibration") != "calibration":
            continue
        try:
            robot = validate_transform(sample["T_base_end"], "机器人位姿")
            board = validate_transform(sample["T_cam_board"], "标定板位姿")
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        pairs.append((robot, board))
        if board[2, 3] > 0:
            depths.append(float(board[2, 3]))
        try:
            _, _, _, _, cell = _image_geometry(sample)
            regions[REGION_NAMES[cell]] += 1
        except (KeyError, TypeError, ValueError, OverflowError, cv2.error):
            # 旧配对数据可能不包含角点，仍可利用真实位姿检查多轴运动。
            pass

    count = len(pairs)
    diagnostics = {"sample_count": count, "target_count": int(target_count), "invalid_samples": invalid,
                   "rotation_axis_ratio": 0.0, "max_relative_rotation_deg": 0.0,
                   "meaningful_rotation_pairs": 0, "translation_span_mm": [0.0, 0.0, 0.0],
                   "coverage_regions": sum(value > 0 for value in regions.values()),
                   "depth_span_mm": (max(depths) - min(depths)) * 1000 if depths else 0.0}
    vectors = []
    for first in range(count):
        for second in range(first + 1, count):
            relative = pairs[second][0][:3, :3] @ pairs[first][0][:3, :3].T
            vectors.append(cv2.Rodrigues(relative)[0].reshape(3))
    suggested_axis = "X"
    if vectors:
        vectors = np.asarray(vectors)
        magnitudes = np.linalg.norm(vectors, axis=1)
        meaningful = vectors[magnitudes >= np.radians(5)]
        diagnostics["max_relative_rotation_deg"] = float(np.degrees(magnitudes.max()))
        diagnostics["meaningful_rotation_pairs"] = len(meaningful)
        if len(meaningful):
            axes = meaningful / np.linalg.norm(meaningful, axis=1)[:, None]
            _, _, directions = np.linalg.svd(axes, full_matrices=False)
            suggested_axis = "XYZ"[int(np.argmin(np.abs(directions[0])))]
        try:
            diagnostics.update(_motion_diagnostics(pairs))
        except ValueError as exc:
            diagnostics["motion_message"] = str(exc)
    diagnostics["suggested_base_axis"] = suggested_axis
    axes_ready = (diagnostics["rotation_axis_ratio"] >= 0.25
                  and diagnostics["max_relative_rotation_deg"] >= 20.0)
    ready = count >= target_count and axes_ready and invalid == 0
    summary = (f"已采 {count}/{target_count} 组 · 画面覆盖 {diagnostics['coverage_regions']}/9 区 · "
               f"多轴比 {diagnostics['rotation_axis_ratio']:.2f}/0.25 · "
               f"最大转角 {diagnostics['max_relative_rotation_deg']:.1f}°/20°")
    target = None
    if invalid:
        instruction = f"有 {invalid} 组位姿无效，请检查或移除对应样本后继续；暂不自动标定。"
    elif count == 0:
        instruction = "先让标定板完整出现在画面中央，手动调整机械臂后停稳，识别合格将自动采集；标定板保持固定。"
        target = [0.5, 0.5]
    elif not axes_ready:
        instruction = (f"下一组请绕机械臂基座坐标系 {suggested_axis} 轴改变倾角约 15°～25°，"
                       "补充不同转轴的姿态，并配合平移保持标定板完整入镜后停稳。"
                       "基座轴方向不是画面左右；标定板保持固定。")
    elif ready:
        instruction = "数量与多轴姿态已达到自动标定条件，正在准备计算；结果仍需查看固定板一致性误差。"
    elif diagnostics["coverage_regions"] < 9:
        order = (4, 0, 2, 6, 8, 1, 3, 5, 7)
        cell = next(index for index in order if regions[REGION_NAMES[index]] == 0)
        target = [((cell % 3) + 0.5) / 3, ((cell // 3) + 0.5) / 3]
        instruction = (f"下一组采画面{REGION_NAMES[cell]}：让画面中的标定板中心移向目标区，完整入镜后停稳。"
                       "只调整机械臂，标定板固定；这是画面目标，不是机械臂的移动方向。")
    else:
        instruction = ("下一组改变远近：让相机适当靠近或远离标定板，保持整板入镜，并换一个倾角后停稳。"
                       "只调整机械臂，标定板保持固定。")
    return {"ready": ready, "summary": summary, "instruction": instruction,
            "target": target, "regions": regions, "diagnostics": diagnostics}
