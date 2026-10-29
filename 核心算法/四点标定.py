"""四点针尖标定：四种法兰姿态下，让同一针尖接触同一固定点。

输入为法兰在基坐标系中的位姿，平移单位为毫米。工具坐标轴沿用
法兰坐标轴，因此输出变换的旋转为单位阵；本算法只标定针尖偏移。
"""

import math

import numpy as np


_MODES = ("rotation_vector_rad", "rpy_deg")
_MAX_CONDITION_NUMBER = 1000.0
DEFAULT_CRITERIA = {
    "min_angle_deg": 10.0,
    "max_condition_number": 30.0,
    "max_rms_mm": 0.05,
    "max_error_mm": 0.10,
}


def validate_criteria(criteria=None):
    """返回有效审查阈值；默认值是工程采点/拟合门槛，不是精度保证。"""
    values = DEFAULT_CRITERIA.copy()
    if criteria is not None:
        if not isinstance(criteria, dict):
            raise ValueError("审查标准必须是参数字典。")
        unknown = set(criteria) - set(values)
        if unknown:
            raise ValueError("审查标准包含不支持的参数。")
        values.update(criteria)
    labels = {
        "min_angle_deg": "最小姿态角差",
        "max_condition_number": "最大条件数",
        "max_rms_mm": "最大 RMS 残差",
        "max_error_mm": "最大单点残差",
    }
    for key, value in values.items():
        try:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{labels[key]}必须是有限正数。") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{labels[key]}必须是有限正数。")
        values[key] = value
    if values["min_angle_deg"] >= 180:
        raise ValueError("最小姿态角差必须小于 180°。")
    if values["max_condition_number"] < 1:
        raise ValueError("最大条件数不能小于 1。")
    if values["max_rms_mm"] > values["max_error_mm"]:
        raise ValueError("最大 RMS 残差不能大于最大单点残差。")
    return values


def _check(check_id, label, passed, actual, limit, detail):
    return {
        "id": check_id,
        "label": label,
        "passed": bool(passed),
        "actual": actual,
        "limit": limit,
        "detail": detail,
    }


def _finite_array(values, shape, description):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{description}必须全部为有效数字。") from exc
    if array.shape != shape:
        if shape == (4, 6):
            raise ValueError("四点标定需要恰好 4 组法兰位姿，每组包含 X、Y、Z 和 3 个姿态值。")
        raise ValueError("姿态需要恰好 3 个数值。")
    if not np.isfinite(array).all():
        raise ValueError(f"{description}不能包含空值、NaN 或无穷大。")
    return array


def rotation_matrix(values, mode):
    """将 3 个姿态值转换为 3×3 numpy 旋转矩阵。

    mode='rotation_vector_rad'：旋转向量，模长为转角（弧度）。
    mode='rpy_deg'：固定轴 X/Y/Z 的滚转、俯仰、偏航角（度），
    使用 Rz(yaw) @ Ry(pitch) @ Rx(roll)。
    """
    if mode not in _MODES:
        raise ValueError("姿态格式无效，请选择旋转向量（弧度）或 RPY 欧拉角（度）。")
    values = _finite_array(values, (3,), "姿态值")
    if mode == "rpy_deg":
        roll, pitch, yaw = np.deg2rad(values)
        sr, cr = math.sin(roll), math.cos(roll)
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

    angle = math.hypot(*values)
    if not math.isfinite(angle):
        raise ValueError("旋转向量数值过大，请检查姿态值及单位。")
    if angle == 0.0:
        return np.eye(3)
    axis = values / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    # 2 sin²(θ/2) 在接近零时比 1-cos(θ) 更稳定。
    return np.eye(3) + math.sin(angle) * skew + 2.0 * math.sin(angle / 2.0) ** 2 * (skew @ skew)


def review_samples(samples, mode, criteria=None):
    """在求解前审查四点输入、姿态角差、可观测性和条件数。

    此函数仅分析输入和 12×6 系数矩阵，不求针尖偏移。点编号从 1 开始；
    条件数在秩不足或计算失败时为 None，使结果可以严格序列化为 JSON。
    姿态角差是 SO(3) 上的相对转角，不是欧拉角逐分量相减。
    """
    limits = validate_criteria(criteria)
    review = {
        "passed": False,
        "checks": [],
        "criteria": limits,
        "min_angle_deg": None,
        "closest_pair": None,
        "angle_pairs": [],
        "rank": None,
        "condition_number": None,
    }
    checks = review["checks"]
    try:
        poses = _finite_array(samples, (4, 6), "法兰位姿")
        rotations = np.array([rotation_matrix(row[3:], mode) for row in poses])
    except ValueError as exc:
        checks.append(_check("input", "四点输入有效", False, None, "4×6 个有限数值", str(exc)))
        return review
    checks.append(_check("input", "四点输入有效", True, "4×6 个有限数值", "4×6 个有限数值",
                         "已录入 4 组有效法兰位姿；仍需确认四次针尖接触同一固定点，且位置单位为毫米。"))

    for i in range(4):
        for j in range(i + 1, 4):
            relative = rotations[i].T @ rotations[j]
            # atan2 在接近 0° 和 180° 时比仅用 acos(trace) 更稳定。
            sine = 0.5 * math.hypot(relative[2, 1] - relative[1, 2],
                                    relative[0, 2] - relative[2, 0],
                                    relative[1, 0] - relative[0, 1])
            cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            angle = math.degrees(math.atan2(sine, cosine))
            review["angle_pairs"].append({"points": [i + 1, j + 1], "angle_deg": angle})
    closest = min(review["angle_pairs"], key=lambda pair: pair["angle_deg"])
    min_angle = closest["angle_deg"]
    first, second = closest["points"]
    review["min_angle_deg"] = min_angle
    review["closest_pair"] = closest["points"]
    # 仅容忍浮点舍入误差，防止恰好处于阈值的姿态被错误拒绝。
    angle_passed = min_angle >= limits["min_angle_deg"] - 1e-9
    angle_detail = f"最接近的是第 {first}、{second} 点，角差 {min_angle:.3f}°。"
    if not angle_passed:
        angle_detail += (f"优先重采第 {second} 点，使它与其他每一点均相差至少 "
                         f"{limits['min_angle_deg']:g}°；保持针尖接触同一固定点，并改变不同轴的倾角。")
    else:
        angle_detail += "全部两两姿态角差满足采点门槛。"
    checks.append(_check("angle", "两两姿态角差", angle_passed, min_angle,
                         limits["min_angle_deg"], angle_detail))

    matrix = np.concatenate([rotations.reshape(12, 3), -np.tile(np.eye(3), (4, 1))], axis=1)
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        checks.append(_check("rank", "针尖位置可唯一确定", False, None, 6,
                             "姿态矩阵分析失败，请核对姿态格式和数值后重新采集。"))
        checks.append(_check("condition", "姿态分布稳定性", False, None,
                             limits["max_condition_number"], "矩阵分析失败，未通过条件数审查。"))
        return review
    tolerance = singular_values[0] * max(matrix.shape) * np.finfo(matrix.dtype).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    review["rank"] = rank
    rank_passed = rank == 6
    rank_detail = "12×6 矩阵满秩，针尖三个方向的偏移均可求解。"
    if not rank_passed:
        rank_detail = (f"矩阵秩为 {rank}，需要 6。不能只绕同一轴转动；"
                       "请重新选择第 2～4 点姿态，包含至少两个不同轴的倾斜，针尖保持接触同一固定点。")
    checks.append(_check("rank", "针尖位置可唯一确定", rank_passed, rank, 6, rank_detail))

    condition = float(singular_values[0] / singular_values[-1]) if rank_passed else None
    review["condition_number"] = condition
    condition_passed = condition is not None and condition <= limits["max_condition_number"]
    condition_detail = "姿态分布满足当前条件数门槛。"
    if not condition_passed:
        value = f"{condition:.3f}" if condition is not None else "不可求（矩阵不满秩）"
        condition_detail = (f"当前条件数 {value}。请增加第 2～4 点在不同轴上的倾角跨度；"
                            "避免四点集中在相似姿态或仅围绕同一轴转动。")
    checks.append(_check("condition", "姿态分布稳定性", condition_passed, condition,
                         limits["max_condition_number"], condition_detail))
    review["passed"] = all(check["passed"] for check in checks)
    return review


def review_result(result, criteria=None):
    """审查候选结果的拟合残差；通过仅表示本次采点/拟合合格。

    不代表独立验证的绝对定位精度。最大残差点只作为复查线索，
    不能由拟合残差断言该点一定是错误来源。
    """
    limits = validate_criteria(criteria)
    review = {"passed": False, "checks": [], "criteria": limits, "worst_point": None}
    checks = review["checks"]
    try:
        residuals = np.asarray(result["residuals_mm"], dtype=float)
        rms = float(result["rms_mm"])
        max_error = float(result["max_error_mm"])
        valid = (residuals.shape == (4,) and np.isfinite(residuals).all()
                 and (residuals >= 0).all() and math.isfinite(rms) and rms >= 0
                 and math.isfinite(max_error) and max_error >= 0)
    except (KeyError, TypeError, ValueError, OverflowError):
        valid = False
    checks.append(_check("result_finite", "候选结果有效", valid, "有效" if valid else "无效",
                         "4 个有限非负残差及有效统计值",
                         "候选结果可进行残差审查。" if valid else "候选结果缺失或包含无效残差，请重新计算。"))
    if not valid:
        return review
    worst_point = int(np.argmax(residuals)) + 1
    review["worst_point"] = worst_point
    rms_passed = rms <= limits["max_rms_mm"]
    max_passed = max_error <= limits["max_error_mm"]
    rms_detail = "四次触点整体拟合残差满足当前门槛。"
    if not rms_passed:
        rms_detail = ("整体触点一致性不足。请检查四次是否接触同一固定点、参考点是否移动、"
                      "针尖是否受压弯曲，以及位姿格式和毫米单位是否正确，然后重新采点。")
    checks.append(_check("rms", "RMS 残差", rms_passed, rms, limits["max_rms_mm"], rms_detail))
    max_detail = f"最大残差来自第 {worst_point} 点，为 {max_error:.4f} mm。"
    if not max_passed:
        max_detail += (f"优先复查并重新采集第 {worst_point} 点，同时检查其他点；"
                       "最大残差点不一定是唯一误差来源。")
    else:
        max_detail += "满足当前单点残差门槛。"
    checks.append(_check("max_error", "最大单点残差", max_passed, max_error,
                         limits["max_error_mm"], max_detail))
    review["passed"] = all(check["passed"] for check in checks)
    return review


def solve_four_point(samples, mode):
    """解 R_i p + t_i = q，返回可直接序列化为 JSON 的标定结果。

    samples：恰好 4×6 个有限数值，每行为 [x, y, z, rx, ry, rz]。
    位姿必须是法兰相对基坐标系的位姿，不能是已有 TCP 的位姿。
    mode：与 rotation_matrix 的姿态格式相同。

    返回值中的 residuals_mm 为 4 个触点到共同参考点的距离；
    rms_mm 是这些三维距离的均方根。条件数大于 1000 时拒绝计算，
    防止法兰姿态过于接近或只有单轴转动导致结果严重不稳定。
    残差反映四次触点的一致性，不代表独立验证的绝对定位精度。
    """
    poses = _finite_array(samples, (4, 6), "法兰位姿")
    rotations = np.array([rotation_matrix(row[3:], mode) for row in poses])
    translations = poses[:, :3]
    matrix = np.concatenate([
        rotations.reshape(12, 3),
        -np.tile(np.eye(3), (4, 1)),
    ], axis=1)
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("姿态求解失败，请检查输入并重新采集四组法兰位姿。") from exc
    if singular_values[-1] <= singular_values[0] * 1e-10:
        raise ValueError("四组姿态无法唯一确定针尖位置。请改变至少两个方向的倾角，保持针尖接触同一固定点，避免姿态相同或只绕同一轴旋转。")
    condition_number = float(singular_values[0] / singular_values[-1])
    if condition_number > _MAX_CONDITION_NUMBER:
        raise ValueError("四组姿态接近退化，结果容易被测量误差放大。请增加不同方向的倾角差异，重新采集。")

    # 以首个法兰位置平移坐标原点，避免基坐标数值较大时影响求解精度。
    origin = translations[0].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        rhs = -(translations - origin).reshape(12)
    if not np.isfinite(rhs).all():
        raise ValueError("位置数值过大，请检查输入及毫米单位。")
    try:
        solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("针尖位置求解失败，请检查输入并重新采集。") from exc
    tcp_offset = solution[:3]
    reference_local = solution[3:]
    with np.errstate(over="ignore", invalid="ignore"):
        reference_point = reference_local + origin
        vectors = np.einsum("nij,j->ni", rotations, tcp_offset) + (translations - origin) - reference_local
        residuals = np.array([math.hypot(*row) for row in vectors])
        rms = math.hypot(*residuals) / 2.0
    if not (np.isfinite(tcp_offset).all() and np.isfinite(reference_point).all()
            and np.isfinite(residuals).all() and math.isfinite(rms)):
        raise ValueError("求解结果超出有效数值范围，请检查法兰位置和单位。")

    flange_to_tcp = np.eye(4)
    flange_to_tcp[:3, 3] = tcp_offset
    return {
        "tcp_offset_mm": tcp_offset.tolist(),
        "reference_point_base_mm": reference_point.tolist(),
        "residuals_mm": residuals.tolist(),
        "rms_mm": float(rms),
        "max_error_mm": float(residuals.max()),
        "condition_number": condition_number,
        "flange_to_tcp": flange_to_tcp.tolist(),
    }
