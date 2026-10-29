"""多姿态针尖标定与逐条审查；只返回建议，不修改或删除原始数据。

自动识别的是姿态冗余及与多数数据不一致的触点。普通刚性针没有接触
信号，拟合通过不能证明针尖确实碰到了标定点，也不是绝对精度认证。
"""

import itertools
import math

import numpy as np

from 四点标定 import rotation_matrix, review_result, validate_criteria


def _angle(first, second):
    relative = first.T @ second
    sine = 0.5 * math.hypot(relative[2, 1] - relative[1, 2],
                            relative[0, 2] - relative[2, 0],
                            relative[1, 0] - relative[0, 1])
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine))


def _geometry(rotations):
    count = len(rotations)
    matrix = np.concatenate([
        rotations.reshape(count * 3, 3), -np.tile(np.eye(3), (count, 1))
    ], axis=1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if rank == 6 else None
    return matrix, rank, condition


def _fit(rotations, translations):
    matrix, rank, condition = _geometry(rotations)
    if rank != 6:
        raise ValueError("姿态分布退化，针尖偏移不能唯一确定。")
    origin = translations[0]
    with np.errstate(over="ignore", invalid="ignore"):
        rhs = -(translations - origin).reshape(-1)
    if not np.isfinite(rhs).all():
        raise ValueError("法兰位置超出有效数值范围，请检查毫米单位。")
    solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    offset = solution[:3]
    reference = solution[3:] + origin
    residuals = _residuals(rotations, translations, offset, reference)
    rms = math.hypot(*residuals) / math.sqrt(len(residuals))
    if not (np.isfinite(offset).all() and np.isfinite(reference).all()
            and np.isfinite(residuals).all() and math.isfinite(rms)):
        raise ValueError("标定结果超出有效数值范围，请检查位姿及单位。")
    transform = np.eye(4)
    transform[:3, 3] = offset
    return {
        "tcp_offset_mm": offset.tolist(),
        "reference_point_base_mm": reference.tolist(),
        "residuals_mm": residuals.tolist(),
        "rms_mm": float(rms),
        "max_error_mm": float(residuals.max()),
        "condition_number": condition,
        "flange_to_tcp": transform.tolist(),
        "transform_direction": "flange_from_tcp",
        "sample_count": len(rotations),
    }


def _residuals(rotations, translations, offset, reference):
    # 先减参考点，减少较大基坐标数值对残差计算的影响。
    vectors = np.einsum("nij,j->ni", rotations, offset) + (translations - reference)
    return np.array([math.hypot(*row) for row in vectors])


def _passes(result, limits):
    if result["condition_number"] > limits["max_condition_number"]:
        return False
    if result["sample_count"] == 4:
        return review_result(result, limits)["passed"]
    # 原四点模块的验收固定要求四条残差；多组沿用相同的两项门槛。
    return (result["rms_mm"] <= limits["max_rms_mm"]
            and result["max_error_mm"] <= limits["max_error_mm"])


def _subsets(count):
    """至多 256 个确定性的四点假设；加上逐一留出的假设。"""
    if math.comb(count, 4) <= 256:
        yield from itertools.combinations(range(count), 4)
    else:
        rng = np.random.default_rng(4351)
        seen = set()
        for _ in range(768):
            chosen = tuple(sorted(int(i) for i in rng.choice(count, 4, replace=False)))
            if chosen not in seen:
                seen.add(chosen)
                yield chosen
                if len(seen) == 256:
                    break
    for omit in range(count):
        yield tuple(i for i in range(count) if i != omit)


def _stable_consensus(rotations, translations, limits):
    """保守多数一致集：至少五条、严格多数、逐条留出复核，并拒绝歧义。

    不对残差最高的点循环删除。四点假设仅用于发现候选多数，最终必须
    通过全部保留点拟合及每个保留点的独立留出预测检查。
    """
    count = len(rotations)
    if count < 6:
        return None, "不足 6 条非冗余数据，不能可靠区分异常触点；请补采不同姿态。"
    minimum = max(5, count // 2 + 1)
    candidates = {}
    for subset in _subsets(count):
        selected = np.asarray(subset, dtype=int)
        try:
            fitted = _fit(rotations[selected], translations[selected])
            for _ in range(3):
                distances = _residuals(rotations, translations, fitted["tcp_offset_mm"],
                                       fitted["reference_point_base_mm"])
                members = tuple(int(i) for i in np.flatnonzero(distances <= limits["max_error_mm"]))
                if len(members) < minimum:
                    break
                selected = np.asarray(members, dtype=int)
                fitted = _fit(rotations[selected], translations[selected])
            else:
                distances = _residuals(rotations, translations, fitted["tcp_offset_mm"],
                                       fitted["reference_point_base_mm"])
                final_members = tuple(int(i) for i in np.flatnonzero(
                    distances <= limits["max_error_mm"]))
                if final_members == members and _passes(fitted, limits):
                    candidates[members] = fitted
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        return None, "没有找到满足门槛的稳定多数，不能判断是哪条触点有误；请检查固定点并补采。"

    ordered = sorted(candidates, key=lambda members: (-len(members), candidates[members]["rms_mm"]))
    best_members = ordered[0]
    best = candidates[best_members]
    # 候选组若包含不同的异常判定，不能以最低残差强选一组。
    best_set = set(best_members)
    if any(not set(other).issubset(best_set) for other in ordered[1:]):
        return None, "存在相互冲突的多数拟合，异常点归属不明确；请补采，不能自动挑选低残差数据。"

    distances = _residuals(rotations, translations, best["tcp_offset_mm"],
                           best["reference_point_base_mm"])
    excluded = [i for i in range(count) if i not in best_set]
    if not excluded or any(distances[i] <= 2.0 * limits["max_error_mm"] for i in excluded):
        return None, "部分残差接近门槛，缺少明确的异常分离；请复核触点后补采。"
    maximum_prediction = 0.0
    maximum_shift = 0.0
    for omit in best_members:
        selected = np.asarray([i for i in best_members if i != omit], dtype=int)
        try:
            fitted = _fit(rotations[selected], translations[selected])
        except (ValueError, np.linalg.LinAlgError):
            return None, "多数拟合在留出一条数据后退化；请增加另一方向的倾斜姿态。"
        if not _passes(fitted, limits):
            return None, "多数拟合在逐条留出复核时不稳定；请补采另一方向的倾斜姿态。"
        prediction = float(_residuals(rotations[omit:omit + 1], translations[omit:omit + 1],
                                     fitted["tcp_offset_mm"], fitted["reference_point_base_mm"])[0])
        shift = math.dist(fitted["tcp_offset_mm"], best["tcp_offset_mm"])
        maximum_prediction = max(maximum_prediction, prediction)
        maximum_shift = max(maximum_shift, shift)
        if prediction > limits["max_error_mm"] or shift > limits["max_error_mm"]:
            return None, "多数拟合未通过逐条留出预测或 TCP 稳定性复核；请增加不同姿态的可靠触点。"
    return {
        "members": best_members,
        "result": best,
        "residuals": distances,
        "holdout_max_error_mm": maximum_prediction,
        "holdout_max_tcp_shift_mm": maximum_shift,
    }, ""


def review_dataset(samples, mode="rpy_deg", criteria=None):
    """审查任意数量的法兰位姿（位置 mm），不删除输入、不保存文件。

    rows 与输入逐条对应。eligible_indices 是最终合格样本的零起始索引；
    result 仅在 passed=True 时提供，残差顺序与 eligible_indices 一致。
    nearest_angle_deg 是与此前选出的姿态代表的最小角差，首条为 None。
    相近姿态不会在一致性检查前丢弃；无法分辨冲突时全部保留待复核。
    至少四个独立姿态可验收；异常识别仅对无冗余姿态的一致集进行。
    """
    limits = validate_criteria(criteria)
    if mode not in ("rpy_deg", "rotation_vector_rad"):
        raise ValueError("姿态格式无效，请选择 RPY 欧拉角或旋转向量。")
    try:
        source = list(samples)
    except TypeError as exc:
        raise ValueError("采集数据必须是法兰位姿列表。") from exc
    review = {
        "passed": False, "rows": [], "result": None, "eligible_indices": [],
        "candidate_indices": [], "summary": "", "guidance": "", "criteria": limits,
        "rank": None, "condition_number": None, "sample_count": len(source),
        "scope": "仅审查采点与拟合一致性；无法检测实际接触，不代表独立验证的绝对定位精度。",
    }
    poses, rotations, indices = [], [], []
    for index, sample in enumerate(source):
        row = {"status": "待审查", "reason": "等待足够的不同姿态。",
               "residual_mm": None, "nearest_angle_deg": None}
        review["rows"].append(row)
        try:
            pose = np.asarray(sample, dtype=float)
            if pose.shape != (6,) or not np.isfinite(pose).all():
                raise ValueError("需要 6 个有限位姿数值。")
            rotation = rotation_matrix(pose[3:], mode)
        except (TypeError, ValueError, OverflowError) as exc:
            row.update(status="不合格", reason=f"输入无效：{exc}")
            continue
        poses.append(pose)
        rotations.append(rotation)
        indices.append(index)
    review["candidate_indices"] = list(indices)
    count = len(indices)
    if count < 4:
        review["summary"] = f"共 {len(source)} 条，有效位姿 {count} 条；至少需要 4 条。"
        review["guidance"] = (
            f"保持针尖对准同一固定点，改变法兰倾角后停稳；与已保留姿态至少相差 "
            f"{limits['min_angle_deg']:g}°。包含前后、左右两个方向的倾斜，不要只绕针轴转。")
        return review
    poses = np.asarray(poses)
    rotations = np.asarray(rotations)
    for index in indices:
        review["rows"][index].update(status="待复核", reason="尚未通过整组一致性验收。")
    # 这里仅统计独立方向，不删样本、不据此判定触点好坏。
    # 先悬空停稳、随后以相近姿态回到触点时，两条都必须接受一致性检查。
    representatives, redundant = [], {}
    for local, rotation in enumerate(rotations):
        if representatives:
            angles = [_angle(rotations[i], rotation) for i in representatives]
            nearest = int(np.argmin(angles))
            angle = angles[nearest]
            review["rows"][indices[local]]["nearest_angle_deg"] = angle
            if angle < limits["min_angle_deg"] - 1e-9:
                redundant[local] = (representatives[nearest], angle)
                continue
        representatives.append(local)
    if len(representatives) < 4:
        description = f"仅 {len(representatives)} 个相差至少 {limits['min_angle_deg']:g}° 的独立姿态，至少需要 4 个"
        for index in indices:
            review["rows"][index]["reason"] = description + "；尚不能判定相近姿态的触点好坏，均保留待复核。"
        review["summary"] = description + "。"
        review["guidance"] = "保持针尖对准同一固定点，补采前后、左右不同方向的倾斜；重复停稳不增加独立姿态数量。"
        return review
    try:
        _, rank, condition = _geometry(rotations[representatives])
        review.update(rank=rank, condition_number=condition)
        if rank != 6 or condition > limits["max_condition_number"]:
            description = f"姿态矩阵秩 {rank}/6" if rank != 6 else (
                f"条件数 {condition:.2f}，上限 {limits['max_condition_number']:g}")
            for index in indices:
                review["rows"][index]["reason"] = f"整组姿态分布不足：{description}；不能判定单条好坏。"
            review["summary"] = description + "，请补采另一方向的姿态。"
            review["guidance"] = (
                "保持针尖对准同一固定点；如果已做前后倾斜，下一组改为左右倾斜，并适当增大跨度。"
                "避免只绕针轴旋转或全部向同一方向倾斜。")
            return review
        fitted = _fit(rotations, poses[:, :3])
        for local, index in enumerate(indices):
            review["rows"][index]["residual_mm"] = fitted["residuals_mm"][local]
        if _passes(fitted, limits):
            # 全部有效数据先通过一致性检查，才允许选取代表、标记冗余。
            # 最终结果仅由独立姿态求解，重复样本不能改善门槛或增加票数。
            accepted = tuple(representatives)
            fitted = _fit(rotations[representatives], poses[representatives, :3])
            residuals = _residuals(rotations, poses[:, :3], fitted["tcp_offset_mm"],
                                   fitted["reference_point_base_mm"])
            for local, index in enumerate(indices):
                review["rows"][index]["residual_mm"] = float(residuals[local])
            if not _passes(fitted, limits) or float(max(residuals)) > limits["max_error_mm"]:
                for index in indices:
                    review["rows"][index]["reason"] = "独立姿态拟合未同时满足全部样本的一致性门槛，尚不能判定单条好坏。"
                review["summary"] = "选取独立姿态后未通过一致性复核；全部有效样本保留待复核。"
                review["guidance"] = "请核对每次针尖是否回到同一固定点，并补采不同方向的姿态；不要仅按先后顺序删除相近姿态。"
                return review
            review["method"] = "全部有效数据一致性检查后选取独立姿态拟合"
        else:
            # 相近姿态若互相冲突，原始条数不能代表独立证据数量。
            # 保守保留所有有效样本，既不先来先赢，也不让重复停稳形成假多数。
            if redundant:
                consensus = None
                explanation = "数据中存在相近姿态且整组不一致，无法可靠选择姿态代表；所有有效样本保留待复核，未按采集先后丢弃。"
            else:
                consensus, explanation = _stable_consensus(rotations, poses[:, :3], limits)
            if consensus is None:
                for index in indices:
                    review["rows"][index]["reason"] = (
                        "整组残差不合格，当前无法可靠区分异常点；显示的残差仅供复查。")
                review["summary"] = (
                    f"整组 RMS {fitted['rms_mm']:.4f} mm、最大残差 "
                    f"{fitted['max_error_mm']:.4f} mm；未通过验收。")
                review["guidance"] = explanation + (
                    " 确认固定点没有移动、针尖未受压弯曲，重新轻触同一点；"
                    "沿另一方向倾斜并停稳，补采后重新审查。")
                return review
            accepted = consensus["members"]
            fitted = consensus["result"]
            residuals = consensus["residuals"]
            review["method"] = "严格多数一致集与逐条留出复核"
            review["holdout_max_error_mm"] = consensus["holdout_max_error_mm"]
            review["holdout_max_tcp_shift_mm"] = consensus["holdout_max_tcp_shift_mm"]
        accepted_set = set(accepted)
        for local, index in enumerate(indices):
            row = review["rows"][index]
            row["residual_mm"] = float(residuals[local])
            if local in accepted_set:
                row.update(status="合格", reason="姿态有效，参与一致集拟合并通过当前残差门槛。")
            elif local in redundant:
                representative, angle = redundant[local]
                row.update(status="不合格", reason=(
                    f"全部有效数据已通过一致性检查；与第 {indices[representative] + 1} 条仅差 {angle:.2f}°，"
                    f"不足 {limits['min_angle_deg']:g}°，姿态冗余，不参与独立姿态拟合。"))
            else:
                row.update(status="不合格", reason=(
                    f"与稳定多数不一致：残差 {residuals[local]:.4f} mm，"
                    "已从候选结果中排除；请复查该次触点。"))
        review.update(passed=True, result=fitted, rank=6,
                      condition_number=fitted["condition_number"],
                      eligible_indices=[indices[i] for i in accepted])
        rejected = len(source) - len(accepted)
        review["summary"] = (
            f"共 {len(source)} 条，合格 {len(accepted)} 条，不合格 {rejected} 条；"
            f"合格项 RMS {fitted['rms_mm']:.4f} mm、最大残差 {fitted['max_error_mm']:.4f} mm。")
        review["guidance"] = (
            "审查通过。可查看各条原因，再点击“删除不合格”保留合格项；保存仍需手动操作。"
            if rejected else "全部非冗余数据通过当前验收门槛，可手动保存标定结果。")
        return review
    except (ValueError, np.linalg.LinAlgError) as exc:
        review["summary"] = f"本次无法完成标定：{exc}"
        review["guidance"] = "请核对法兰位姿及毫米单位，保持针尖在同一点，增加两个方向的不同倾角。"
        return review
