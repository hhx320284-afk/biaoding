"""自动跟随的当帧孔身份复核；只检查实测，不从模型补造观测。"""

import numpy as np


def selected_hole_support(refiner, K, pose, result, selected_id):
    """沿用孔口物理模型，复核选中孔、唯一对应及附近独立孔支持。"""
    identity = str(selected_id)
    out = dict(id=identity, detected=False, accepted=False, residual_px=None,
               nearest_other_px=None, neighbor_support_count=0, neighbor_ids=[],
               unique_support_count=0, support_median_px=None,
               reason='selected_hole_not_measured')

    def reject(reason):
        out['reason'] = reason
        return out

    ids = [str(value) for value in result.get('hole_ids', [])]
    if len(ids) != len(set(ids)) or ids.count(identity) != 1:
        return reject('selected_hole_identity_missing_or_duplicate')
    n, index = len(ids), ids.index(identity)
    observed = np.asarray(result.get('observed_uv', []), float)
    inliers = np.asarray(result.get('inliers', []), bool)
    holdout = np.asarray(result.get('holdout', []), bool)
    if observed.shape != (n, 2) or inliers.shape != (n,) or holdout.shape != (n,):
        return reject('hole_measurements_invalid')
    out['detected'] = bool(np.isfinite(observed[index]).all())
    if not result.get('accepted'):
        return reject('hole_pose_not_supported:' + str(result.get('reason', 'unknown')))
    # 透射孔口与前缘椭圆的图像中心都不等于 CAD 圆心投影。
    projected = np.asarray(refiner.project_measurements(K, pose, result), float)
    if projected.shape != observed.shape:
        return reject('hole_projection_invalid')
    finite = np.isfinite(observed).all(axis=1) & np.isfinite(projected).all(axis=1)
    errors = np.linalg.norm(observed-projected, axis=1)
    out['residual_px'] = float(errors[index]) if finite[index] else None
    if not finite[index] or not (inliers[index] or holdout[index]):
        return reject('selected_hole_not_independently_supported')
    if errors[index] > 1.:
        return reject('selected_hole_residual_too_large')

    # 同一观测不能给两个 CAD 编号投票；也不能比自己的编号更靠近邻孔。
    distances = np.linalg.norm(observed[:, None]-projected[None, :], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    np.fill_diagonal(distances, np.inf)
    nearest_other = distances.min(axis=1)
    spacing = np.linalg.norm(projected[:, None]-projected[None, :], axis=2)
    spacing[~np.isfinite(spacing)] = np.inf
    np.fill_diagonal(spacing, np.inf)
    nearest_spacing = spacing.min(axis=1)
    observed_distance = np.linalg.norm(observed[:, None]-observed[None, :], axis=2)
    observed_distance[~np.isfinite(observed_distance)] = np.inf
    np.fill_diagonal(observed_distance, np.inf)
    unique = ((nearest_other > errors*2.+2.) & (errors < nearest_spacing*.42)
              & (observed_distance.min(axis=1) >= 1.7))
    out['nearest_other_px'] = (float(nearest_other[index])
                               if np.isfinite(nearest_other[index]) else None)
    if not unique[index]:
        return reject('selected_hole_neighbor_ambiguous')
    support = ((inliers | holdout) & finite & unique
               & (errors <= refiner.config.max_reprojection_px))
    count = int(support.sum())
    out['unique_support_count'] = count
    out['support_ids'] = [ids[i] for i in np.flatnonzero(support)]
    out['ambiguous_ids'] = [ids[i] for i in np.flatnonzero(finite & ~unique)]
    if count < max(8, refiner.config.min_matches):
        return reject('too_few_unique_current_frame_holes')
    median = float(np.median(errors[support]))
    out['support_median_px'] = median
    if median > 1.:
        return reject('current_frame_hole_residual_too_large')

    # 检查最近四个模型邻孔；至少两个必须有独立、低残差且无歧义的当帧观测。
    neighbors = np.argsort(spacing[index])[:4]
    neighbors = neighbors[np.isfinite(spacing[index, neighbors])]
    reliable_neighbors = neighbors[support[neighbors] & (errors[neighbors] <= 1.)]
    out['neighbor_ids'] = [ids[i] for i in neighbors]
    out['neighbor_support_ids'] = [ids[i] for i in reliable_neighbors]
    out['neighbor_support_count'] = int(len(reliable_neighbors))
    if len(reliable_neighbors) < 2:
        return reject('selected_hole_neighbors_not_supported')
    out.update(accepted=True, reason='selected_hole_verified_in_current_rgb',
               held_out=bool(holdout[index]))
    return out
