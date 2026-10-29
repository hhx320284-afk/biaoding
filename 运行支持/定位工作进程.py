"""FoundationPose / SAM3 隔离工作进程；图像与结果仅经内存管道传递。"""

from __future__ import annotations

import gc
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# 保留协议专用描述符；包括 C/CUDA 库在内的 stdout 输出统一转到 stderr。
PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
ROOT = Path('/home/w/codes/foundationpose_origin')
SAM_MODE = '--sam' in sys.argv
if not SAM_MODE:
    # 在 NumPy/OpenCV/torch 绑定 C++ 运行库之前加载已编译的本地扩展。
    sys.path.insert(0, str(ROOT / 'FoundationPose/mycpp/build'))
    import mycpp  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '核心算法'))
from 选孔视觉 import (HOLES_PATH, MESH_PATH, SAM_PYTHON, VisionError, _packet_bytes,
                   _read_packet, _write_packet, load_holes, project_holes, validate_pose)
from 跟随孔身份 import selected_hole_support

import cv2
import numpy as np


def run_sam(request):
    sys.path.insert(0, str(ROOT / 'sam3'))
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    if not torch.cuda.is_available():
        raise VisionError('SAM3 未检测到可用 CUDA GPU。')
    torch.set_num_threads(4)
    rgb = np.asarray(request['rgb'])
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = map(float, request['bbox'])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise VisionError('请在画面内框选完整工件。')
    box = [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        model = build_sam3_image_model(
            bpe_path=str(ROOT / 'sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
            checkpoint_path=str(ROOT / 'sam3/checkpoints/sam3.pt'), load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=.3)
        state = processor.set_image(Image.fromarray(rgb))
        state = processor.add_geometric_prompt(box=box, label=True, state=state)
        masks = state.get('masks')
        if masks is None:
            raise VisionError('SAM3 未找到框选工件，请重新框选。')
        masks = masks.detach().cpu().numpy().astype(bool).reshape(-1, height, width)
        scores = state['scores'].detach().cpu().float().numpy().reshape(-1)
        roi = np.zeros((height, width), bool)
        roi[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
        candidates = [i for i, mask in enumerate(masks)
                      if mask.sum() >= 100 and (mask & roi).sum() / max(mask.sum(), 1) >= .85]
        if not candidates:
            raise VisionError('SAM3 没有得到与整件框一致的有效掩码，请重新框选。')
        best = max(candidates, key=lambda i: float(scores[i]))
        return dict(ok=True, mask=masks[best], score=float(scores[best]))


class PoseWorker:
    def __init__(self):
        self.estimator = None
        self.context = None
        self.pose = None
        self.holes = load_holes()
        sys.path.insert(0, str(ROOT))
        from realsense_pose_holes import HoleRefiner
        self.refiner = HoleRefiner(HOLES_PATH)

    def load_engine(self):
        if self.estimator is not None:
            return
        import torch
        import trimesh
        import nvdiffrast.torch as dr
        sys.path.insert(0, str(ROOT / 'FoundationPose'))
        from estimater import FoundationPose

        class MemoryMesh(trimesh.Trimesh):
            # 上游无条件导出一份未被推理使用的 /tmp OBJ；此网格不允许落盘。
            def copy(self, include_cache=False, include_visual=True):
                return MemoryMesh(vertices=self.vertices.copy(), faces=self.faces.copy(),
                                  visual=self.visual.copy() if include_visual else None, process=False)

            def export(self, *args, **kwargs):
                return None

        torch.set_num_threads(4)
        original = trimesh.load(str(MESH_PATH), force='mesh')
        mesh = MemoryMesh(vertices=original.vertices.copy() * .001, faces=original.faces.copy(),
                          visual=original.visual.copy(), process=False)
        self.context = dr.RasterizeCudaContext()
        with torch.inference_mode():
            self.estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                           mesh=mesh, glctx=self.context, debug=0,
                                           debug_dir=str(Path(__file__).resolve().parent), use_icp=False)
            self.estimator.diameter = float(self.estimator.diameter)
            self.estimator.mesh_path = None

    def unload_engine(self):
        self.estimator = self.context = self.pose = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def segment(self, rgb, bbox):
        # SAM 运行完退出释放 GPU 后，再加载 FP，避免两个大模型同时占满显存。
        self.unload_engine()
        process = subprocess.Popen([str(SAM_PYTHON), '-B', str(Path(__file__).resolve()), '--sam'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
        try:
            output, _ = process.communicate(_packet_bytes(dict(rgb=rgb, bbox=bbox)), timeout=180.)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise VisionError('SAM3 分割超时，请重新开始。') from exc
        try:
            result = _read_packet(io.BytesIO(output))
        except Exception as exc:
            raise VisionError('SAM3 工作进程启动失败，请查看定位错误信息。') from exc
        if process.returncode or not result.get('ok'):
            raise VisionError(result.get('error', 'SAM3 分割失败。'))
        return result

    @staticmethod
    def frame_arrays(frame):
        if not isinstance(frame, dict):
            raise VisionError('缺少同步 RGB、深度和内参快照。')
        rgb = np.asarray(frame['rgb'])
        depth = np.asarray(frame['depth'])
        K = np.asarray(frame['K'], dtype=np.float32)
        coefficients = np.asarray(frame.get('dist_coeffs', np.zeros(5)), dtype=float).reshape(-1)
        if coefficients.size == 0:
            coefficients = np.zeros(5)
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or depth.shape != rgb.shape[:2]
                or depth.dtype.kind != 'f' or K.shape != (3, 3) or not np.isfinite(K).all()
                or min(K[0, 0], K[1, 1]) <= 0 or coefficients.size not in (4, 5, 8, 12, 14)
                or not np.isfinite(coefficients).all()):
            raise VisionError('快照需 uint8 RGB、对齐的米单位浮点深度、有效内参与 OpenCV 畸变参数。')
        if not np.isfinite(float(frame['monotonic'])):
            raise VisionError('视觉帧缺少有效采集时间。')
        depth = np.asarray(depth, dtype=np.float32).copy()
        depth[~np.isfinite(depth) | (depth < .001)] = 0
        rgb = np.ascontiguousarray(rgb)
        maps = None
        if np.any(np.abs(coefficients) > 1e-10):
            height, width = depth.shape
            maps = cv2.initUndistortRectifyMap(K, coefficients, None, K, (width, height), cv2.CV_32FC1)
            rgb_rect = cv2.remap(rgb, *maps, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, *maps, cv2.INTER_NEAREST)
        else:
            rgb_rect = rgb
        return rgb, rgb_rect, depth, K, coefficients, maps

    def sync_tracking_pose(self):
        import torch
        self.estimator.pose_last = (torch.as_tensor(self.pose, device='cuda', dtype=torch.float32)
                                    @ torch.linalg.inv(self.estimator.get_tf_to_centered_mesh()))

    def render_depth(self, pose, K, shape):
        import torch
        from Utils import nvdiffrast_render
        centered = (torch.as_tensor(pose, device='cuda', dtype=torch.float32)
                    @ torch.linalg.inv(self.estimator.get_tf_to_centered_mesh()))
        _, depth, _ = nvdiffrast_render(K=K, H=shape[0], W=shape[1], ob_in_cams=centered[None],
                                       glctx=self.context, mesh_tensors=self.estimator.mesh_tensors,
                                       use_light=False, extra={})
        return depth[0].detach().cpu().numpy()

    @staticmethod
    def depth_support(stats):
        if stats.get('compared_pixels', 0) < 100 or stats.get('valid_depth_fraction', 0.) < .3:
            return False, 'depth_support_insufficient'
        if (stats.get('median_absolute_difference_m', float('inf')) > .02
                or stats.get('fraction_within_20mm', 0.) < .6):
            return False, 'depth_large_disagreement'
        return True, 'depth_consistency_passed_not_accuracy'

    def handle(self, request):
        import torch
        from realsense_pose_overlay import depth_agreement, hole_depth_consistent
        op = request.get('op')
        if op == 'warmup':
            self.load_engine()
            return dict(ok=True, engine_ready=True, gpu=torch.cuda.get_device_name(0))
        if op == 'reset':
            self.pose = None
            if self.estimator is not None:
                self.estimator.pose_last = None
            return dict(ok=True, reset=True)
        if op not in ('init', 'track', 'refine', 'follow'):
            raise VisionError('未知视觉操作。')
        following = op == 'follow'
        near_rgb = following and bool(request.get('near_rgb', False))
        selected_id = str(request.get('selected_hole_id', ''))
        if following and selected_id not in self.refiner.hole_ids:
            raise VisionError('自动跟随缺少有效的所选孔编号。')
        selected_support = dict(id=selected_id, detected=False, accepted=False,
                                reason='foundationpose_projection_only') if following else None
        frame = request.get('frame')
        raw_rgb, rgb, depth, K, coefficients, maps = self.frame_arrays(frame)
        started = time.monotonic()
        refined, mask_score, depth_stats = None, None, None
        with torch.inference_mode():
            if op == 'init':
                segmentation = self.segment(raw_rgb, request.get('bbox', []))
                mask = segmentation['mask']
                mask_score = segmentation['score']
                if maps is not None:
                    mask = cv2.remap(mask.astype(np.uint8), *maps, cv2.INTER_NEAREST).astype(bool)
                if (mask & (depth > .001)).sum() < 100:
                    raise VisionError('工件掩码内的有效深度不足，无法完成粗定位。')
                self.load_engine()
                self.pose = validate_pose(self.estimator.register(K=K, rgb=rgb, depth=depth,
                                                                 ob_mask=mask, iteration=5))
            elif self.pose is None or self.estimator is None:
                raise VisionError('请先框选整件并初始化粗定位。')
            elif op in ('track', 'follow'):
                self.pose = validate_pose(self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=2))
            # 自动跟随先追踪本帧整件，再用本帧 RGB 实测；旧帧只作模型内部搜索先验。
            if op == 'refine' or near_rgb:
                seed = self.pose.copy() if following else validate_pose(request.get('T_cam_cad', self.pose))
                refined = self.refiner.refine(rgb, K, seed)
                if not refined['accepted']:
                    confirmed = self.refiner.confirm_pose(K, seed, refined)
                    if confirmed['accepted']:
                        refined = confirmed
                if following:
                    selected_support = selected_hole_support(
                        self.refiner, K, validate_pose(refined['T']), refined, selected_id)
                    refined['diagnostics']['selected_hole_support'] = selected_support
                if refined['accepted']:
                    candidate = validate_pose(refined['T'])
                    original_depth = self.render_depth(seed, K, depth.shape)
                    candidate_depth = self.render_depth(candidate, K, depth.shape)
                    before = depth_agreement(original_depth, depth)
                    depth_stats = depth_agreement(candidate_depth, depth)
                    supported, reason = self.depth_support(depth_stats)
                    if supported and not hole_depth_consistent(before, depth_stats, original_depth, candidate_depth, depth):
                        supported, reason = False, 'hole_fit_disagrees_with_depth'
                    refined['diagnostics'].update(depth_before=before, depth_after=depth_stats,
                                                  depth_note='仅排查缺少表面支持或粗大偏差，不作为定位精度证据。')
                    if supported and following and not selected_support['accepted']:
                        supported, reason = False, selected_support['reason']
                    if supported:
                        self.pose = candidate
                        self.sync_tracking_pose()
                    else:
                        refined['accepted'] = False
                        refined['reason'] = refined['diagnostics']['reason'] = reason
            if op != 'refine' and not near_rgb:
                depth_stats = depth_agreement(self.render_depth(self.pose, K, depth.shape), depth)
        projection = project_holes(self.pose, K, raw_rgb.shape, coefficients, self.holes)
        camera_cad = -self.pose[:3, :3].T @ self.pose[:3, 3]
        view = abs(float(camera_cad[1])) / max(float(np.linalg.norm(camera_cad)), 1e-9)
        visible = [hole for hole in projection if hole['visible']]
        quality = dict(accepted=bool(visible), reason='粗定位结果，等待停稳后的 RGB 孔位验证。', matches=0,
                       residual_px=None, view_cosine=view, visible_count=len(visible),
                       min_diameter_px=min((h['diameter_px'] for h in visible), default=0.),
                       observed_hole_ids=[], scope='图像与模型一致性，不代表独立验证的 0.1 mm 定位精度。')
        quality['depth'] = depth_stats
        if refined is None:
            supported, reason = self.depth_support(depth_stats)
            if not supported:
                quality.update(accepted=False, reason=reason)
        if refined is not None:
            residual = np.asarray(refined['residual_after_px'], dtype=float)
            measured = np.asarray(refined['inliers'], bool) & np.isfinite(residual)
            matches = int(measured.sum())
            error = float(np.median(residual[measured])) if matches else None
            accepted = bool(refined['accepted'] and matches >= self.refiner.config.min_matches
                            and error is not None and np.isfinite(error))
            identifiers = [str(refined['hole_ids'][i]) for i in np.flatnonzero(measured)]
            quality.update(accepted=accepted, reason=refined['reason'], matches=matches,
                           residual_px=error, observed_hole_ids=identifiers,
                           measurement_image='去畸变 RGB' if maps is not None else 'RGB')
            for hole in projection:
                hole['measured'] = hole['id'] in identifiers
        if following:
            measured = bool(near_rgb and refined is not None and refined['accepted']
                            and selected_support['accepted'])
            quality.update(tracking_method='rgb_holes' if near_rgb else 'foundationpose',
                           selected_hole_support=selected_support, measured_this_frame=measured)
            if near_rgb:
                # 近孔验证失效不能退回模型投影继续运动；上层据此立即停止。
                quality['accepted'] = bool(quality['accepted'] and measured)
                if measured:
                    quality.update(matches=selected_support['unique_support_count'],
                                   residual_px=selected_support['support_median_px'],
                                   observed_hole_ids=selected_support['support_ids'])
                    for hole in projection:
                        hole['measured'] = hole['id'] in selected_support['support_ids']
        return dict(ok=True, op=op, T_cam_cad=self.pose.tolist(), frame_id=frame['frame_id'],
                    monotonic=float(frame['monotonic']), projected_holes=projection, quality=quality,
                    refinement=refined, mask_score=mask_score, processing_seconds=time.monotonic() - started,
                    translation_unit='m', coordinate_frame='original_CAD', K=K.tolist(),
                    follow_generation=request.get('follow_generation'),
                    follow_epoch=request.get('follow_epoch'), selected_hole_id=selected_id,
                    near_rgb=near_rgb)


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    if SAM_MODE:
        try:
            _write_packet(PROTOCOL, run_sam(_read_packet(sys.stdin.buffer)))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _write_packet(PROTOCOL, dict(ok=False, error=str(exc)))
        return
    import torch
    if not torch.cuda.is_available():
        _write_packet(PROTOCOL, dict(ok=False, error='FoundationPose 未检测到可用 CUDA GPU。'))
        return
    worker = PoseWorker()
    _write_packet(PROTOCOL, dict(ok=True, ready=True, gpu=torch.cuda.get_device_name(0)))
    while True:
        try:
            request = _read_packet(sys.stdin.buffer)
        except EOFError:
            return
        try:
            response = worker.handle(request)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = dict(ok=False, error=str(exc))
        _write_packet(PROTOCOL, response)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _write_packet(PROTOCOL, dict(ok=False, error=str(exc)))
