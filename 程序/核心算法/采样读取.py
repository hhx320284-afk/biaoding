"""读取本程序导出的完整采样文件夹；仅恢复内存数据，不写入源文件。"""

from __future__ import annotations

import json
from pathlib import Path
import struct

import cv2
import numpy as np

from 核心算法.标定板 import BoardSpec
from 核心算法.标定算法 import _samples


def read_sample_folder(folder, max_bytes=512 * 1024 * 1024) -> dict:
    """完整校验成功后返回 metadata、board、samples、folder。

    照片保留原始编码字节，角点仍为原图坐标；不重新检测、不应用预览裁剪。
    max_bytes 限制读取文件的总大小，并限制单张照片解码后的大小。
    """
    folder = Path(folder).expanduser().resolve()
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("可用于导入采样的内存不足，请先保存并清空其他采样。")
    if not folder.is_dir():
        raise ValueError(f"采样目录不存在：{folder}")
    used_bytes = 0

    def read_bytes(path):
        nonlocal used_bytes
        try:
            if path.stat().st_size > max_bytes - used_bytes:
                raise ValueError(f"{path.name}：整批采样超过可用内存上限，请先保存并清空其他采样。")
            with path.open('rb') as stream:
                value = stream.read(max_bytes - used_bytes + 1)
        except OSError as exc:
            raise ValueError(f"无法读取 {path.name}：{exc.strerror or exc}") from exc
        used_bytes += len(value)
        if used_bytes > max_bytes:
            raise ValueError(f"{path.name}：整批采样超过可用内存上限。")
        return value

    def read_json(path):
        try:
            value = json.loads(read_bytes(path).decode('utf-8-sig'))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path.name}：JSON 内容损坏或编码不正确。") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}：应为本程序导出的 JSON 对象。")
        return value

    manifest_path = folder / '采样信息.json'
    if not manifest_path.is_file():
        raise ValueError("未找到采样信息.json。请选择含该文件及采样照片的完整采样文件夹。")
    metadata = read_json(manifest_path)
    try:
        if type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 1:
            raise ValueError("不支持此采样文件版本。")
        if metadata.get('object_point_unit') != 'm':
            raise ValueError("标定板三维坐标必须使用米。")
        if metadata.get('model') not in ('pinhole', 'fisheye'):
            raise ValueError("缺少有效的针孔或鱼眼镜头模型。")
        size = metadata.get('image_size')
        if (not isinstance(size, list) or len(size) != 2
                or any(type(v) is not int or v <= 0 for v in size)):
            raise ValueError("原始图像宽、高必须为正整数。")
        if size[0] * size[1] * 3 > max_bytes:
            raise ValueError("单张照片解码所需内存超过可用内存上限。")
        if not isinstance(metadata.get('board'), dict):
            raise ValueError("缺少标定板参数。")
        board = BoardSpec(**metadata['board'])
        board.validate()
        records = metadata.get('sample_records')
        if not isinstance(records, list) or not records:
            raise ValueError("没有采样记录。")
        indexes = [r.get('index') if isinstance(r, dict) else None for r in records]
        if (any(type(i) is not int or i <= 0 for i in indexes)
                or len(set(indexes)) != len(indexes)):
            raise ValueError("采样序号必须为不重复的正整数。")
        for name in ('camera_name', 'device'):
            if not isinstance(metadata.get(name), str):
                raise ValueError(f"缺少有效的相机记录：{name}。")
        if not isinstance(metadata.get('actual', {}), dict):
            raise ValueError("相机实际采集参数格式不正确。")
    except (TypeError, ValueError, cv2.error) as exc:
        raise ValueError(f"采样信息.json：{exc}") from exc

    samples = []
    for record in records:
        base = folder / f'采样_{record["index"]:03d}'
        record_path = base.with_suffix('.json')
        photo_path = base.with_suffix('.png')
        sample = read_json(record_path)
        try:
            objects, pixels = _samples([sample], size)
            ids = np.asarray(sample['ids'])
            if (ids.dtype.kind not in 'iu' or ids.size != len(objects[0])
                    or np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max)
                    or np.unique(ids).size != ids.size):
                raise ValueError("角点 ID 必须为数量匹配、不重复的非负整数。")
            if len(objects[0]) > board.expected_count:
                raise ValueError("角点数量超过当前标定板的角点总数。")
            captured_at = sample.get('captured_at')
            if not isinstance(captured_at, str) or not captured_at:
                raise ValueError("缺少拍摄时间。")
            if record.get('captured_at') != captured_at:
                raise ValueError("拍摄时间与采样信息.json 不一致。")
            if not isinstance(sample.get('metrics', {}), dict):
                raise ValueError("采样质量记录格式不正确。")
            sample.update(object_points=objects[0].reshape(-1, 3),
                          image_points=pixels[0].reshape(-1, 2),
                          ids=np.ascontiguousarray(ids, dtype=np.int32).reshape(-1, 1),
                          metrics=sample.get('metrics', {}))
        except (KeyError, TypeError, ValueError, cv2.error) as exc:
            raise ValueError(f"{record_path.name}：{exc}") from exc
        image = read_bytes(photo_path)
        if (len(image) < 24 or image[:8] != b'\x89PNG\r\n\x1a\n'
                or image[12:16] != b'IHDR'):
            raise ValueError(f"{photo_path.name}：不是有效的原始 PNG 照片。")
        if struct.unpack('>II', image[16:24]) != tuple(size):
            raise ValueError(f"{photo_path.name}：照片尺寸与采样信息.json 不一致，不能使用裁剪或缩放后的照片。")
        try:
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise ValueError(f"{photo_path.name}：照片解码失败。") from exc
        if decoded is None or decoded.shape[:2] != (size[1], size[0]):
            raise ValueError(f"{photo_path.name}：照片损坏或尺寸不正确。")
        del decoded
        sample['image'] = image
        samples.append(sample)
    return dict(metadata=metadata, board=board, samples=samples, folder=str(folder))
