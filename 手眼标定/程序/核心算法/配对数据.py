"""手眼配对样本的显式导入与保存；读取和验证不会写入任何文件。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np

from 手眼算法 import estimate_board, load_camera_record, parse_pose, validate_transform


SCHEMA = "handeye_samples_v1"
MAX_SAMPLES = 200
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 256 * 1024 * 1024
_PROTECTED_ROOT = Path(__file__).resolve().parents[3] / "鱼眼相机安装调试"


def _public_json(value):
    """转换 NumPy 数值，并递归剔除仅属于本次运行的私有字段。"""
    if isinstance(value, dict):
        return {str(key): _public_json(item) for key, item in value.items()
                if not str(key).startswith("_")}
    if isinstance(value, np.ndarray):
        return _public_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_public_json(item) for item in value]
    if isinstance(value, np.generic):
        return _public_json(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("保存数据包含非有限数值或不能写入 JSON 的内容；照片应单独保存。")


def _normalize_samples(samples, tcp_definition=None):
    if not isinstance(samples, (list, tuple)) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ValueError(f"样本文件必须包含 1～{MAX_SAMPLES} 组配对数据。")
    normalized, identifiers, hashes, poses = [], set(), {}, []
    for index, raw in enumerate(samples, 1):
        try:
            if not isinstance(raw, dict):
                raise ValueError("样本必须为 JSON 对象。")
            identifier = raw.get("id")
            if (isinstance(identifier, bool) or not isinstance(identifier, (str, int))
                    or not str(identifier).strip()):
                raise ValueError("缺少有效的样本 id。")
            key = str(identifier).strip()
            if key in identifiers:
                raise ValueError(f"样本 id 重复：{key}。")
            role = raw.get("role")
            if role not in ("calibration", "validation"):
                raise ValueError("样本 role 无效。")
            robot = validate_transform(raw.get("T_base_end"), "机器人位姿 T_base_end")
            board = validate_transform(raw.get("T_cam_board"), "标定板位姿 T_cam_board")
            rms = raw.get("reprojection_rms_px")
            if isinstance(rms, bool) or not isinstance(rms, (int, float, np.number)):
                raise ValueError("缺少有效的重投影误差 reprojection_rms_px。")
            rms = float(rms)
            if not math.isfinite(rms) or rms < 0:
                raise ValueError("重投影误差必须为有限的非负数。")
            digest = raw.get("image_sha256")
            if digest is not None:
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                    raise ValueError("image_sha256 必须为 64 位十六进制图像摘要。")
                digest = digest.lower()
                if digest in hashes:
                    raise ValueError(f"图像与第 {hashes[digest]} 组重复；请换一个姿态重新采集。")
            for previous_index, previous_pose in enumerate(poses, 1):
                if np.allclose(robot, previous_pose, rtol=0, atol=1e-8):
                    raise ValueError(f"机器人位姿与第 {previous_index} 组重复；请换一个姿态采集。")
            sample = {"id": identifier, "role": role, "T_base_end": robot.tolist(),
                      "T_cam_board": board.tolist(), "reprojection_rms_px": rms}
            if "T_base_flange" in raw:
                flange = validate_transform(raw["T_base_flange"], "法兰位姿 T_base_flange")
                sample["T_base_flange"] = flange.tolist()
            elif tcp_definition is not None:
                raise ValueError("带 TCP 定义的样本必须包含原始法兰位姿 T_base_flange。")
            if tcp_definition is not None:
                expected = flange @ np.asarray(tcp_definition["T_flange_tcp"], dtype=np.float64)
                if not np.allclose(robot, expected, rtol=0, atol=1e-8):
                    raise ValueError("末端位姿与原始法兰位姿、TCP 固定偏移不一致。")
            if digest is not None:
                sample["image_sha256"] = digest
            # 只保存测量与来源元数据；图像字节及未知照片载荷不进入样本 JSON。
            for field in ("source", "captured_at", "robot_channel", "pose_source"):
                if field in raw:
                    if not isinstance(raw[field], str):
                        raise ValueError(f"{field} 必须为文本。")
                    sample[field] = raw[field]
            if "pair_delta_ms" in raw:
                delta = raw["pair_delta_ms"]
                if (isinstance(delta, bool) or not isinstance(delta, (int, float, np.number))
                        or not math.isfinite(float(delta)) or delta < 0):
                    raise ValueError("pair_delta_ms 必须为有限的非负毫秒数。")
                sample["pair_delta_ms"] = float(delta)
            if "corners" in raw:
                corners = np.asarray(raw["corners"], dtype=np.float64)
                if (corners.ndim not in (2, 3) or corners.shape[-1] != 2 or corners.size == 0
                        or (corners.ndim == 3 and corners.shape[1] != 1)
                        or not np.isfinite(corners).all()):
                    raise ValueError("corners 必须为有限的二维角点坐标。")
                sample["corners"] = corners.reshape(-1, 2).tolist()
            if "image_size" in raw:
                size = raw["image_size"]
                if (not isinstance(size, (list, tuple)) or len(size) != 2
                        or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in size)):
                    raise ValueError("image_size 必须为正整数 [宽, 高]。")
                sample["image_size"] = [int(v) for v in size]
            if "point_count" in raw:
                count = raw["point_count"]
                if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
                    raise ValueError("point_count 必须为正整数。")
                if "corners" in sample and len(sample["corners"]) != count:
                    raise ValueError("point_count 与 corners 数量不一致。")
                sample["point_count"] = int(count)
            if "warnings" in raw:
                if not isinstance(raw["warnings"], list) or not all(isinstance(v, str) for v in raw["warnings"]):
                    raise ValueError("warnings 必须为文本列表。")
                sample["warnings"] = list(raw["warnings"])
            identifiers.add(key)
            if digest is not None:
                hashes[digest] = index
            poses.append(robot)
            normalized.append(sample)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"第 {index} 组样本无效：{exc}") from exc
    return normalized


def validate_tcp_definition(raw, frame=None, tool_name=None) -> dict:
    """TCP 在法兰坐标系中的固定变换，平移统一为米；不补造旧数据的工具偏移。"""
    if not isinstance(raw, dict):
        raise ValueError("tcp_definition 必须为 TCP 定义对象。")
    mode = raw.get("mode")
    if mode not in ("flange", "gripper", "custom"):
        raise ValueError("TCP 模式必须为 flange、gripper 或 custom。")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("TCP 定义必须包含非空名称 name。")
    transform = validate_transform(raw.get("T_flange_tcp"), "TCP 固定偏移 T_flange_tcp")
    expected_frame = "flange" if mode == "flange" else "tcp"
    if frame is not None and frame != expected_frame:
        raise ValueError("TCP 模式与 reference_frame 不一致。")
    if tool_name is not None and name.strip() != tool_name.strip():
        raise ValueError("TCP 定义名称与 tool_name 不一致。")
    if mode == "flange" and not np.allclose(transform, np.eye(4), rtol=0, atol=1e-8):
        raise ValueError("选择法兰时，TCP 固定偏移必须为单位矩阵。")
    definition = {"mode": mode, "T_flange_tcp": transform.tolist(), "name": name.strip()}
    if "source" in raw:
        if not isinstance(raw["source"], str):
            raise ValueError("TCP 定义来源 source 必须为文本。")
        definition["source"] = raw["source"]
    return definition


def make_bundle(camera_record, frame, tool_name, samples, tcp_definition=None) -> dict:
    record = {"schema": SCHEMA, "camera": camera_record,
              "reference_frame": frame, "tool_name": tool_name,
              "translation_unit": "m", "samples": samples}
    if tcp_definition is not None:
        record["tcp_definition"] = tcp_definition
    return validate_bundle(record)


def validate_bundle(raw) -> dict:
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError(f"不是受支持的配对样本文件；schema 必须为 {SCHEMA}。")
    if raw.get("reference_frame") not in ("flange", "tcp"):
        raise ValueError("reference_frame 必须为 flange（法兰）或 tcp。")
    if raw.get("translation_unit") != "m":
        raise ValueError("样本文件中的矩阵平移必须统一使用米（translation_unit: m）。")
    if not isinstance(raw.get("tool_name"), str):
        raise ValueError("tool_name 必须为工具名称文本。")
    if raw["reference_frame"] == "tcp" and not raw["tool_name"].strip():
        raise ValueError("TCP 样本必须记录工具名称／编号及偏移标识，不能使用空 tool_name。")
    camera = load_camera_record(raw.get("camera"))
    definition = (validate_tcp_definition(raw["tcp_definition"], raw["reference_frame"], raw["tool_name"])
                  if "tcp_definition" in raw else None)
    samples = _normalize_samples(raw.get("samples"), definition)
    expected_size = camera["calibration"]["image_size"]
    for index, sample in enumerate(samples, 1):
        if "image_size" in sample and sample["image_size"] != expected_size:
            raise ValueError(f"第 {index} 组样本分辨率与相机内参不一致。")
    record = {"schema": SCHEMA, "camera": _public_json(camera),
              "reference_frame": raw["reference_frame"], "tool_name": raw["tool_name"],
              "translation_unit": "m", "samples": samples}
    if definition is not None:
        record["tcp_definition"] = definition
    return record


def _read_limited(path, limit):
    with Path(path).open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"文件过大，上限为 {limit // (1024 * 1024)} MiB：{Path(path).name}")
    return content


def read_bundle(path) -> dict:
    try:
        raw = json.loads(_read_limited(path, MAX_JSON_BYTES).decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"样本文件不是有效的 UTF-8 JSON：{exc}") from exc
    return validate_bundle(raw)


def _check_output_path(path, protected_root):
    requested = Path(path).expanduser().absolute()
    resolved = requested.resolve()
    protected = Path(protected_root).expanduser().absolute()
    for target in (requested, resolved):
        for root in (protected, protected.resolve()):
            if target == root or root in target.parents:
                raise ValueError("不能写入受保护的“鱼眼相机安装调试”目录或指向该目录的符号链接。")
    return resolved


def write_json(path, record, protected_root=None):
    """仅供用户明确保存/导出后调用；同目录临时文件完成后原子替换。"""
    protected = _PROTECTED_ROOT if protected_root is None else Path(protected_root)
    target = _check_output_path(path, protected)
    content = json.dumps(_public_json(record), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        # 不自动建数据目录，避免选择错误路径时留下空目录。
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".手眼保存-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if _check_output_path(path, protected) != target:
            raise ValueError("保存路径在写入期间发生变化，请重新选择位置。")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def image_digest(frame: np.ndarray) -> str:
    """按解码后的 BGR 图像尺寸与像素去重，避免同图不同压缩格式被重复使用。"""
    digest = hashlib.sha256(str(frame.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def import_csv(path, camera_record, representation, translation_unit, angle_unit) -> list:
    """一次性读取全部配对图像；任一行失败时不返回部分结果、不写文件。"""
    camera = load_camera_record(camera_record)
    columns = {"rotvec": ["x", "y", "z", "rx", "ry", "rz"],
               "rpy": ["x", "y", "z", "rx", "ry", "rz"],
               "quaternion": ["x", "y", "z", "qx", "qy", "qz", "qw"],
               "matrix": [f"m{row}{column}" for row in range(4) for column in range(4)]}
    if representation not in columns:
        raise ValueError("CSV 位姿格式必须为 matrix、rotvec、rpy 或 quaternion。")
    if translation_unit not in ("m", "mm") or angle_unit not in ("rad", "deg"):
        raise ValueError("CSV 平移单位必须为 m/mm，角度单位必须为 rad/deg。")
    csv_path = Path(path).expanduser().resolve()
    try:
        text = _read_limited(csv_path, MAX_JSON_BYTES).decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("CSV 必须使用 UTF-8 编码。") from exc
    reader = csv.DictReader(io.StringIO(text), strict=True)
    try:
        fields = reader.fieldnames
    except csv.Error as exc:
        raise ValueError(f"CSV 第 1 行表头无效：{exc}") from exc
    required = ["image", *columns[representation]]
    if not fields or len(set(fields)) != len(fields) or any(key not in fields for key in required):
        raise ValueError("CSV 表头缺失或重复；当前格式需要：" + ", ".join(required))
    samples, images, line_numbers = [], [], []
    image_memory = 0
    try:
        for row in reader:
            line = reader.line_num
            try:
                if len(samples) >= MAX_SAMPLES:
                    raise ValueError(f"最多导入 {MAX_SAMPLES} 组样本。")
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("列数与表头不一致。")
                image_name = row["image"].strip()
                if not image_name:
                    raise ValueError("image 路径为空。")
                image_path = Path(image_name).expanduser()
                if not image_path.is_absolute():
                    image_path = csv_path.parent / image_path
                image_path = image_path.resolve()
                encoded = _read_limited(image_path, MAX_IMAGE_BYTES - image_memory)
                image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                del encoded
                if image is None:
                    raise ValueError(f"无法解码图像：{image_path.name}")
                image_memory += image.nbytes
                if image_memory > MAX_IMAGE_BYTES:
                    raise ValueError("全部图像内存超过 256 MiB，请减少样本或分批整理数据。")
                pose = parse_pose(" ".join(row[field].strip() for field in columns[representation]),
                                  representation, translation_unit, angle_unit)
                assessment = estimate_board(image, camera)
                ok, png = cv2.imencode(".png", image)
                if not ok:
                    raise ValueError("图像无法编码为 PNG。")
                image_memory += png.nbytes
                if image_memory > MAX_IMAGE_BYTES:
                    raise ValueError("全部图像内存超过 256 MiB，请减少样本或分批整理数据。")
                sample = {"id": row.get("id", "").strip() or len(samples) + 1,
                          "role": row.get("role", "").strip() or "calibration",
                          "T_base_end": pose.tolist(), **assessment,
                          "source": str(image_path), "captured_at": "导入",
                          "image_sha256": image_digest(image)}
                samples.append(sample)
                images.append(png.tobytes())
                line_numbers.append(line)
            except (OSError, ValueError, TypeError, cv2.error) as exc:
                raise ValueError(f"CSV 第 {line} 行导入失败：{exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"CSV 第 {reader.line_num} 行格式无效：{exc}") from exc
    try:
        normalized = _normalize_samples(samples)
    except ValueError as exc:
        match = re.match(r"第 (\d+) 组样本", str(exc))
        line = line_numbers[int(match.group(1)) - 1] if match else reader.line_num
        raise ValueError(f"CSV 第 {line} 行导入失败：{exc}") from exc
    for sample, png in zip(normalized, images):
        sample["_image"] = png
    return normalized
