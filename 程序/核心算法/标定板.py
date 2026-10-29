"""标定板几何、内存预览与按真实毫米尺寸绘制的单页矢量 PDF。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import math

import cv2
import numpy as np


KIND_NAMES = {
    "charuco": "ChArUco 混合棋盘格",
    "aruco": "ArUco 标签网格",
    "aprilgrid": "AprilGrid · tag36h11",
    "chessboard": "普通棋盘格",
}

PRINT_CHECK_WIDTH_MM = 160.0
PRINT_CHECK_HEIGHT_MM = 34.0
_PRINT_HEADER_MM = 22.0


@dataclass(frozen=True)
class BoardSpec:
    kind: str = "charuco"
    columns: int = 5
    rows: int = 7
    square_mm: float = 24.0
    marker_mm: float = 17.0
    gap_mm: float = 6.0
    dictionary: str = "DICT_4X4_1000"
    first_id: int = 0

    @property
    def dictionary_name(self) -> str:
        return "DICT_APRILTAG_36h11" if self.kind == "aprilgrid" else self.dictionary

    @property
    def size_mm(self) -> tuple[float, float]:
        if self.kind in ("charuco", "chessboard"):
            return self.columns * self.square_mm, self.rows * self.square_mm
        return (self.columns * self.marker_mm + (self.columns - 1) * self.gap_mm,
                self.rows * self.marker_mm + (self.rows - 1) * self.gap_mm)

    @property
    def expected_count(self) -> int:
        if self.kind in ("charuco", "chessboard"):
            return (self.columns - 1) * (self.rows - 1)
        return self.columns * self.rows * 4

    def to_dict(self) -> dict:
        result = asdict(self)
        result["dictionary"] = self.dictionary_name
        return result

    def validate(self) -> None:
        if self.kind not in KIND_NAMES:
            raise ValueError("请选择受支持的标定板类型。")
        for name, value in (("列数", self.columns), ("行数", self.rows)):
            if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 40:
                raise ValueError(f"{name}必须是 2 至 40 的整数。")
        if self.expected_count < 16:
            raise ValueError("标定板至少需要 16 个角点；棋盘格建议至少 5 × 5 个方格。")
        for name, value in (("方格边长", self.square_mm), ("标签边长", self.marker_mm), ("标签间距", self.gap_mm)):
            if not math.isfinite(value) or not 0 < value <= 1000:
                raise ValueError(f"{name}必须大于 0 且不超过 1000 毫米。")
        if not isinstance(self.first_id, int) or isinstance(self.first_id, bool) or self.first_id < 0:
            raise ValueError("起始标签 ID 必须为非负整数。")
        if self.kind == "chessboard":
            return
        dictionary = _dictionary(self.dictionary_name)
        marker_count = (self.columns * self.rows // 2 if self.kind == "charuco"
                        else self.columns * self.rows)
        if self.first_id + marker_count > len(dictionary.bytesList):
            raise ValueError(f"标签 ID 超出 {self.dictionary_name} 的容量，请减少网格或降低起始 ID。")
        if self.kind == "charuco":
            if self.marker_mm >= self.square_mm:
                raise ValueError("ChArUco 的标签边长必须小于方格边长。")
            # 给标签外侧保留至少一格编码宽度，防止棋盘黑格与标签边框粘连。
            cell_mm = self.marker_mm / (dictionary.markerSize + 2)
            if (self.square_mm - self.marker_mm) / 2 < cell_mm * 0.7:
                raise ValueError("ChArUco 标签周围白边太窄，请减小标签边长或增大方格边长。")


PRESETS = {
    "ChArUco · A4 · 5 × 7 方格": BoardSpec(),
    "ChArUco · A4 · 7 × 10 方格": BoardSpec("charuco", 7, 10, 20, 14, 6),
    "ArUco · A4 · 5 × 7 标签": BoardSpec("aruco", 5, 7, 24, 20, 6),
    "AprilGrid · A4 · 6 × 6 标签": BoardSpec("aprilgrid", 6, 6, 24, 20, 6),
    "棋盘格 · A4 · 7 × 10 方格": BoardSpec("chessboard", 7, 10, 20, 14, 6),
    "ChArUco · A3 · 7 × 10 方格": BoardSpec("charuco", 7, 10, 28, 20, 8),
    "AprilGrid · A3 · 6 × 6 标签": BoardSpec("aprilgrid", 6, 6, 30, 30, 9),
}


@lru_cache(maxsize=24)
def _dictionary(name: str):
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        raise ValueError(f"不支持的标签字典：{name}")
    try:
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    except (cv2.error, TypeError) as exc:
        raise ValueError(f"不支持的标签字典：{name}") from exc


@lru_cache(maxsize=24)
def opencv_board(spec: BoardSpec):
    spec.validate()
    dictionary = _dictionary(spec.dictionary_name)
    count = spec.columns * spec.rows
    if spec.kind == "charuco":
        return cv2.aruco.CharucoBoard(
            (spec.columns, spec.rows), spec.square_mm / 1000, spec.marker_mm / 1000,
            dictionary, np.arange(spec.first_id, spec.first_id + count // 2, dtype=np.int32))
    if spec.kind in ("aruco", "aprilgrid"):
        return cv2.aruco.GridBoard(
            (spec.columns, spec.rows), spec.marker_mm / 1000, spec.gap_mm / 1000,
            dictionary, np.arange(spec.first_id, spec.first_id + count, dtype=np.int32))
    raise ValueError("普通棋盘格不使用标签板对象。")


def _black_rectangles(spec: BoardSpec):
    """生成毫米坐标的黑色矩形，屏幕预览和 PDF 共用同一几何。"""
    if spec.kind in ("charuco", "chessboard"):
        for row in range(spec.rows):
            for col in range(spec.columns):
                if (row + col) % 2 == 0:
                    yield col * spec.square_mm, row * spec.square_mm, spec.square_mm, spec.square_mm
    if spec.kind == "chessboard":
        return
    dictionary = _dictionary(spec.dictionary_name)
    board = opencv_board(spec)
    for marker_id, corners in zip(board.getIds().reshape(-1), board.getObjPoints()):
        x, y = np.asarray(corners, dtype=float)[0, :2] * 1000
        cell_count = dictionary.markerSize + 2
        bitmap = cv2.aruco.generateImageMarker(dictionary, int(marker_id), cell_count, borderBits=1)
        cell = spec.marker_mm / cell_count
        for row in range(cell_count):
            # 相邻黑格合成一条矩形，减少 PDF 路径数量，同时保持矢量边缘。
            col = 0
            while col < cell_count:
                if bitmap[row, col] != 0:
                    col += 1
                    continue
                start = col
                while col < cell_count and bitmap[row, col] == 0:
                    col += 1
                yield x + start * cell, y + row * cell, (col - start) * cell, cell


def make_board_image(spec: BoardSpec, max_side: int = 1000) -> np.ndarray:
    """带白边的灰度预览，仅存在于内存；预览像素不用于打印。"""
    spec.validate()
    if not 100 <= max_side <= 6000:
        raise ValueError("预览长边应在 100 至 6000 像素之间。")
    width, height = spec.size_mm
    margin = max(10.0, spec.square_mm if spec.kind in ("charuco", "chessboard") else spec.marker_mm)
    scale = max_side / max(width + margin * 2, height + margin * 2)
    canvas = np.full((round((height + margin * 2) * scale),
                      round((width + margin * 2) * scale)), 255, np.uint8)
    for x, y, w, h in _black_rectangles(spec):
        left, top = round((x + margin) * scale), round((y + margin) * scale)
        right, bottom = round((x + w + margin) * scale), round((y + h + margin) * scale)
        canvas[top:bottom, left:right] = 0
    return canvas


def pdf_layout(spec: BoardSpec, paper: str = "A4") -> dict:
    spec.validate()
    width, height = spec.size_mm
    quiet = max(10.0, spec.square_mm if spec.kind in ("charuco", "chessboard") else spec.marker_mm)
    required_w = max(PRINT_CHECK_WIDTH_MM, width + 2 * quiet)
    required_h = height + 2 * quiet + _PRINT_HEADER_MM + PRINT_CHECK_HEIGHT_MM
    if paper == "自定义":
        page_w, page_h = required_w, max(100.0, required_h)
    elif paper in ("A4", "A3"):
        portrait = (210.0, 297.0) if paper == "A4" else (297.0, 420.0)
        candidates = (portrait, tuple(reversed(portrait)))
        fit = [size for size in candidates if required_w <= size[0] and required_h <= size[1]]
        if not fit:
            raise ValueError(f"整块标定板及白边至少需要 {required_w:g} × {required_h:g} 毫米，"
                             f"{paper} 放不下。请减少尺寸、选更大纸张或选择自定义；不会缩放或裁切。")
        page_w, page_h = fit[0]
    else:
        raise ValueError("纸张请选择 A4、A3 或自定义。")
    return {"page_mm": (page_w, page_h), "board_mm": (width, height),
            "origin_mm": ((page_w - width) / 2,
                          _PRINT_HEADER_MM + quiet + (page_h - required_h) / 2),
            "quiet_mm": quiet, "footer_y_mm": page_h - PRINT_CHECK_HEIGHT_MM}


def _draw_mm_text(painter, rect, text, alignment=None) -> None:
    """按当前毫米缩放计算实际像素字号，避免小字号的 hinting 被放大。"""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QFont

    scale = painter.worldTransform().m11()
    painter.save()
    try:
        font = QFont("Noto Sans CJK SC")
        font.setPixelSize(max(1, round(3 * scale)))
        painter.setFont(font)
        painter.scale(1 / scale, 1 / scale)
        painter.drawText(QRectF(rect.x() * scale, rect.y() * scale,
                               rect.width() * scale, rect.height() * scale),
                         Qt.AlignmentFlag.AlignLeft if alignment is None else alignment, text)
    finally:
        painter.restore()


def draw_print_checks(painter, page_width_mm: float, footer_y_mm: float) -> None:
    """在毫米坐标系绘制打印校验尺；调用方负责背景和等比缩放。"""
    from PySide6.QtCore import QLineF, QRectF, Qt
    from PySide6.QtGui import QColor, QPen

    # 两边各保留 10 mm，连最小的自定义纸张也不会把刻线放在纸边。
    left = (page_width_mm - PRINT_CHECK_WIDTH_MM) / 2 + 10
    ruler_y = footer_y_mm + 12
    square_x, square_y = left + 120, footer_y_mm + 4

    painter.save()
    try:
        painter.setPen(QPen(QColor("black"), 0.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        _draw_mm_text(painter, QRectF(left, footer_y_mm, 108, 5), "打印尺寸校验：100 mm（10 cm）")
        painter.drawLine(QLineF(left, ruler_y, left + 100, ruler_y))
        for offset in range(0, 101, 10):
            tick = 3.0 if offset in (0, 50, 100) else 1.5
            painter.drawLine(QLineF(left + offset, ruler_y - tick,
                                   left + offset, ruler_y + 1))
        for offset, label in ((0, "0"), (50, "50"), (100, "100 mm")):
            _draw_mm_text(painter, QRectF(left + offset - 8, ruler_y + 2, 16, 5),
                      label, Qt.AlignmentFlag.AlignCenter)
        _draw_mm_text(painter, QRectF(left, footer_y_mm + 19, 110, 5), "量两端长刻线中心：应为 100 mm。")
        _draw_mm_text(painter, QRectF(left, footer_y_mm + 24, 110, 5), "打印选实际大小 / 100%；勿量屏幕预览。")
        _draw_mm_text(painter, QRectF(square_x - 10, footer_y_mm, 40, 4),
                  "横、纵校验", Qt.AlignmentFlag.AlignCenter)
        painter.drawRect(QRectF(square_x, square_y, 20, 20))
        _draw_mm_text(painter, QRectF(square_x, square_y, 20, 20),
                  "20 mm\n× 20 mm", Qt.AlignmentFlag.AlignCenter)
        _draw_mm_text(painter, QRectF(square_x - 10, footer_y_mm + 25, 40, 4),
                  "量两对边线中心", Qt.AlignmentFlag.AlignCenter)
    finally:
        painter.restore()


def draw_print_page(painter, spec: BoardSpec, layout: dict) -> None:
    """在毫米坐标系绘制整张打印纸；预览和 PDF 仅使用不同的等比缩放。"""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor

    page_w, page_h = layout["page_mm"]
    origin_x, origin_y = layout["origin_mm"]
    painter.save()
    try:
        painter.fillRect(QRectF(0, 0, page_w, page_h), QColor("white"))
        for x, y, w, h in _black_rectangles(spec):
            painter.fillRect(QRectF(origin_x + x, origin_y + y, w, h), QColor("black"))
        painter.setPen(QColor("black"))
        width, height = spec.size_mm
        lines = [f"{KIND_NAMES[spec.kind]}  |  {spec.columns} 列 × {spec.rows} 行  |  真实尺寸 1:1",
                 f"图案：{width:g} × {height:g} 毫米  ·  方格：{spec.square_mm:g} 毫米" if spec.kind in ("charuco", "chessboard")
                 else f"图案：{width:g} × {height:g} 毫米  ·  标签：{spec.marker_mm:g} 毫米  ·  间距：{spec.gap_mm:g} 毫米",
                 "棋盘格行列数指方格，内角点各少 1；请粘贴在平整、硬质底板。" if spec.kind == "chessboard"
                 else f"字典：{spec.dictionary_name}  ·  起始 ID：{spec.first_id}  ·  标签边长：{spec.marker_mm:g} 毫米",
                 "打印选择「实际大小 / 100%」，关闭适合页面；打印后用尺复核下方标尺。"]
        for index, line in enumerate(lines):
            _draw_mm_text(painter, QRectF(10, 2 + index * 5, page_w - 20, 5), line)
        draw_print_checks(painter, page_w, layout["footer_y_mm"])
    finally:
        painter.restore()


def export_board_pdf(spec: BoardSpec, path, paper: str = "A4") -> None:
    """只在显式导出调用时写文件；整板单页、矢量绘制、毫米尺寸 1:1。"""
    layout = pdf_layout(spec, paper)  # 打开文件前完成尺寸检查。
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QGuiApplication, QPageLayout, QPageSize, QPainter, QPdfWriter

    app = QGuiApplication.instance()
    if app is None:
        # 命令行导出不需要桌面服务；正常 UI 调用直接复用已有 QApplication。
        app = QGuiApplication(["标定板导出", "-platform", "offscreen"])
    output = Path(path)
    if not output.parent.is_dir():
        raise ValueError("导出目录不存在，请选择已有目录。")
    page_w, page_h = layout["page_mm"]
    writer = QPdfWriter(str(output))
    writer.setTitle(f"{KIND_NAMES[spec.kind]} · 真实尺寸 1:1")
    writer.setCreator("相机标定")
    writer.setResolution(2540)
    writer.setPageSize(QPageSize(QSizeF(page_w, page_h), QPageSize.Unit.Millimeter, "标定板打印纸"))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
    painter = QPainter(writer)
    if not painter.isActive():
        raise OSError(f"无法创建 PDF：{output}")
    try:
        painter.scale(100, 100)  # 2540 dpi 正好对应每毫米 100 个设备单位。
        draw_print_page(painter, spec, layout)
    finally:
        painter.end()
    # QPdfWriter 在析构时写入交叉引用表，返回前完成关闭。
    del writer
