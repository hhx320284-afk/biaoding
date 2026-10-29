"""相机预览的多边形保留区域；只在内存中编辑显示范围。"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QFont
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

import 界面主题 as 主题

方案 = 主题.取方案('深空青')


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (abs(_cross(a, b, p)) <= 1e-12
            and min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)


def _intersects(a, b, c, d):
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > 1e-12 and ab_d < -1e-12 or ab_c < -1e-12 and ab_d > 1e-12)
            and (cd_a > 1e-12 and cd_b < -1e-12 or cd_a < -1e-12 and cd_b > 1e-12)):
        return True
    return any((_on_segment(a, b, c), _on_segment(a, b, d),
                _on_segment(c, d, a), _on_segment(c, d, b)))


def validate_region(region, image_size=None):
    """校验并返回独立的规范化区域；无区域应由调用者直接使用 None。"""
    if not isinstance(region, dict):
        raise ValueError('保留区域格式不正确。')
    size = region.get('image_size')
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
        raise ValueError('保留区域必须记录正确的原始图像宽、高。')
    if image_size is not None and tuple(size) != tuple(image_size):
        raise ValueError('保留区域的分辨率与当前相机画面不同，请重新圈选。')
    raw = region.get('points')
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise ValueError('至少需要 3 个轮廓顶点。')
    points = []
    for point in raw:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in point)):
            raise ValueError('轮廓顶点必须是归一化的二维坐标。')
        p = [float(point[0]), float(point[1])]
        if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in p):
            raise ValueError('轮廓顶点必须位于图像内。')
        if any(abs(p[0] - old[0]) < 1e-12 and abs(p[1] - old[1]) < 1e-12 for old in points):
            raise ValueError('轮廓顶点不能重复；应用时会自动闭合。')
        points.append(p)
    count = len(points)
    for i in range(count):
        # 相邻边不能沿原边反向重叠；普通的同向共线顶点仍允许。
        before, at, after = points[i - 1], points[i], points[(i + 1) % count]
        if (abs(_cross(before, at, after)) <= 1e-12
                and ((before[0] - at[0]) * (after[0] - at[0])
                     + (before[1] - at[1]) * (after[1] - at[1])) > 1e-12):
            raise ValueError('轮廓边不能重叠，请撤销后重新圈选。')
        for j in range(i + 1, count):
            if j == i + 1 or i == 0 and j == count - 1:
                continue
            if _intersects(points[i], points[(i + 1) % count], points[j], points[(j + 1) % count]):
                raise ValueError('轮廓边不能交叉，请撤销后沿外轮廓顺序圈选。')
    area2 = sum(points[i][0] * points[(i + 1) % count][1]
                - points[(i + 1) % count][0] * points[i][1] for i in range(count))
    if abs(area2) <= 1e-10:
        raise ValueError('保留区域面积为零，请圈出有面积的轮廓。')
    return {'points': points, 'image_size': [int(size[0]), int(size[1])]}


class _PolygonCanvas(QWidget):
    changed = Signal()

    def __init__(self, image, points=None, parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.points = [list(p) for p in (points or [])]
        self.hover_point = None
        self.setMinimumSize(420, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def image_rect(self):
        box = QRectF(self.rect()).adjusted(16, 16, -16, -16)
        scale = min(box.width() / self.image.width(), box.height() / self.image.height())
        return QRectF(box.center().x() - self.image.width() * scale / 2,
                      box.center().y() - self.image.height() * scale / 2,
                      self.image.width() * scale, self.image.height() * scale)

    def _image_point(self, position):
        target = self.image_rect()
        if not target.contains(position):
            return None
        return [min(1., max(0., (position.x() - target.left()) / target.width())),
                min(1., max(0., (position.y() - target.top()) / target.height()))]

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = self._image_point(event.position())
            if point is not None:
                self.points.append(point)
                self.changed.emit()
                self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.hover_point = self._image_point(event.position())
        self.update()

    def leaveEvent(self, event):
        self.hover_point = None
        self.update()
        super().leaveEvent(event)

    def undo(self):
        if self.points:
            self.points.pop()
            self.changed.emit()
            self.update()

    def reset(self):
        self.points.clear()
        self.changed.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(方案['bg']))
        target = self.image_rect()
        painter.drawImage(target, self.image)

        def pos(point):
            return QPointF(target.left() + point[0] * target.width(),
                           target.top() + point[1] * target.height())

        if len(self.points) >= 3:
            polygon = QPainterPath(pos(self.points[0]))
            for point in self.points[1:]:
                polygon.lineTo(pos(point))
            polygon.closeSubpath()
            exterior = QPainterPath()
            exterior.addRect(target)
            mask = QColor(方案['bg'])
            mask.setAlpha(175)
            painter.fillPath(exterior.subtracted(polygon), mask)
        if self.points:
            path = QPainterPath(pos(self.points[0]))
            for point in self.points[1:]:
                path.lineTo(pos(point))
            painter.setPen(QPen(QColor(方案['card']), 4))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(方案['primary']), 2))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(方案['card']), 1.5, Qt.DashLine))
            if len(self.points) >= 3:
                painter.drawLine(pos(self.points[-1]), pos(self.points[0]))
            if self.hover_point is not None:
                painter.drawLine(pos(self.points[-1]), pos(self.hover_point))
        painter.setFont(QFont('Noto Sans CJK SC', 10, QFont.DemiBold))
        for number, point in enumerate(self.points, 1):
            at = pos(point)
            painter.setPen(QPen(QColor(方案['card']), 2))
            painter.setBrush(QColor(方案['primary']))
            painter.drawEllipse(at, 5, 5)
            tag = QRectF(at.x() + 8, at.y() - 26, max(25, len(str(number)) * 10 + 12), 23)
            if tag.right() > target.right():
                tag.moveRight(at.x() - 8)
            if tag.top() < target.top():
                tag.moveTop(at.y() + 8)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(方案['card']))
            painter.drawRoundedRect(tag, 4, 4)
            painter.setPen(QColor(方案['bg']))
            painter.drawText(tag, Qt.AlignCenter, str(number))


class PolygonCropDialog(QDialog):
    """冻结完整原图，逐点圈选保留区域；接受后读取 region。"""
    def __init__(self, image: QImage, region=None, parent=None):
        super().__init__(parent)
        if not isinstance(image, QImage) or image.isNull():
            raise ValueError('当前没有可圈选的相机画面，请先打开相机。')
        self.region = None
        points = validate_region(region, [image.width(), image.height()])['points'] if region is not None else []
        self.setWindowTitle('圈选相机保留区域')
        self.resize(1080, 800)
        self.setMinimumSize(680, 500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        title = QLabel('沿轮廓圈出要保留的画面')
        title.setObjectName('页标题')
        layout.addWidget(title)
        note = QLabel('在完整原图的定格画面上，依次点击轮廓顶点，应用时自动闭合。\n裁剪后仅显示圈选部分，区域外透明，按原比例显示；标定仍使用完整原图。')
        note.setWordWrap(True)
        note.setObjectName('弱提示')
        layout.addWidget(note)
        self.canvas = _PolygonCanvas(image, points, self)
        layout.addWidget(self.canvas, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(24)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.undo_button = QPushButton('撤销上一个点')
        self.undo_button.setAutoDefault(False)
        self.undo_button.clicked.connect(self.canvas.undo)
        buttons.addWidget(self.undo_button)
        reset = QPushButton('重新圈选')
        reset.setAutoDefault(False)
        reset.clicked.connect(self.canvas.reset)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        cancel = QPushButton('取消')
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.apply_button = QPushButton('应用裁剪')
        self.apply_button.setObjectName('主按钮')
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply_region)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)
        self.canvas.changed.connect(self._update_state)
        self._update_state()

    def _candidate(self):
        return {'points': self.canvas.points,
                'image_size': [self.canvas.image.width(), self.canvas.image.height()]}

    def _set_status(self, text, error=False):
        self.status.setText(text)
        self.status.setObjectName('徽章错误' if error else '弱提示')
        主题.刷新样式(self.status)

    def _update_state(self):
        count = len(self.canvas.points)
        self.undo_button.setEnabled(count > 0)
        try:
            validate_region(self._candidate())
        except ValueError as error:
            self.apply_button.setEnabled(False)
            self._set_status(f'已选 {count} 个顶点 · {error}', error=count >= 3)
        else:
            self.apply_button.setEnabled(True)
            self._set_status(f'已选 {count} 个顶点 · 可继续添加，或点击“应用裁剪”（Enter）。')

    def apply_region(self):
        try:
            self.region = validate_region(self._candidate())
        except ValueError:
            self._update_state()
            return
        self.accept()
