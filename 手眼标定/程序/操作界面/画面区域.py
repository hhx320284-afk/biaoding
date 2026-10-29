"""复用相机标定的圈选和坐标变换，仅适配手眼预览与当前主题。"""
from __future__ import annotations

import importlib.util
import time

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

import 界面主题 as 主题
import 画面裁剪 as 相机裁剪
from 预览组件 import Preview

方案 = 主题.取方案('色盲友好橙')

# 独立模块实例避免改变相机标定窗口的深色画布；原文件保持只读。
_spec = importlib.util.spec_from_file_location('手眼圈选组件', 相机裁剪.__file__)
_crop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_crop)
_crop.方案 = 方案
validate_region = _crop.validate_region


class _LightPolygonCanvas(_crop._PolygonCanvas):
    def paintEvent(self, event):
        super().paintEvent(event)
        # 原组件的序号使用深色主题背景作文字，浅色主题改用正文色。
        painter = QPainter(self)
        painter.setFont(QFont('Noto Sans CJK SC', 10, QFont.DemiBold))
        painter.setPen(QColor(方案['text']))
        target = self.image_rect()
        for number, point in enumerate(self.points, 1):
            at = QPointF(target.left() + point[0] * target.width(),
                         target.top() + point[1] * target.height())
            tag = QRectF(at.x() + 8, at.y() - 26, max(25, len(str(number)) * 10 + 12), 23)
            if tag.right() > target.right():
                tag.moveRight(at.x() - 8)
            if tag.top() < target.top():
                tag.moveTop(at.y() + 8)
            painter.drawText(tag, Qt.AlignCenter, str(number))


_crop._PolygonCanvas = _LightPolygonCanvas
PolygonCropDialog = _crop.PolygonCropDialog


class RegionPreview(Preview):
    """原图、识别位置与目标共用相机标定的裁剪几何换算。"""
    def __init__(self, parent=None):
        super().__init__(parent=parent, compact=True)
        self.guide_target = None
        self.detected_center = None
        self.detected_at = 0.0

    def heightForWidth(self, width):
        bounds, _ = self._content_bounds()
        # 与绘图区的 12 px 内边距一致，避免边距使画面再次受高度限制。
        return max(25, round(24 + max(1, width - 24) * bounds.height() / bounds.width()))

    def clear(self):
        self.image = None
        self.update()

    def setText(self, text):
        self.empty_text = text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(方案['selected']))
        painter.drawRoundedRect(QRectF(self.rect()), 10, 10)
        box = QRectF(self.rect()).adjusted(12, 12, -12, -12)
        if self.image is None:
            painter.setPen(QColor(方案['muted']))
            painter.drawText(box, Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
            return
        target, scale, points = self._display_geometry(box)

        def pos(point):
            return QPointF(target.left() + point[0] * target.width(),
                           target.top() + point[1] * target.height())

        clip = QPainterPath()
        if points is None:
            clip.addRect(target)
        else:
            clip.moveTo(target.left() + points[0][0] * scale,
                        target.top() + points[0][1] * scale)
            for x, y in points[1:]:
                clip.lineTo(target.left() + x * scale, target.top() + y * scale)
            clip.closeSubpath()
        painter.save()
        painter.setClipRect(box)
        painter.setClipPath(clip, Qt.IntersectClip)
        painter.drawImage(target, self.image)
        outside = False
        if self.guide_target is not None:
            end = pos(self.guide_target)
            outside = not clip.contains(end)
            w, h = target.width(), target.height()
            rect = QRectF(end.x() - w * .12, end.y() - h * .12, w * .24, h * .24)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(方案['card']), 5))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QPen(QColor(方案['text']), 2))
            painter.drawRoundedRect(rect, 8, 8)
            if self.detected_center is not None and 0 <= time.monotonic() - self.detected_at < .8:
                start = pos(self.detected_center)
                delta = end - start
                length = (delta.x() ** 2 + delta.y() ** 2) ** .5
                if length > 20:
                    painter.setPen(QPen(QColor(方案['card']), 5))
                    painter.drawLine(start, end)
                    painter.setPen(QPen(QColor(方案['text']), 2))
                    painter.drawLine(start, end)
                    direction = delta / length
                    for side in (-1, 1):
                        tip = end - direction * 13 + QPointF(-direction.y(), direction.x()) * (side * 6)
                        painter.drawLine(end, tip)
            caption = QRectF(rect.left(), max(box.top(), rect.top() - 26), max(rect.width(), 100), 24)
            painter.fillRect(caption, QColor(方案['selected']))
            painter.setPen(QColor(方案['text']))
            painter.setFont(QFont('Noto Sans CJK SC', 10))
            painter.drawText(caption, Qt.AlignCenter, '板中心目标区')
        painter.restore()
        if outside:
            visible = box.intersected(QRectF(self.visibleRegion().boundingRect()))
            caption = QRectF(visible.left(), visible.bottom() - 38, visible.width(), 38)
            painter.fillRect(caption, QColor(方案['selected']))
            painter.setPen(QColor(方案['text']))
            painter.drawText(caption, Qt.AlignCenter | Qt.TextWordWrap,
                             '目标在保留区域外，请恢复完整画面查看引导')
