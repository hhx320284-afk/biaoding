"""仅在内存中绘制标定板、相机画面及采样提示。"""
from PySide6.QtCore import Qt, QRectF, QPointF, QSize, Signal, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QFont, QFontMetrics, QPainterPath
from PySide6.QtWidgets import QWidget, QSizePolicy, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QScrollArea
import cv2
import 界面主题 as 主题
from 核心算法.标定板 import pdf_layout, draw_print_page
from 画面裁剪 import validate_region

方案 = 主题.取方案('深空青')
# 画面上只有三种质量状态，用方案里的成功/警告/错误色做实心标签，文字统一取白色。
引导色 = {'red': 主题.状态色['错误'], 'amber': 主题.状态色['警告'], 'green': 主题.状态色['成功']}


def image_from_array(frame):
    if frame.ndim == 2:
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return image.copy()


class Preview(QWidget):
    doubleClicked = Signal()

    def __init__(self, board=False, parent=None, compact=False):
        super().__init__(parent)
        self.board = board
        self.compact = compact and not board
        self.empty_text = '打开相机，开始实时预览\n将标定板完整放入画面'
        self.image = None
        self.print_spec = None
        self.print_layout = None
        self.detection = None
        self.guide = {}
        self.display_region = None
        if self.compact:
            self.setMinimumSize(0, 0)
            policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            policy.setHeightForWidth(True)
            self.setSizePolicy(policy)
        else:
            self.setMinimumSize(420, 290)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame, detection=None, guide=None):
        previous_size = self.image.size() if self.image is not None else None
        self.print_spec = None
        self.print_layout = None
        self.image = image_from_array(frame) if frame is not None else None
        self.detection = detection
        self.guide = guide or {}
        if self.compact and previous_size != (self.image.size() if self.image is not None else None):
            self.updateGeometry()
        self.update()

    def set_print_page(self, spec, paper):
        layout = pdf_layout(spec, paper)
        self.print_spec, self.print_layout = spec, layout
        self.image = None
        self.update()

    def copy_from(self, other):
        previous_size = self.image.size() if self.image is not None else None
        previous_region = self.display_region
        self.image = other.image
        self.detection = other.detection
        self.guide = other.guide
        self.empty_text = other.empty_text
        self.display_region = (validate_region(other.display_region)
                               if other.display_region is not None else None)
        if self.compact and (previous_size != (self.image.size() if self.image is not None else None)
                             or previous_region != self.display_region):
            self.updateGeometry()
        self.update()

    def set_display_region(self, region):
        """仅改变绘制范围，不裁切 image 或改动检测坐标。"""
        self.display_region = validate_region(region) if region is not None else None
        if self.compact:
            self.updateGeometry()
        self.update()

    def _content_bounds(self):
        if self.image is None:
            return QRectF(0, 0, 16, 9), None
        width, height = self.image.width(), self.image.height()
        bounds = QRectF(0, 0, width, height)
        region = self.display_region
        if self.board or region is None or region['image_size'] != [width, height]:
            points = None
        else:
            points = [(x * width, y * height) for x, y in region['points']]
            left, right = min(p[0] for p in points), max(p[0] for p in points)
            top, bottom = min(p[1] for p in points), max(p[1] for p in points)
            bounds = QRectF(left, top, right - left, bottom - top)
        return bounds, points

    def heightForWidth(self, width):
        if not self.compact:
            return super().heightForWidth(width)
        bounds, _ = self._content_bounds()
        return max(1, round(width * bounds.height() / bounds.width()))

    def sizeHint(self):
        if self.compact:
            return QSize(840, self.heightForWidth(840))
        return super().sizeHint()

    def _display_geometry(self, box):
        width, height = self.image.width(), self.image.height()
        bounds, points = self._content_bounds()
        scale = min(box.width() / bounds.width(), box.height() / bounds.height())
        target = QRectF(box.center().x() - bounds.center().x() * scale,
                        box.center().y() - bounds.center().y() * scale,
                        width * scale, height * scale)
        return target, scale, points

    def mouseDoubleClickEvent(self, event):
        if not self.board and event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 紧凑预览不画矩形底色：多边形外由父页面显示。
        if not self.compact:
            painter.fillRect(self.rect(), QColor(方案['bg']))
        box = QRectF(self.rect()).adjusted(22, 22, -22, -22) if self.board else QRectF(self.rect())
        if self.board and self.print_spec is not None:
            page_w, page_h = self.print_layout['page_mm']
            scale = min(box.width() / page_w, box.height() / page_h)
            painter.save()
            # 相邻黑色矩形保持实心，避免屏幕抗锯齿在标签内部产生细白缝。
            painter.setRenderHint(QPainter.Antialiasing, False)
            painter.translate(box.center().x() - page_w * scale / 2,
                              box.center().y() - page_h * scale / 2)
            painter.scale(scale, scale)
            draw_print_page(painter, self.print_spec, self.print_layout)
            painter.restore()
            return
        show_banner = not self.compact and bool(self.guide) and self.guide.get('show_banner', True)
        if not self.board and show_banner:
            box.setBottom(box.bottom() - 95)
        if self.image is None:
            painter.setPen(QColor(方案['muted']))
            painter.setFont(QFont('Noto Sans CJK SC', 15))
            painter.drawText(box, Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
            return
        target, scale, crop_points = self._display_geometry(box)
        def pos(x, y):
            return QPointF(target.left() + x * scale, target.top() + y * scale)
        painter.save()
        if crop_points is not None:
            clip = QPainterPath(pos(*crop_points[0]))
            for point in crop_points[1:]:
                clip.lineTo(pos(*point))
            clip.closeSubpath()
            painter.setClipPath(clip, Qt.IntersectClip)
        painter.drawImage(target, self.image)
        if self.board:
            painter.restore()
            return
        # 角点、轮廓与目标提示用主色和白色画在视频上；质量状态交给下方实心标签。
        grid_color = QColor(方案['card'])
        grid_color.setAlpha(70)
        painter.setPen(QPen(grid_color, 1, Qt.DashLine))
        for i in (1, 2):
            x, y = target.left() + target.width() * i / 3, target.top() + target.height() * i / 3
            painter.drawLine(QPointF(x, target.top()), QPointF(x, target.bottom()))
            painter.drawLine(QPointF(target.left(), y), QPointF(target.right(), y))
        if self.detection is not None:
            accent = QColor(方案['primary'])
            painter.setPen(QPen(accent, 2))
            points = self.detection.outline.reshape(-1, 2)
            for i in range(len(points)):
                painter.drawLine(pos(*points[i]), pos(*points[(i + 1) % len(points)]))
            painter.setBrush(accent)
            painter.setPen(Qt.NoPen)
            for x, y in self.detection.image_points.reshape(-1, 2):
                painter.drawEllipse(pos(x, y), 2.3, 2.3)
            painter.setBrush(Qt.NoBrush)
        projected = self.guide.get('projected')
        if projected is not None and self.detection is not None:
            painter.setPen(QPen(QColor(方案['card']), 1.5))
            for observed, point in zip(self.detection.image_points, projected):
                at = pos(*point)
                painter.drawLine(pos(*observed), at)
                painter.drawLine(at + QPointF(-4, 0), at + QPointF(4, 0))
                painter.drawLine(at + QPointF(0, -4), at + QPointF(0, 4))
        destination = self.guide.get('target')
        if destination is not None:
            x = target.left() + destination[0] * target.width()
            y = target.top() + destination[1] * target.height()
            painter.setPen(QPen(QColor(方案['card']), 2, Qt.DashLine))
            painter.drawEllipse(QPointF(x, y), 28, 28)
            painter.setPen(QColor(方案['card']))
            painter.setFont(QFont('Noto Sans CJK SC', 10))
            painter.drawText(QRectF(x - 65, y + 30, 130, 30), Qt.AlignCenter, '移向此区域')
        # 采样提示横幅独立于图像裁剪，始终完整显示。
        painter.restore()
        if show_banner:
            banner = QRectF(14, self.height() - 95, self.width() - 28, 81)
            painter.setPen(QPen(QColor(方案['selected']), 1))
            painter.setBrush(QColor(方案['card']))
            painter.drawRoundedRect(banner, 10, 10)
            key = self.guide.get('color')
            known = key in 引导色
            status_color = QColor(引导色.get(key, 方案['primary']))
            text_color = QColor(方案['card'] if known else 方案['primary_fg'])
            status_text = self.guide.get('status', '')
            chip_font = QFont('Noto Sans CJK SC', 11, QFont.DemiBold)
            painter.setFont(chip_font)
            metrics = QFontMetrics(chip_font)
            chip_width = min(banner.width() - 28, metrics.horizontalAdvance(status_text) + 22)
            chip = QRectF(banner.left() + 14, banner.top() + 9, chip_width, 26)
            painter.setPen(Qt.NoPen)
            painter.setBrush(status_color)
            painter.drawRoundedRect(chip, 6, 6)
            painter.setPen(text_color)
            painter.drawText(chip.adjusted(11, 0, -11, 0), Qt.AlignVCenter | Qt.AlignLeft,
                             metrics.elidedText(status_text, Qt.ElideRight, int(chip_width) - 22))
            painter.setPen(QColor(方案['muted']))
            painter.setFont(QFont('Noto Sans CJK SC', 10))
            painter.drawText(banner.adjusted(16, 41, -16, -6), Qt.AlignVCenter | Qt.TextWordWrap,
                             self.guide.get('instruction', ''))


class PrintPreviewDialog(QDialog):
    """同一页的整页查看与放大查看，不生成磁盘预览文件。"""
    def __init__(self, spec, paper, parent=None):
        super().__init__(parent)
        self.setWindowTitle('标定板 PDF · 完整页面预览')
        self.resize(1100, 900)
        self.setMinimumSize(700, 500)
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel('与导出的 PDF 同页、同位置、同比例'), 1)
        self.zoom = QComboBox()
        self.zoom.addItems(['适合窗口', '放大 2 倍', '放大 3 倍'])
        bar.addWidget(self.zoom)
        close = QPushButton('关闭（Esc）')
        close.clicked.connect(self.reject)
        bar.addWidget(close)
        layout.addLayout(bar)
        self.area = QScrollArea()
        self.area.setAlignment(Qt.AlignCenter)
        self.page = Preview(board=True)
        self.page.setMinimumSize(1, 1)
        self.page.set_print_page(spec, paper)
        self.area.setWidget(self.page)
        layout.addWidget(self.area, 1)
        self.zoom.currentIndexChanged.connect(self.fit_page)
        self.zoom.setCurrentIndex(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.fit_page)

    def fit_page(self, *_):
        page_w, page_h = self.page.print_layout['page_mm']
        # 预留滚动条与纸张外边距，避免出现后再次改变缩放基准。
        available_w, available_h = max(1, self.area.width() - 64), max(1, self.area.height() - 64)
        scale = min(available_w / page_w, available_h / page_h) * (self.zoom.currentIndex() + 1)
        self.page.setFixedSize(round(page_w * scale) + 44, round(page_h * scale) + 44)


class Coverage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = [0] * 9
        self.setMinimumHeight(125)
        self.setMaximumHeight(145)

    def set_values(self, values):
        self.values = list(values)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for i, value in enumerate(self.values[:9]):
            x, y = i % 3, i // 3
            rect = QRectF(x * self.width() / 3 + 3, y * self.height() / 3 + 3,
                          self.width() / 3 - 6, self.height() / 3 - 6)
            if value:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(主题.状态色['成功']))
                painter.drawRoundedRect(rect, 6, 6)
                painter.setPen(QColor(方案['card']))
            else:
                painter.setPen(QPen(QColor(方案['描边']), 1))
                painter.setBrush(QColor(方案['selected']))
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
                painter.setPen(QColor(方案['muted']))
            painter.drawText(rect, Qt.AlignCenter, f'{value} 次' if value else '待覆盖')
