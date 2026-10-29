"""三路相机独立内参标定工作台。采样只存在内存，保存由用户明确触发。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import io
import json
import logging
import os
import time
import uuid

import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, QTimer, QUrl, QEvent
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QStackedWidget, QFrame, QScrollArea, QFileDialog, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter, QDialog,
)
import 界面主题 as 主题
from 核心算法.标定板 import BoardSpec, PRESETS, export_board_pdf
from 核心算法.标定算法 import detect_board, calibrate
from 核心算法.采样引导 import Guide
from 核心算法.采样读取 import read_sample_folder
from 核心算法.相机接入 import CaptureController, discover_devices, probe_device
from 预览组件 import Preview, Coverage, PrintPreviewDialog
from 画面裁剪 import PolygonCropDialog, validate_region
from 实时验证 import ValidationPage

# 配色只取「深空青」一行，色值与控件样式来自界面主题.py，不在本文件另写色值。
方案 = 主题.取方案('深空青')
图标目录 = Path(__file__).resolve().parent


def 补充样式(方案):
    """主题未覆盖的控件（数值框、进度条、分割条、对话框）沿用同一套方案色。"""
    return f'''
QDialog, QMessageBox {{ background: {方案['bg']}; }}
QSpinBox, QDoubleSpinBox {{
    background: {方案['card']}; color: {方案['text']};
    border: 1px solid {方案['描边']}; border-radius: 9px;
    min-height: 20px; padding: 8px 10px;
    selection-background-color: {方案['selected']}; selection-color: {方案['text']};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {方案['primary']}; padding: 7px 9px; }}
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {方案['muted']}; background: {方案['bg']}; border-color: {方案['selected']};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right;
    width: 20px; border: 0; background: {方案['selected']};
    border-top-right-radius: 8px; margin: 2px 2px 0 0;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 20px; border: 0; background: {方案['selected']};
    border-bottom-right-radius: 8px; margin: 0 2px 2px 0;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{图标目录}/上调箭头.svg"); width: 9px; height: 6px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{图标目录}/下拉箭头.svg"); width: 9px; height: 6px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url("{图标目录}/下拉箭头.svg"); width: 10px; height: 7px; }}
QCheckBox {{ spacing: 8px; padding: 4px 0; }}
QPushButton#主按钮:disabled, QPushButton#危险按钮:disabled {{
    background: {方案['selected']}; color: {方案['muted']};
    border: 1px solid {方案['selected']};
}}
QPushButton#链接按钮:disabled {{ color: {方案['muted']}; }}
QProgressBar {{
    background: {方案['selected']}; border: 1px solid {方案['selected']};
    border-radius: 6px; min-height: 10px;
}}
QProgressBar::chunk {{ background: {方案['primary']}; border-radius: 5px; }}
QSplitter::handle {{ background: {方案['selected']}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {方案['描边']}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:vertical {{ background: {方案['描边']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:hover {{ background: {方案['primary']}; }}
QToolTip {{
    background: {方案['card']}; color: {方案['text']};
    border: 1px solid {方案['描边']}; padding: 6px 8px;
}}
'''


STYLE = 主题.生成样式(方案) + 补充样式(方案)

# 旧代码里的文字角色映射到主题已写好的对象名。
文字角色 = {'title': '页标题', 'subtitle': '页说明', 'section': '卡片标题',
            'muted': '弱提示', 'metric': '指标值'}


def label(text='', kind=None):
    result = QLabel(text)
    result.setWordWrap(True)
    if kind:
        result.setObjectName(文字角色.get(kind, kind))
    return result


def button(text, callback, primary=False):
    result = 主题.按钮(text, '主按钮' if primary else None)
    result.clicked.connect(callback)
    return result


def card(title):
    frame, layout = 主题.卡片(title)
    layout.setSpacing(12)
    return frame, layout


def scroll(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setFrameShape(QFrame.Shape.NoFrame)
    return area


class Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = Signals()

    def run(self):
        try:
            self.signals.done.emit(self.function())
        except Exception as error:
            logging.getLogger('相机标定').exception('后台操作失败')
            self.signals.failed.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self, project, logger, log_path):
        super().__init__()
        QApplication.instance().setStyle('Fusion')
        self.project, self.logger, self.log_path = Path(project), logger, Path(log_path)
        self.sessions = [self.new_session() for _ in range(3)]
        self.index = 0
        self.active = False
        self.automatic = False
        self.busy = False
        self.hardware_busy = False
        self.analyzing = False
        self.generation = 0
        self.jobs = {}
        self.controller = CaptureController()
        self.snapshot = None
        self.devices = []
        self.last_analysis = 0.0
        self.last_detection = None
        self.last_guidance = {}
        self.guidance_at = 0.0
        self.dark_frame = False
        self.fullscreen_dialog = None
        self.loading = False
        self.setWindowTitle('相机标定 · 三相机工作台')
        self.resize(1380, 940)
        # 侧栏占用宽度后，1240 是采样页四个操作按钮不被裁切的最小宽度。
        self.setMinimumSize(1240, 800)
        # 设在应用级别，保证提示框、对话框和无父窗口的浮层也用同一套深空青样式。
        QApplication.instance().setStyleSheet(STYLE)
        self.build_ui()
        self.update_board()
        self.update_state()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(35)
        QTimer.singleShot(200, self.scan)

    @staticmethod
    def new_session():
        return dict(device='', mode=None, model='pinhole', bypass=False, samples=[], result=None,
                    guide=Guide(), size=None, camera_name='', actual={}, saved=False,
                    display_regions={}, imported_from='')

    @property
    def session(self):
        return self.sessions[self.index]

    def build_ui(self):
        root = QWidget()
        root.setObjectName('页面')
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(14)

        self.sidebar = self.build_sidebar()
        outer.addWidget(self.sidebar)

        right = QVBoxLayout()
        right.setSpacing(12)
        self.header = self.build_header()
        self.metrics_widget = self.build_metrics()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.build_board_page())
        self.pages.addWidget(self.build_capture_page())
        self.pages.addWidget(self.build_result_page())
        self.validation = ValidationPage(self, label, button, card, scroll)
        self.pages.addWidget(self.validation)
        self.log_card = self.build_log()
        right.addWidget(self.header)
        right.addWidget(self.metrics_widget)
        right.addWidget(self.pages, 1)
        right.addWidget(self.log_card)
        outer.addLayout(right, 1)
        self.set_page(0)

    def build_sidebar(self):
        bar = QFrame()
        bar.setObjectName('侧栏')
        bar.setFixedWidth(196)
        column = QVBoxLayout(bar)
        column.setContentsMargins(12, 16, 12, 16)
        column.setSpacing(8)

        brand = QLabel('相机标定')
        brand.setObjectName('卡片标题')
        column.addWidget(brand)
        column.addWidget(label('三路内参 · 实时验证', 'muted'))
        column.addWidget(主题.分隔线())

        group = QLabel('相机')
        group.setObjectName('分组名')
        column.addWidget(group)
        self.slot_buttons = []
        for i in range(3):
            item = button(f'相机 {i+1}   ·   未采样', lambda checked=False, n=i: self.select_session(n))
            item.setObjectName('导航项')
            item.setCheckable(True)
            column.addWidget(item)
            self.slot_buttons.append(item)
        column.addWidget(主题.分隔线())

        group = QLabel('页面')
        group.setObjectName('分组名')
        column.addWidget(group)
        self.nav = []
        for i, text in enumerate(['01  准备标定板', '02  实时采样', '03  标定结果', '04  实时验证']):
            item = button(text, lambda checked=False, n=i: self.set_page(n))
            item.setObjectName('导航项')
            item.setCheckable(True)
            column.addWidget(item)
            self.nav.append(item)
        column.addStretch(1)
        column.addWidget(主题.分隔线())
        column.addWidget(label('三路独立标定；未保存的采样与结果仅本次运行有效。', 'muted'))
        return bar

    def build_header(self):
        head = QWidget()
        row = QHBoxLayout(head)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        words = QVBoxLayout()
        words.setSpacing(4)
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_row.addWidget(label('相机标定', 'title'))
        self.state_badge = 主题.徽章('待打开')
        title_row.addWidget(self.state_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)
        words.addLayout(title_row)
        words.addWidget(label('标定板 → 自动采样 → 标定结果 → 实时验证；采样与参考只保存在内存。', 'subtitle'))
        row.addLayout(words, 1)

        logs = button('打开日志', lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_path))))
        logs.setToolTip(f'日志文件：{self.log_path}')
        row.addWidget(logs, 0, Qt.AlignmentFlag.AlignVCenter)
        return head

    def build_metrics(self):
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(30)
        self.camera_metric = self.add_metric(row, '当前相机', '1 / 3')
        self.count_metric = self.add_metric(row, '采样', '0', '张')
        self.rms_metric = self.add_metric(row, '重投影 RMS', '—', 'px')
        self.coverage_metric = self.add_metric(row, '画面覆盖', '0 / 9', '格')
        row.addStretch(1)
        return box

    @staticmethod
    def add_metric(row, name, value, unit=''):
        block = 主题.指标(name, value, unit)
        row.addWidget(block)
        return block.findChild(QLabel, '指标值')

    def build_log(self):
        card_frame, column = 主题.卡片()
        head = QHBoxLayout()
        head.setSpacing(10)
        title = QLabel('运行记录')
        title.setObjectName('卡片标题')
        self.status = label('准备好标定板后，进入“实时采样”。未保存的数据仅在本次运行有效。', 'muted')
        clear = button('清空记录', lambda: self.log_view.clear())
        clear.setObjectName('链接按钮')
        head.addWidget(title)
        head.addWidget(self.status, 1)
        head.addWidget(clear)
        column.addLayout(head)
        self.log_view = 主题.日志面板((), 84)
        self.log_view.setMaximumBlockCount(400)
        self.log_view.setToolTip('记录本次运行的启动、扫描、采样、保存与错误；不记录图像内容。')
        column.addWidget(self.log_view)
        return card_frame

    def log(self, text):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def build_board_page(self):
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(18)
        pane, layout = card('标定板设置')
        pane.setMaximumWidth(370)
        self.board_controls = QWidget()
        form = QFormLayout(self.board_controls)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(12)
        self.presets = QComboBox()
        self.presets.addItems(list(PRESETS))
        form.addRow('常用规格', self.presets)
        self.board_type = QComboBox()
        for name, value in [('ChArUco 混合棋盘格', 'charuco'), ('ArUco 标签网格', 'aruco'), ('AprilGrid 标签网格', 'aprilgrid'), ('普通棋盘格', 'chessboard')]:
            self.board_type.addItem(name, value)
        form.addRow('板型', self.board_type)
        self.columns, self.rows = QSpinBox(), QSpinBox()
        for box, value in [(self.columns, 5), (self.rows, 7)]:
            box.setRange(3, 20)
            box.setValue(value)
        form.addRow('列数（横向）', self.columns)
        form.addRow('行数（纵向）', self.rows)
        self.square, self.marker, self.gap = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for box, value, minimum in [(self.square, 24, 5), (self.marker, 17, 3), (self.gap, 6, 1)]:
            box.setRange(minimum, 200)
            box.setDecimals(2)
            box.setSuffix(' mm')
            box.setValue(value)
        form.addRow('方格边长', self.square)
        form.addRow('标签边长', self.marker)
        form.addRow('标签间距', self.gap)
        self.dictionary = QComboBox()
        self.dictionary.addItems(['DICT_4X4_1000', 'DICT_5X5_1000', 'DICT_6X6_1000', 'DICT_APRILTAG_36h11'])
        form.addRow('编码字典', self.dictionary)
        self.first_id = QSpinBox()
        self.first_id.setRange(0, 900)
        form.addRow('起始标签 ID', self.first_id)
        layout.addWidget(self.board_controls)
        self.board_details = label('', 'muted')
        layout.addWidget(self.board_details)
        self.paper = QComboBox()
        self.paper.addItems(['A4', 'A3', '自定义'])
        paper_row = QFormLayout()
        paper_row.addRow('PDF 纸张', self.paper)
        layout.addLayout(paper_row)
        self.export_board = button('导出 1∶1 标定板 PDF', self.save_board, True)
        layout.addWidget(label('打印选择“实际大小 / 100%”，关闭“适应页面”。用直尺检查下方横尺为 100 mm、方框宽高均为 20 mm，再贴在平整硬板上。', 'muted'))
        layout.addWidget(button('重新设置标定板…', self.reset_all))
        layout.addStretch()
        left_scroll = scroll(pane)
        left_scroll.setFixedWidth(370)
        row.addWidget(left_scroll)
        right, detail = card('')
        preview_bar = QHBoxLayout()
        preview_bar.addWidget(label('完整 PDF 页面预览', 'section'), 1)
        self.enlarge_board_button = button('放大预览', self.show_board_preview)
        preview_bar.addWidget(self.enlarge_board_button)
        detail.addLayout(preview_bar)
        self.board_preview = Preview(board=True)
        detail.addWidget(self.board_preview, 1)
        detail.addWidget(label('整张纸统一缩放，与导出的 PDF 布局一致；可放大查看文字与刻线。\n打印选实际大小 / 100%，实测横尺 100 mm、方框宽高各 20 mm（量线中心）。', 'muted'))
        actions = QHBoxLayout()
        actions.addWidget(self.export_board)
        actions.addWidget(button('下一步：连接相机与采样', lambda: self.set_page(1)))
        detail.addLayout(actions)
        row.addWidget(right, 1)
        self.presets.currentIndexChanged.connect(self.apply_preset)
        for widget in (self.board_type, self.dictionary):
            widget.currentIndexChanged.connect(self.update_board)
        for widget in (self.columns, self.rows, self.square, self.marker, self.gap, self.first_id):
            widget.valueChanged.connect(self.update_board)
        self.paper.currentTextChanged.connect(self.update_board)
        return page

    def show_board_preview(self):
        if self.board_preview.print_spec is not None:
            PrintPreviewDialog(self.spec(), self.paper.currentText(), self).exec()

    def build_capture_page(self):
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(18)
        left, settings = card('当前相机')
        left.setMaximumWidth(350)
        self.device_combo, self.mode_combo = QComboBox(), QComboBox()
        self.device_combo.setMinimumWidth(240)
        self.device_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.setMinimumContentsLength(12)
        self.device_combo.setToolTip('按设备名称与连接路径选择实际相机；型号需支持 Linux V4L2 / UVC。')
        settings.addWidget(self.device_combo)
        self.refresh_button = button('刷新设备', self.scan)
        settings.addWidget(self.refresh_button)
        self.device_path = label('正在查找视频设备…', 'muted')
        settings.addWidget(self.device_path)
        settings.addWidget(label('采集分辨率与帧率', 'muted'))
        settings.addWidget(self.mode_combo)
        self.model_combo = QComboBox()
        self.model_combo.addItem('普通镜头 · 针孔与径向畸变', 'pinhole')
        self.model_combo.addItem('鱼眼镜头 · OpenCV 鱼眼模型', 'fisheye')
        settings.addWidget(label('镜头模型（与相机品牌无关）', 'muted'))
        settings.addWidget(self.model_combo)
        self.bypass = QCheckBox('此路暂不标定，仅预览')
        settings.addWidget(self.bypass)
        self.open_button = button('打开相机', self.toggle_camera, True)
        settings.addWidget(self.open_button)
        self.actual_label = label('原始图像用于标定，画面仅按比例显示。', 'muted')
        settings.addWidget(self.actual_label)
        settings.addWidget(label('自动采样 · 仅保存在内存', 'section'))
        self.count_label = label('0 张', 'metric')
        settings.addWidget(self.count_label)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        settings.addWidget(self.progress)
        self.guidance_text = label('等待相机画面', 'muted')
        settings.addWidget(self.guidance_text)
        settings.addWidget(label('画面覆盖 · 标定板中心位置', 'section'))
        self.coverage = Coverage()
        settings.addWidget(self.coverage)
        settings.addWidget(label('绿：可用　黄：等待 / 调整　红：不可采样\n点击“保存采样照片”才写入文件。', 'muted'))
        settings.addWidget(label('采样条件', 'section'))
        form = QFormLayout()
        self.min_samples, self.max_samples = QSpinBox(), QSpinBox()
        self.min_samples.setRange(12, 120)
        self.min_samples.setValue(20)
        self.max_samples.setRange(20, 120)
        self.max_samples.setValue(45)
        self.sharpness = QDoubleSpinBox()
        self.sharpness.setRange(5, 1000)
        self.sharpness.setValue(60)
        self.sharpness.setDecimals(0)
        self.stable = QDoubleSpinBox()
        self.stable.setRange(0.3, 4)
        self.stable.setValue(0.8)
        self.stable.setSuffix(' 秒')
        self.rms_limit = QDoubleSpinBox()
        self.rms_limit.setRange(0.1, 5)
        self.rms_limit.setValue(1.0)
        self.rms_limit.setSuffix(' px')
        form.addRow('最少样本', self.min_samples)
        form.addRow('样本上限', self.max_samples)
        form.addRow('清晰度门限', self.sharpness)
        form.addRow('保持稳定', self.stable)
        form.addRow('误差参考线', self.rms_limit)
        settings.addLayout(form)
        settings.addWidget(label('满足数量、覆盖与姿态多样性后自动求解。门限是采样辅助，最终以重投影评估为准。', 'muted'))
        settings.addStretch()
        sidebar = scroll(left)
        sidebar.setFixedWidth(350)
        row.addWidget(sidebar)
        right = QWidget()
        detail = QVBoxLayout(right)
        detail.setContentsMargins(0, 0, 0, 0)
        detail.setSpacing(12)
        preview_bar = QHBoxLayout()
        self.preview_source = label('尚未打开相机', 'section')
        preview_bar.addWidget(self.preview_source, 1)
        self.import_samples_button = button('导入采样文件夹', self.import_samples)
        self.import_samples_button.setToolTip('选择包含“采样信息.json”、原始照片和同名角点记录的文件夹；无需打开相机。')
        preview_bar.addWidget(self.import_samples_button)
        self.fullscreen_button = button('全屏查看', self.show_preview_fullscreen)
        self.fullscreen_button.setToolTip('也可双击画面进入全屏，按 Esc 返回。')
        preview_bar.addWidget(self.fullscreen_button)
        detail.addLayout(preview_bar)
        crop_bar = QHBoxLayout()
        self.crop_button = button('圈选保留区域', self.edit_display_region)
        self.restore_crop_button = button('恢复完整画面', self.restore_display_region)
        self.save_crop_button = button('保存区域', self.save_display_region)
        self.load_crop_button = button('加载区域', self.load_display_region)
        for item in (self.crop_button, self.restore_crop_button, self.save_crop_button, self.load_crop_button):
            crop_bar.addWidget(item)
        crop_bar.addStretch()
        detail.addLayout(crop_bar)
        self.crop_status = label('打开相机后，可沿轮廓逐点圈选保留区域。', 'muted')
        detail.addWidget(self.crop_status)
        self.preview = Preview(compact=True)
        self.preview.doubleClicked.connect(self.show_preview_fullscreen)
        self.image_scroll = self.make_image_scroll(self.preview)
        self.image_scroll.setMinimumHeight(180)
        detail.addWidget(self.image_scroll, 1)
        self.preview_guidance = label('', 'muted')
        self.preview_guidance.setTextFormat(Qt.PlainText)
        self.preview_guidance.hide()
        detail.addWidget(self.preview_guidance)
        self.metrics_label = label('清晰度 —    ·    检测点 —    ·    当前分辨率 —', 'muted')
        detail.addWidget(self.metrics_label)
        bar = QHBoxLayout()
        self.auto_button = button('开始自动采样', self.toggle_auto, True)
        self.solve_button = button('计算当前样本', self.solve)
        self.save_samples_button = button('保存采样照片', self.save_samples)
        self.clear_button = button('清空当前采样…', self.clear_samples)
        for item in (self.auto_button, self.solve_button, self.save_samples_button, self.clear_button):
            bar.addWidget(item)
        detail.addLayout(bar)
        row.addWidget(right, 1)
        self.device_combo.currentIndexChanged.connect(self.device_changed)
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        self.model_combo.currentIndexChanged.connect(self.model_changed)
        self.bypass.toggled.connect(self.bypass_changed)
        for widget in (self.min_samples, self.max_samples, self.sharpness, self.stable, self.rms_limit):
            widget.valueChanged.connect(self.configure_guide)
        return page

    def show_preview_fullscreen(self):
        if self.fullscreen_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle('相机标定 · 全屏预览')
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(16, 12, 16, 16)
            toolbar = QHBoxLayout()
            self.fullscreen_source = label('', 'section')
            toolbar.addWidget(self.fullscreen_source, 1)
            toolbar.addWidget(button('退出全屏（Esc）', dialog.reject))
            layout.addLayout(toolbar)
            self.fullscreen_preview = Preview(compact=True)
            self.fullscreen_preview.doubleClicked.connect(dialog.reject)
            self.fullscreen_image_scroll = self.make_image_scroll(self.fullscreen_preview)
            layout.addWidget(self.fullscreen_image_scroll, 1)
            self.fullscreen_guidance = label('', 'muted')
            self.fullscreen_guidance.setTextFormat(Qt.PlainText)
            layout.addWidget(self.fullscreen_guidance)
            self.fullscreen_dialog = dialog
        self.fullscreen_source.setText(self.preview_source.text())
        self.fullscreen_preview.copy_from(self.preview)
        self.fullscreen_guidance.setText(self.preview_guidance.text())
        self.fullscreen_guidance.setVisible(not self.preview_guidance.isHidden())
        self.fullscreen_dialog.showFullScreen()
        self.fullscreen_dialog.raise_()
        self.fullscreen_dialog.activateWindow()
        self.resize_previews()

    def make_image_scroll(self, preview):
        area = QScrollArea()
        area.setWidgetResizable(False)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setAlignment(Qt.AlignCenter)
        area.setWidget(preview)
        area.viewport().installEventFilter(self)
        return area

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Resize:
            areas = (getattr(self, 'image_scroll', None),
                     getattr(self, 'fullscreen_image_scroll', None))
            if any(area is not None and watched is area.viewport() for area in areas):
                QTimer.singleShot(0, self.resize_previews)
        return super().eventFilter(watched, event)

    @staticmethod
    def fit_preview(preview, area):
        # 在视口内完整显示整幅画面并保持宽高比，不出现内部滚动条。
        viewport = area.viewport()
        width = max(1, viewport.width())
        height = max(1, viewport.height())
        bounds, _ = preview._content_bounds()
        fitted = max(1, min(width, round(height * bounds.width() / bounds.height())))
        image_height = preview.heightForWidth(fitted)
        if preview.width() != fitted or preview.height() != image_height:
            preview.setFixedSize(fitted, image_height)

    def resize_previews(self):
        self.fit_preview(self.preview, self.image_scroll)
        if self.fullscreen_dialog is not None and self.fullscreen_dialog.isVisible():
            self.fit_preview(self.fullscreen_preview, self.fullscreen_image_scroll)

    def set_live_frame(self, frame, detection=None, guide=None):
        self.preview.set_frame(frame, detection, guide)
        guide = guide or {}
        text = '\n'.join(str(guide[key]) for key in ('status', 'instruction') if guide.get(key))
        self.preview_guidance.setText(text)
        self.preview_guidance.setVisible(bool(text) and frame is not None and guide.get('show_banner', True))
        self.refresh_display_region()

    def display_region_key(self):
        image = self.preview.image
        if image is None or image.isNull():
            return None
        return (self.session['device'], image.width(), image.height())

    def refresh_display_region(self):
        key = self.display_region_key()
        region = self.session['display_regions'].get(key) if key else None
        if self.preview.display_region != region:
            self.preview.set_display_region(region)
        if self.fullscreen_dialog is not None and self.fullscreen_dialog.isVisible():
            self.fullscreen_preview.copy_from(self.preview)
            self.fullscreen_guidance.setText(self.preview_guidance.text())
            self.fullscreen_guidance.setVisible(not self.preview_guidance.isHidden())
        self.crop_button.setEnabled(key is not None)
        self.load_crop_button.setEnabled(key is not None)
        self.restore_crop_button.setEnabled(region is not None)
        self.save_crop_button.setEnabled(region is not None)
        if key is None:
            text = '打开相机后，可沿轮廓逐点圈选保留区域。'
        elif region:
            text = f'相机 {self.index + 1} · 已保留 {len(region["points"])} 点轮廓 · 完整显示保留区域 · 标定仍用完整原图。'
        else:
            text = f'相机 {self.index + 1} · 完整显示原图 · 圈选只改变显示。'
        self.crop_status.setText(text)
        self.resize_previews()

    def edit_display_region(self):
        key = self.display_region_key()
        if key is None:
            return
        slot = self.index
        dialog = PolygonCropDialog(self.preview.image,
                                   self.session['display_regions'].get(key), self)
        if dialog.exec() != QDialog.Accepted:
            return
        if self.index != slot or self.display_region_key() != key:
            self.error('相机或分辨率已变化，请在当前画面重新圈选。')
            return
        self.session['display_regions'][key] = validate_region(dialog.region, key[1:])
        self.refresh_display_region()
        self.notice('已应用当前相机的显示区域；仅本次运行有效，需要保留时请点击“保存区域”。')
        self.logger.info('相机 %s 应用多边形显示区域', self.index + 1)

    def restore_display_region(self):
        key = self.display_region_key()
        self.session['display_regions'].pop(key, None)
        self.refresh_display_region()
        self.notice('已恢复当前相机的完整画面，已保存的区域文件不受影响。')
        self.logger.info('相机 %s 恢复完整预览', self.index + 1)

    def save_display_region(self):
        key = self.display_region_key()
        region = self.session['display_regions'].get(key)
        if region is None:
            return
        record = {'type': '相机预览显示区域', 'version': 1, 'device': key[0],
                  'camera_name': self.session['camera_name'], 'region': validate_region(region)}
        try:
            path = self.choose_file('保存相机显示区域', f'相机{self.index + 1}显示区域.json', 'JSON 文件 (*.json)')
            if path:
                self.write_atomic(path, json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8'))
                self.notice(f'显示区域已保存：{path.name}。下次打开相机后可点击“加载区域”。')
                self.logger.info('用户保存相机显示区域')
        except Exception as error:
            self.logger.exception('保存显示区域失败')
            self.error(str(error))

    def load_display_region(self):
        key = self.display_region_key()
        if key is None:
            return
        slot = self.index
        value, _ = QFileDialog.getOpenFileName(self, '加载相机显示区域', str(self.project / '结果'), 'JSON 文件 (*.json)')
        if not value:
            return
        try:
            record = json.loads(Path(value).read_text(encoding='utf-8'))
            if not isinstance(record, dict) or record.get('type') != '相机预览显示区域' or record.get('version') != 1:
                raise ValueError('请选择通过“保存区域”保存的显示区域文件。')
            if record.get('device') != key[0]:
                raise ValueError('区域文件记录的相机设备与当前设备不同，请选择对应相机，或在当前画面重新圈选。')
            region = validate_region(record.get('region'), key[1:])
            if self.index != slot or self.display_region_key() != key:
                raise ValueError('相机或分辨率已变化，请在当前画面重新加载。')
            self.session['display_regions'][key] = region
            self.refresh_display_region()
            self.notice('已加载当前相机的显示区域；原始采样、标定参数和检测坐标保持不变。')
            self.logger.info('用户加载相机显示区域')
        except Exception as error:
            self.logger.exception('加载显示区域失败')
            self.error(str(error))

    def build_result_page(self):
        page, layout = card('重投影评估')
        self.result_title = label('尚未计算标定结果', 'metric')
        layout.addWidget(self.result_title)
        self.result_description = label('完成采样后，在这里查看总体误差、每张照片的误差和相机参数。', 'muted')
        layout.addWidget(self.result_description)
        self.result_parameters = label('', 'muted')
        self.result_parameters.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.result_parameters)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['采样序号', '有效角点', '重投影 RMS / px', '检查建议'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.table)
        self.reprojection = Preview()
        self.reprojection.empty_text = '选择一张样本，查看重投影对比'
        split.addWidget(self.reprojection)
        split.setSizes([560, 580])
        layout.addWidget(split, 1)
        self.table.itemSelectionChanged.connect(self.show_reprojection)
        row = QHBoxLayout()
        self.remove_button = button('剔除所选样本并重新计算', self.remove_samples)
        self.export_result_button = button('导出标定结果 JSON', self.save_result, True)
        self.export_csv_button = button('导出误差 CSV', self.save_csv)
        self.validate_result_button = button('用此结果实时验证', self.validate_current_result)
        row.addWidget(self.remove_button)
        row.addStretch()
        row.addWidget(self.validate_result_button)
        row.addWidget(self.export_csv_button)
        row.addWidget(self.export_result_button)
        layout.addLayout(row)
        layout.addWidget(label('此处计算各相机内参与畸变；不包含相机间外参或机械臂手眼标定。低重投影误差仍需结合覆盖、姿态和独立验证判断。', 'muted'))
        return page

    def set_page(self, index):
        if index == 3 and (self.busy or self.hardware_busy):
            for n, item in enumerate(self.nav):
                item.setChecked(n == self.pages.currentIndex())
            self.notice('请等待当前计算或设备探测完成，再进入实时验证。')
            return
        if self.pages.currentIndex() == 3 and index != 3:
            self.validation.stop()
        if index == 3:
            if self.active:
                self.stop_camera()
            if self.fullscreen_dialog is not None:
                self.fullscreen_dialog.hide()
        self.pages.setCurrentIndex(index)
        for n, item in enumerate(self.nav):
            item.setChecked(n == index)
        if index == 3:
            self.validation.enter()

    def validate_current_result(self):
        if self.session['result'] is None or self.busy or self.hardware_busy:
            return
        try:
            record = self.metadata()
            record['calibration'] = self.session['result']
            self.validation.set_source(record, f'相机 {self.index + 1} · 本次标定结果')
            self.set_page(3)
        except (ValueError, TypeError, KeyError) as error:
            self.error(str(error))

    def spec(self):
        return BoardSpec(kind=self.board_type.currentData(), columns=self.columns.value(), rows=self.rows.value(),
                         square_mm=self.square.value(), marker_mm=self.marker.value(), gap_mm=self.gap.value(),
                         dictionary=self.dictionary.currentText(), first_id=self.first_id.value())

    def apply_preset(self, index):
        name = list(PRESETS)[index]
        spec = PRESETS[name]
        self.loading = True
        self.board_type.setCurrentIndex(self.board_type.findData(spec.kind))
        for widget, value in zip((self.columns, self.rows, self.square, self.marker, self.gap),
                                 (spec.columns, spec.rows, spec.square_mm, spec.marker_mm, spec.gap_mm)):
            widget.setValue(value)
        self.dictionary.setCurrentText(spec.dictionary_name)
        self.first_id.setValue(spec.first_id)
        self.paper.setCurrentText('A3' if 'A3' in name else 'A4')
        self.loading = False
        self.update_board()

    def update_board(self, *_):
        if self.loading:
            return
        kind = self.board_type.currentData()
        self.square.setEnabled(kind in ('charuco', 'chessboard'))
        self.marker.setEnabled(kind != 'chessboard')
        self.gap.setEnabled(kind in ('aruco', 'aprilgrid'))
        self.dictionary.setEnabled(kind in ('charuco', 'aruco'))
        self.first_id.setEnabled(kind != 'chessboard')
        if kind == 'aprilgrid':
            self.dictionary.blockSignals(True)
            self.dictionary.setCurrentText('DICT_APRILTAG_36h11')
            self.dictionary.blockSignals(False)
        self.generation += 1
        try:
            spec = self.spec()
            spec.validate()
            w, h = spec.size_mm
            units = '标签' if kind in ('aruco', 'aprilgrid') else '实际方格'
            self.board_details.setText(f'板面 {w:g} × {h:g} mm（另留白边）\n行列数表示{units}数。'
                                       + ('内角点数各比方格数少 1。' if kind in ('charuco', 'chessboard') else ''))
            self.board_valid = True
            try:
                self.board_preview.set_print_page(spec, self.paper.currentText())
                self.export_board.setEnabled(True)
                self.enlarge_board_button.setEnabled(True)
            except ValueError as error:
                self.board_preview.set_frame(None)
                self.board_preview.empty_text = str(error)
                self.board_preview.update()
                self.export_board.setEnabled(False)
                self.enlarge_board_button.setEnabled(False)
        except Exception as error:
            self.board_preview.set_frame(None)
            self.board_preview.empty_text = str(error)
            self.board_details.setText(str(error))
            self.export_board.setEnabled(False)
            self.enlarge_board_button.setEnabled(False)
            self.board_valid = False
        if hasattr(self, 'auto_button'):
            self.update_state()

    def notice(self, text):
        self.status.setText(text)
        self.log(text)

    def error(self, text):
        self.status.setText(text)
        self.log('错误：' + text)
        QMessageBox.warning(self, '操作未完成', f'{text}\n\n日志位置：{self.log_path}')

    def run_job(self, function, done, failed=None):
        job = Job(function)
        key = id(job)
        self.jobs[key] = job
        def success(value):
            self.jobs.pop(key, None)
            done(value)
        def failure(message):
            self.jobs.pop(key, None)
            (failed or self.error)(message)
        job.signals.done.connect(success)
        job.signals.failed.connect(failure)
        QThreadPool.globalInstance().start(job)

    @staticmethod
    def camera_display_name(device):
        name = device.name.strip()
        if any(word in name.lower() for word in ('integrated', 'built-in', 'built in')):
            name = '电脑内置摄像头'
        else:
            parts = name.split(': ', 1)
            if len(parts) == 2 and parts[0] == parts[1]:
                name = parts[0]
        return f'{name} · {os.path.basename(os.path.realpath(device.path))}'

    def scan(self):
        if self.active or self.hardware_busy:
            return
        self.hardware_busy = True
        self.update_state()
        self.notice('正在读取相机设备列表…')
        def done(devices):
            self.hardware_busy = False
            self.devices = devices
            current = self.session['device']
            self.device_combo.blockSignals(True)
            self.device_combo.clear()
            for item in devices:
                self.device_combo.addItem(self.camera_display_name(item), item.path)
                self.device_combo.setItemData(self.device_combo.count() - 1,
                    f'{item.name}\n{os.path.realpath(item.path)}', Qt.ToolTipRole)
            match = self.device_combo.findData(current)
            self.device_combo.setCurrentIndex(match if match >= 0 else (0 if devices else -1))
            self.device_combo.blockSignals(False)
            self.notice(f'发现 {len(devices)} 个视频采集节点；同一设备可能提供多个视频流。' if devices else '没有发现可用视频节点，请连接相机并刷新。')
            self.device_changed()
            self.update_state()
        def failed(message):
            self.hardware_busy = False
            self.update_state()
            self.error(message)
        self.run_job(discover_devices, done, failed)

    def device_changed(self, *_):
        if self.loading or self.active:
            return
        if self.session['imported_from'] and self.session['samples']:
            self.restore_imported_source()
            return
        path = self.device_combo.currentData() or ''
        if self.session['device'] != path:
            self.session['mode'] = None
        self.session['device'] = path
        self.session['camera_name'] = self.device_combo.currentText()
        selected = self.session['camera_name']
        description = '这是电脑内置摄像头；外接鱼眼请选择 LRCP。\n' if '电脑内置' in selected and any('LRCP' in d.name for d in self.devices) else ''
        self.device_path.setText(description + (f'设备节点：{os.path.realpath(path)}' if path else '未发现视频设备。专用工业相机需确认型号与 SDK。'))
        self.device_combo.setToolTip(f'{selected}\n{path}' if path else '设备尚未连接')
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.blockSignals(False)
        if not path:
            return
        self.hardware_busy = True
        slot = self.index
        self.update_state()
        def done(result):
            self.hardware_busy = False
            if self.index == slot and self.session['device'] == path:
                previous = self.session['mode']
                self.mode_combo.blockSignals(True)
                self.mode_combo.clear()
                modes, _ = result
                modes = sorted(modes, key=lambda m: (m.pixel_format != 'MJPG', abs(m.width - 1280), abs(m.fps - 30)))
                for m in modes:
                    value = (m.width, m.height, m.fps, m.pixel_format)
                    self.mode_combo.addItem(f'{m.width} × {m.height} · {m.fps:g} fps · {m.pixel_format}', value)
                found = self.mode_combo.findData(previous)
                self.mode_combo.setCurrentIndex(found if found >= 0 else 0)
                self.mode_combo.blockSignals(False)
                self.mode_changed()
            self.update_state()
        def failed(message):
            self.hardware_busy = False
            self.update_state()
            self.error(message)
        self.run_job(lambda: probe_device(path), done, failed)

    def restore_imported_source(self):
        """保留离线采样的设备身份和模式，不用当前枚举的第一台设备覆盖。"""
        session = self.session
        self.device_combo.blockSignals(True)
        self.device_combo.setCurrentIndex(self.device_combo.findData(session['device']))
        self.device_combo.blockSignals(False)
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        if session['mode']:
            w, h, fps, fmt = session['mode']
            self.mode_combo.addItem(f'{w} × {h} · {fps:g} fps · {fmt}', session['mode'])
        self.mode_combo.blockSignals(False)
        self.device_path.setText(f'采样设备：{session["camera_name"]}\n可直接离线计算，无需打开相机。')
        self.device_path.setToolTip(session['device'])
        w, h = session['size']
        self.actual_label.setText(f'采样原图：{w} × {h}\n已恢复原始角点与镜头模型')

    def mode_changed(self, *_):
        if not self.loading:
            self.session['mode'] = self.mode_combo.currentData()
        self.update_state()

    def model_changed(self, *_):
        if not self.loading:
            self.session['model'] = self.model_combo.currentData()
        self.update_state()

    def bypass_changed(self, checked):
        self.session['bypass'] = checked
        if checked:
            self.automatic = False
        self.update_state()

    def configure_guide(self, *_):
        self.max_samples.setMinimum(max(20, max(len(s['samples']) for s in self.sessions)))
        if self.min_samples.value() > self.max_samples.value():
            self.max_samples.setValue(self.min_samples.value())
        for session in self.sessions:
            guide = session['guide']
            guide.min_samples = self.min_samples.value()
            guide.max_samples = self.max_samples.value()
            guide.sharpness = self.sharpness.value()
            guide.stable_seconds = self.stable.value()
            guide.target_rms = self.rms_limit.value()
        self.render_result()

    def select_session(self, index):
        if index == self.index or self.busy or self.hardware_busy:
            # 侧栏按钮是可勾选按钮，提前返回时把勾选状态同步回当前相机。
            self.update_state()
            return
        if self.pages.currentIndex() == 3:
            self.set_page(1)
        self.stop_camera()
        self.index = index
        self.loading = True
        self.model_combo.setCurrentIndex(self.model_combo.findData(self.session['model']))
        self.bypass.setChecked(self.session['bypass'])
        self.device_combo.setCurrentIndex(self.device_combo.findData(self.session['device']))
        self.loading = False
        self.device_changed()
        self.coverage.set_values(getattr(self.session['guide'], 'coverage', [0] * 9))
        self.render_result()
        self.update_state()

    def toggle_camera(self):
        if self.active:
            self.stop_camera()
            return
        mode = self.mode_combo.currentData()
        if not mode:
            self.error('请先选择有效的视频模式。')
            return
        try:
            width, height, fps, fmt = mode
            self.controller.start(self.session['device'], int(width), int(height), float(fps), fmt)
            self.active = True
            self.dark_frame = False
            self.generation += 1
            self.notice('正在打开相机，等待原始视频画面…')
            self.logger.info('打开相机 %s：%s，设备 %s，镜头模型 %s', self.index + 1,
                             self.session['camera_name'], os.path.realpath(self.session['device']), self.session['model'])
        except Exception as error:
            self.logger.exception('相机打开失败')
            self.error(str(error))
        self.update_state()

    def stop_camera(self):
        if self.active:
            self.logger.info('相机 %s 停止取流', self.index + 1)
        self.automatic = False
        self.active = False
        self.generation += 1
        self.snapshot = None
        self.last_detection = None
        self.last_guidance = {}
        self.dark_frame = False
        self.controller.stop()
        self.set_live_frame(None)
        self.guidance_text.setText('相机已关闭，采样缓存仍保留在内存')
        self.notice('相机已关闭。未保存的数据仅在本次运行有效。')
        self.update_state()

    def toggle_auto(self):
        if not self.active or self.session['bypass']:
            return
        if not self.automatic and self.session['guide']._coverage_state()['complete']:
            if len(self.session['samples']) >= self.max_samples.value():
                self.notice('已达到采样上限。可提高上限后补充样本，或剔除低质量样本重新采集。')
                return
            self.min_samples.setValue(min(self.max_samples.value(), len(self.session['samples']) + 5))
        self.automatic = not self.automatic
        self.configure_guide()
        self.generation += 1
        self.notice('自动采样已开启：移动标定板，按画面提示保持稳定。照片仅缓存在内存。' if self.automatic else '已暂停自动采样，仍可预览。')
        self.logger.info('相机 %s %s自动采样', self.index + 1, '开启' if self.automatic else '暂停')
        self.update_state()

    def poll(self):
        if not self.active:
            return
        try:
            events = self.controller.poll()
        except Exception as error:
            self.logger.exception('相机取流失败')
            self.stop_camera()
            self.error(str(error))
            return
        for event in events:
            if event['type'] == 'opened':
                self.session['actual'] = event['actual']
            elif event['type'] in ('closed', 'error'):
                self.stop_camera()
                if event['type'] == 'error':
                    self.error(event['message'])
                return
            elif event['type'] == 'frame':
                frame = event['frame']
                size = (frame.shape[1], frame.shape[0])
                if self.session['samples'] and tuple(self.session['size']) != size:
                    self.stop_camera()
                    self.error('相机实际分辨率与已有采样不同。请恢复原分辨率，或清空当前采样后重新采集。')
                    return
                self.session['size'] = size
                if self.snapshot is None:
                    self.notice('实时预览中。点击“开始自动采样”后，将按质量与姿态条件收集到内存。')
                self.snapshot = event
                self.actual_label.setText(f'实际画面：{size[0]} × {size[1]}\n{event["measured_fps"]:.1f} 帧 / 秒 · 原始像素')
                pixels = frame[::16, ::16]
                dark = float(pixels.mean()) < 28 and float(pixels.std()) < 2.0
                if dark != self.dark_frame:
                    self.dark_frame = dark
                    self.generation += 1
                    self.last_detection = None
                    self.last_guidance = {}
                    if dark:
                        self.session['guide'].evaluate(frame, None)
                self.update_state()
                if dark:
                    instruction = '检查镜头遮挡、隐私盖和环境照明；确认上方设备名称是否正确。'
                    if '电脑内置' in self.session['camera_name']:
                        instruction = '当前打开的是电脑内置摄像头。请先关闭相机，再选择外接鱼眼 LRCP。' if any('LRCP' in d.name for d in self.devices) else '当前打开的是电脑内置摄像头。检查隐私盖，或关闭后选择需要的外接相机。'
                    self.set_live_frame(frame, guide=dict(color='red', status='相机返回近乎纯黑画面', instruction=instruction))
                    self.guidance_text.setText('画面近乎纯黑，暂不能采样')
                    self.metrics_label.setText('已收到视频帧，但缺少有效图像细节；请按画面提示检查。')
                    continue
                if self.session['bypass']:
                    self.set_live_frame(frame)
                    self.guidance_text.setText('仅预览 · 此路暂不标定')
                    continue
                now = time.monotonic()
                if now - self.guidance_at < 0.6:
                    self.set_live_frame(frame, self.last_detection, self.last_guidance)
                else:
                    self.set_live_frame(frame, guide=dict(color='amber', status='正在检测画面', instruction='保持画面清晰；检测完成后显示采样建议。'))
                if not self.analyzing and not self.busy and now - self.last_analysis > 0.16 and self.board_valid:
                    self.analyzing = True
                    self.last_analysis = now
                    generation = self.generation
                    spec = self.spec()
                    captured_at = event['captured_at']
                    def done(payload, generation=generation, captured_at=captured_at):
                        self.analyzing = False
                        if generation == self.generation and self.active:
                            self.analyzed(*payload, captured_at)
                    def failed(message, generation=generation):
                        self.analyzing = False
                        if generation != self.generation or not self.active:
                            return
                        self.automatic = False
                        self.update_state()
                        self.notice('检测未完成：' + message)
                    self.run_job(lambda frame=frame, spec=spec: (frame, detect_board(frame, spec)), done, failed)

    def analyzed(self, frame, detection, captured_at):
        guide = self.session['guide']
        state = guide.evaluate(frame, detection)
        if not self.automatic and state['accepted']:
            state = {**state, 'status': '当前视角可用', 'instruction': '点击“开始自动采样”，保持标定板稳定后自动采集。'}
        if self.automatic and state['accepted'] and detection is not None:
            ok, encoded = cv2.imencode('.png', frame)
            if not ok:
                self.automatic = False
                self.error('图像编码失败，已暂停自动采样。')
                return
            used = sum(len(s['image']) for session in self.sessions for s in session['samples'])
            if used + encoded.nbytes > 512 * 1024 * 1024:
                self.automatic = False
                self.update_state()
                self.error('采样缓存已达到 512 MiB。请先保存照片与结果，再清空不再使用的采样。')
                return
            if not guide.accept(detection, frame.shape):
                return
            self.session['samples'].append(dict(object_points=detection.object_points.copy(),
                image_points=detection.image_points.copy(), ids=detection.ids.copy(), image=encoded.tobytes(),
                captured_at=captured_at, metrics=state.get('metrics', {})))
            self.session['result'] = None
            self.session['saved'] = False
            self.logger.info('相机 %s 接受样本 %s', self.index + 1, len(self.session['samples']))
            state = {**state, 'status': f'已采集第 {len(self.session["samples"])} 张',
                     'instruction': '请换一个位置、距离或倾角，等待下一次自动采集。', 'color': 'green'}
        self.last_detection = detection
        self.last_guidance = state
        self.guidance_at = time.monotonic()
        self.guidance_text.setText(state['status'])
        self.coverage.set_values(state.get('coverage', [0] * 9))
        metrics = state.get('metrics', {})
        sharp = metrics.get('sharpness', metrics.get('blur', 0))
        self.metrics_label.setText(f'清晰度 {sharp:.0f}    ·    检测点 {len(detection.image_points) if detection else 0}'
                                   f'    ·    {frame.shape[1]} × {frame.shape[0]}')
        self.progress.setValue(int(state.get('progress', 0) * 100))
        self.update_state()
        if self.automatic and state.get('complete'):
            self.automatic = False
            self.notice('采样覆盖与姿态条件已满足，正在自动计算标定结果…')
            self.solve(automatic=True)
        elif self.automatic and metrics.get('paused'):
            self.automatic = False
            self.notice('达到采样上限，但覆盖或姿态条件不足。请按提示调整，必要时提高上限或清空后重采。')
            self.update_state()

    def solve(self, checked=False, automatic=False):
        if self.busy or len(self.session['samples']) < 8:
            self.error('至少需要 8 张有效样本才能计算；建议遵循自动采样流程。')
            return
        self.automatic = False
        self.busy = True
        self.generation += 1
        session = self.session
        samples = [{key: value for key, value in sample.items() if key in ('object_points', 'image_points')} for sample in session['samples']]
        size, model = tuple(session['size']), session['model']
        self.notice('正在优化相机参数并计算每张照片的重投影误差…')
        self.update_state()
        def done(result):
            self.busy = False
            result['sample_count'] = len(samples)
            result['reference_rms_px'] = self.rms_limit.value()
            if not session['guide']._coverage_state()['complete']:
                result.setdefault('warnings', []).append('当前数量、位置覆盖或尺度/倾角多样性尚未满足自动采样条件，请补充不同姿态。')
            session['result'] = result
            session['saved'] = False
            self.render_result()
            self.update_state()
            if automatic and result['rms'] > self.rms_limit.value() and len(samples) < self.max_samples.value() and self.active:
                self.min_samples.setValue(min(self.max_samples.value(), len(samples) + 5))
                self.automatic = True
                self.update_state()
                self.set_page(1)
                self.notice(f'本批 RMS {result["rms"]:.3f} px 高于参考线，继续自动补采不同姿态；达到上限后停止。')
            else:
                self.set_page(2)
                self.notice('标定计算完成。请检查误差与提示，确认后导出结果。')
            self.logger.info('相机 %s 求解完成，RMS %.5f px', self.index + 1, result['rms'])
        def failed(message):
            self.busy = False
            self.update_state()
            self.error('标定未成功：' + message + '\n建议增加不同倾角与位置的清晰样本，再检查镜头模型。')
        self.run_job(lambda: calibrate(samples, size, model=model), done, failed)

    def render_result(self):
        result = self.session['result']
        self.table.setRowCount(0)
        if result is None:
            self.result_title.setText('尚未计算标定结果')
            self.result_description.setText('采样后计算；每路相机的样本和结果相互独立。')
            self.result_parameters.clear()
            self.reprojection.set_frame(None)
            return
        rms, limit = result['rms'], self.rms_limit.value()
        self.result_title.setText(f'整体重投影 RMS   {rms:.3f} px')
        warnings = result.get('warnings', [])
        text = f'当前误差参考线 {limit:.2f} px · {len(self.session["samples"])} 张样本。'
        text += '训练集误差低于参考线。' if rms <= limit else '误差超过参考线，请检查高误差样本并补拍。'
        text += '\n' + ('；'.join(warnings) if warnings else '请结合不同位置、尺度和倾角检查，不以单个误差数值判定全部准确性。')
        text += '\n评估方式：本批采样的拟合重投影误差；未以独立照片验证。'
        self.result_description.setText(text)
        k = np.asarray(result['K'])
        d = np.asarray(result['D']).reshape(-1)
        self.result_parameters.setText(f'模型：{"鱼眼" if self.session["model"] == "fisheye" else "普通针孔"}    '
            f'分辨率：{self.session["size"][0]} × {self.session["size"][1]}    '
            f'fx={k[0,0]:.3f}    fy={k[1,1]:.3f}    cx={k[0,2]:.3f}    cy={k[1,2]:.3f}\n'
            f'畸变系数 D = {np.array2string(d, precision=6)}')
        errors = result['per_view_rms']
        self.table.setRowCount(len(errors))
        for i, error in enumerate(errors):
            values = [str(i+1), str(len(self.session['samples'][i]['image_points'])), f'{error:.4f}',
                      '建议检查 / 重拍' if error > limit else '低于参考线']
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                if error > limit:
                    # 深色表格上用加粗代替红字，避免状态色直接压在暗底上难以辨认。
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(i, j, item)
        if errors:
            self.table.selectRow(0)

    def show_reprojection(self):
        result = self.session['result']
        row = self.table.currentRow()
        if result is None or row < 0 or row >= len(self.session['samples']):
            return
        from types import SimpleNamespace
        sample = self.session['samples'][row]
        frame = cv2.imdecode(np.frombuffer(sample['image'], np.uint8), cv2.IMREAD_COLOR)
        points = sample['image_points']
        detection = SimpleNamespace(image_points=points, outline=cv2.convexHull(points.astype(np.float32)))
        guide = dict(color='green', status=f'第 {row+1} 张 · RMS {result["per_view_rms"][row]:.4f} px',
                     instruction='绿点：检测角点；黄十字：重投影位置；连线表示残差。',
                     projected=result['reprojected_points'][row])
        self.reprojection.set_frame(frame, detection, guide)

    def update_state(self):
        if not hasattr(self, 'table'):
            return
        count = len(self.session['samples'])
        source = self.session['camera_name'] or '请选择相机'
        state = '正在预览' if self.active else '离线采样' if self.session['imported_from'] else '待打开'
        self.preview_source.setText(f'{state}：{source}')
        if self.fullscreen_dialog is not None and self.fullscreen_dialog.isVisible():
            self.fullscreen_source.setText(self.preview_source.text())
        locked = self.active or bool(count) or self.busy
        self.device_combo.setEnabled(not locked and not self.hardware_busy)
        self.mode_combo.setEnabled(not locked and not self.hardware_busy)
        self.model_combo.setEnabled(not locked)
        self.refresh_button.setEnabled(not self.active and not self.hardware_busy and not count and not self.busy)
        self.open_button.setText('关闭相机' if self.active else '打开相机')
        known_device = any(d.path == self.session['device'] for d in self.devices)
        self.open_button.setEnabled(not self.hardware_busy and not self.busy and (self.active or (self.mode_combo.currentData() is not None and known_device)))
        self.import_samples_button.setEnabled(not self.busy and not self.hardware_busy and not self.automatic)
        self.auto_button.setText('暂停自动采样' if self.automatic else '开始自动采样')
        self.auto_button.setEnabled(self.active and self.snapshot is not None and not self.busy and not self.session['bypass'] and getattr(self, 'board_valid', False))
        self.solve_button.setEnabled(count >= 8 and not self.busy and not self.session['bypass'])
        self.save_samples_button.setEnabled(count > 0 and not self.automatic and not self.busy)
        self.clear_button.setEnabled(count > 0 and not self.busy)
        self.count_label.setText(f'{count} 张  /  至少 {self.min_samples.value()} 张')
        self.board_controls.setEnabled(not self.active and not self.busy and not any(s['samples'] for s in self.sessions))
        for box in (self.min_samples, self.max_samples, self.sharpness, self.stable, self.rms_limit):
            box.setEnabled(not self.automatic and not self.busy)
        self.bypass.setEnabled(not self.busy)
        for i, item in enumerate(self.slot_buttons):
            item.setChecked(i == self.index)
            item.setEnabled(not self.busy and not self.hardware_busy)
            s = self.sessions[i]
            state = f'RMS {s["result"]["rms"]:.2f} px' if s['result'] else f'{len(s["samples"])} 张' if s['samples'] else '仅预览' if s['bypass'] else '未采样'
            item.setText(f'相机 {i+1}   ·   {state}')
        ready = self.session['result'] is not None and not self.busy and not self.automatic
        self.export_result_button.setEnabled(ready)
        self.export_csv_button.setEnabled(ready)
        self.remove_button.setEnabled(ready)
        self.validate_result_button.setEnabled(ready)
        self.update_metrics()
        self.refresh_display_region()

    def update_metrics(self):
        session = self.session
        self.camera_metric.setText(f'{self.index + 1} / 3')
        self.count_metric.setText(str(len(session['samples'])))
        result = session['result']
        self.rms_metric.setText(f'{result["rms"]:.3f}' if result else '—')
        coverage = getattr(session['guide'], 'coverage', [0] * 9)
        self.coverage_metric.setText(f'{sum(1 for value in coverage if value)} / 9')
        if self.busy:
            text, state = '计算中', ''
        elif result is not None:
            text, state = '有结果', '成功'
        elif self.active:
            text, state = '预览中', '成功'
        elif session['imported_from']:
            text, state = '离线样本', ''
        elif self.hardware_busy:
            text, state = '探测设备', ''
        else:
            text, state = '待打开', ''
        self.state_badge.setText(text)
        主题.设徽章状态(self.state_badge, state)

    def reset_all(self):
        if self.busy:
            return
        if any(s['samples'] for s in self.sessions):
            answer = QMessageBox.question(self, '重新设置标定板', '这会清空三路相机内存中的采样与标定结果。已导出的文件不受影响。是否继续？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.stop_camera()
        for s in self.sessions:
            s.update(samples=[], result=None, size=None, saved=False, imported_from='')
            s['guide'].reset()
        self.coverage.set_values([0]*9)
        self.render_result()
        self.update_state()
        self.set_page(0)

    def clear_samples(self):
        answer = QMessageBox.question(self, '清空当前采样', '清空此路相机内存中的采样与标定结果？已保存的文件不会删除。', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self.automatic = False
        self.generation += 1
        self.session.update(samples=[], result=None, saved=False, imported_from='')
        self.session['guide'].reset()
        self.coverage.set_values([0]*9)
        self.progress.setValue(0)
        self.render_result()
        self.update_state()

    def remove_samples(self):
        selected = sorted({item.row() for item in self.table.selectedIndexes()}, reverse=True)
        if not selected:
            self.notice('请先在表格中选择需要剔除的样本。')
            return
        if len(self.session['samples']) - len(selected) < 8:
            self.error('剔除后不足 8 张；请先补充采样。')
            return
        for index in selected:
            self.session['samples'].pop(index)
        self.session['result'] = None
        self.session['guide'].reset()
        # 根据保留角点重建引导历史，避免删样本后累计数量失真。
        from types import SimpleNamespace
        for sample in self.session['samples']:
            points = sample['image_points']
            detection = SimpleNamespace(image_points=points, object_points=sample['object_points'], ids=sample['ids'],
                outline=cv2.convexHull(points.astype(np.float32)).reshape(-1, 2), found_count=len(points), expected_count=len(points))
            self.session['guide'].accept(detection, (self.session['size'][1], self.session['size'][0], 3))
        self.solve()

    def safe_path(self, path):
        path = Path(path).expanduser().resolve()
        protected = (self.project.parent / '鱼眼相机安装调试').resolve()
        if path == protected or protected in path.parents:
            raise ValueError('此项目不能写入受保护的“鱼眼相机安装调试”目录，请选择本项目“结果”或其他位置。')
        return path

    def choose_file(self, title, filename, file_filter):
        value, _ = QFileDialog.getSaveFileName(self, title, str(self.project / '结果' / filename), file_filter)
        return self.safe_path(value) if value else None

    @staticmethod
    def write_atomic(path, data):
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with open(temporary, 'xb') as output:
                output.write(data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save_board(self):
        try:
            path = self.choose_file('导出完整标定板 PDF', '标定板.pdf', 'PDF 文档 (*.pdf)')
            if not path:
                return
            export_board_pdf(self.spec(), path, paper=self.paper.currentText())
            self.notice(f'已导出 {path.name}。按实际大小 100% 打印，实测横尺 100 mm、方框宽高各 20 mm。')
            self.logger.info('用户导出标定板 PDF')
        except Exception as error:
            self.logger.exception('导出标定板失败')
            self.error(str(error))

    def import_samples(self):
        if self.busy or self.hardware_busy or self.automatic:
            return
        selected = QFileDialog.getExistingDirectory(
            self, '导入采样文件夹（内含采样信息.json）', str(self.project / '结果'))
        if not selected:
            return
        self.stop_camera()
        self.busy = True
        self.update_state()
        self.notice('正在读取原始照片与角点记录，校验分辨率和镜头模型…')
        # 旧缓存保留到完整读取并确认成功，失败不会清空已有样本。
        available = 512 * 1024 * 1024 - sum(
            len(sample['image']) for session in self.sessions for sample in session['samples'])

        def failed(message):
            self.busy = False
            self.update_state()
            self.error('导入未完成：' + message)

        def done(batch):
            from types import SimpleNamespace
            try:
                spec = batch['board']
                if any(s is not self.session and s['samples'] for s in self.sessions) and spec.to_dict() != self.spec().to_dict():
                    raise ValueError('导入文件使用的标定板与其他相机已有样本不同。请先保存并清空其他相机采样，再导入。')
                fields = ((self.columns, spec.columns), (self.rows, spec.rows),
                          (self.square, spec.square_mm), (self.marker, spec.marker_mm),
                          (self.gap, spec.gap_mm), (self.first_id, spec.first_id))
                if any(not widget.minimum() <= value <= widget.maximum() for widget, value in fields):
                    raise ValueError('文件中的标定板参数超出当前界面的可用范围。')
                if self.dictionary.findText(spec.dictionary_name) < 0:
                    raise ValueError('文件中的标签字典不受当前界面支持。')
                metadata, samples = batch['metadata'], batch['samples']
                width, height = metadata['image_size']
                guide = Guide(self.min_samples.value(), max(self.max_samples.value(), len(samples)),
                              self.sharpness.value(), self.stable.value(), self.rms_limit.value())
                for sample in samples:
                    guide.accept(SimpleNamespace(**sample), (height, width, 3))
                first = cv2.imdecode(np.frombuffer(samples[0]['image'], np.uint8), cv2.IMREAD_COLOR)
                if self.session['samples']:
                    answer = QMessageBox.question(self, '替换当前采样',
                        f'用导入的 {len(samples)} 张照片替换当前相机的 {len(self.session["samples"])} 张内存样本及计算结果？\n'
                        '未导出的当前结果会丢失，已保存的原始文件不会改动。',
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if answer != QMessageBox.Yes:
                        self.busy = False
                        self.update_state()
                        self.notice('已取消导入，原采样仍保留。相机已关闭。')
                        return
                actual = metadata.get('actual', {})
                mode = None
                if (actual.get('width'), actual.get('height')) == (width, height):
                    fps, fmt = actual.get('fps'), actual.get('pixel_format')
                    if isinstance(fps, (int, float)) and np.isfinite(fps) and fps > 0 and isinstance(fmt, str) and fmt:
                        mode = (width, height, float(fps), fmt)
                self.loading = True
                self.board_type.setCurrentIndex(self.board_type.findData(spec.kind))
                for widget, value in fields:
                    widget.setValue(value)
                self.dictionary.setCurrentText(spec.dictionary_name)
                preset = next((i for i, p in enumerate(PRESETS.values()) if p.to_dict() == spec.to_dict()), -1)
                self.presets.blockSignals(True)
                self.presets.setCurrentIndex(preset)
                self.presets.blockSignals(False)
                self.session.update(samples=samples, result=None, size=(width, height),
                    model=metadata['model'], device=metadata.get('device', ''),
                    camera_name=metadata.get('camera_name') or '文件中的相机', actual=actual,
                    mode=mode, guide=guide, bypass=False, saved=True, imported_from=batch['folder'])
                self.model_combo.setCurrentIndex(self.model_combo.findData(metadata['model']))
                self.bypass.setChecked(False)
                self.loading = False
                self.busy = False
                self.max_samples.setMinimum(max(20, max(len(s['samples']) for s in self.sessions)))
                self.update_board()
                self.restore_imported_source()
                self.coverage.set_values(guide.coverage)
                self.progress.setValue(min(100, round(len(samples) * 100 / self.min_samples.value())))
                self.guidance_text.setText(f'已导入 {len(samples)} 张原始采样，可直接计算')
                self.metrics_label.setText(f'离线原图 · 第 1 张 / 共 {len(samples)} 张 · {width} × {height}')
                self.set_live_frame(first, guide=dict(status='已恢复原始照片和角点记录',
                    instruction='点击下方“计算当前样本”，无需打开相机或重新采样。'))
                self.render_result()
                self.set_page(1)
                self.update_state()
                self.notice(f'已导入 {len(samples)} 张采样，镜头模型与标定板已从文件恢复。点击“计算当前样本”。')
                self.logger.info('用户导入相机 %s 采样，共 %s 张，模型 %s，原图 %s × %s',
                                 self.index + 1, len(samples), metadata['model'], width, height)
            except Exception as error:
                self.loading = False
                self.logger.exception('采样导入失败')
                failed(str(error))

        self.run_job(lambda: read_sample_folder(selected, max_bytes=available), done, failed)

    def metadata(self):
        return dict(schema_version=1, camera_slot=self.index+1, camera_name=self.session['camera_name'],
                    device=self.session['device'], image_size=self.session['size'], actual=self.session['actual'],
                    board=self.spec().to_dict(), model=self.session['model'], object_point_unit='m',
                    created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                    sample_records=[dict(index=i+1, captured_at=s['captured_at']) for i,s in enumerate(self.session['samples'])],
                    scope='单相机内参与畸变，不含相机间外参或手眼标定')

    def save_samples(self):
        selected = QFileDialog.getExistingDirectory(self, '选择采样照片保存目录', str(self.project / '结果'))
        if not selected:
            return
        folder = None
        try:
            parent = self.safe_path(selected)
            folder = parent / f'相机{self.index+1}_采样_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}'
            folder.mkdir()
            for i, sample in enumerate(self.session['samples'], 1):
                photo = folder / f'采样_{i:03d}.png'
                photo.write_bytes(sample['image'])
                metadata = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sample.items() if k != 'image'}
                record = photo.with_suffix('.json')
                record.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            manifest = folder / '采样信息.json'
            manifest.write_text(json.dumps(self.metadata(), ensure_ascii=False, indent=2), encoding='utf-8')
            self.session['saved'] = True
            self.notice(f'已保存 {len(self.session["samples"])} 张原始照片及角点记录至：{folder}')
            self.logger.info('用户保存相机 %s 采样照片，共 %s 张', self.index+1, len(self.session['samples']))
        except Exception as error:
            self.logger.exception('保存采样失败')
            self.error(f'保存未完成：{error}。已写入的文件保留在 {folder or selected}，可检查后重试。')

    def save_result(self):
        try:
            path = self.choose_file('导出标定参数', f'相机{self.index+1}_标定结果.json', 'JSON 文件 (*.json)')
            if path:
                record = self.metadata()
                record['calibration'] = self.session['result']
                self.write_atomic(path, json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
                self.notice(f'已导出标定结果：{path}')
                self.logger.info('用户导出相机 %s 标定结果', self.index+1)
        except Exception as error:
            self.logger.exception('导出结果失败')
            self.error(str(error))

    def save_csv(self):
        try:
            path = self.choose_file('导出逐张误差', f'相机{self.index+1}_重投影误差.csv', 'CSV 文件 (*.csv)')
            if path:
                text = io.StringIO(newline='')
                writer = csv.writer(text)
                writer.writerow(['采样序号', '角点数', '重投影RMS_px'])
                for i, value in enumerate(self.session['result']['per_view_rms']):
                    writer.writerow([i+1, len(self.session['samples'][i]['image_points']), f'{value:.8f}'])
                self.write_atomic(path, text.getvalue().encode('utf-8-sig'))
                self.notice(f'已导出逐张重投影误差：{path}')
                self.logger.info('用户导出重投影误差 CSV')
        except Exception as error:
            self.logger.exception('导出误差失败')
            self.error(str(error))

    def closeEvent(self, event):
        self.validation.stop()
        self.timer.stop()
        self.automatic = False
        self.active = False
        self.generation += 1
        self.controller.stop()
        self.logger.info('相机标定窗口关闭；未保存缓存释放')
        event.accept()
