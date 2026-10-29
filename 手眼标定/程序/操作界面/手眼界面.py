"""通用 eye-in-hand 工作台：TCP 设置、自动配对采集与手眼求解。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal, QTimer, QUrl, QProcess
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QComboBox, QCheckBox, QLineEdit, QPlainTextEdit, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter, QSizePolicy, QSpinBox, QProgressBar,
)

# 配色只取「色盲友好橙」一行，色值与控件样式来自界面主题.py，不在本文件另写色值。
import 界面主题 as 主题
from 标定界面 import 图标目录, label, button, card, scroll
from 核心算法.标定板 import BoardSpec, KIND_NAMES
from 手眼算法 import load_camera_record, estimate_board, parse_pose, solve_handeye, validate_transform, evaluate_samples
from 配对数据 import make_bundle, validate_bundle, read_bundle, write_json
from 采集面板 import CameraPanel
from 自动采集 import CaptureGate
from 采集引导 import assess_image, capture_guidance
from 机械臂接入 import discover_can_interfaces

方案 = 主题.取方案('色盲友好橙')


def 补充样式(方案):
    """主题未覆盖的控件（数值框、进度条、分割条、复选框等）沿用同一套方案色。"""
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
QCheckBox::indicator:unchecked {{ background: {方案['card']}; border: 1px solid {方案['描边']}; border-radius: 3px; }}
QCheckBox::indicator:checked {{ background: {方案['primary']}; border: 1px solid {方案['primary']}; border-radius: 3px; }}
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
QSplitter::handle {{ background: {方案['selected']}; width: 3px; height: 3px; }}
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
QFrame#header {{ background: {方案['card']}; border: 1px solid {方案['selected']}; border-radius: 14px; }}
QFrame#guidance {{ background: {方案['selected']}; border: 1px solid {方案['描边']}; border-radius: 10px; }}
'''


class Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    def __init__(self, function, quiet=False):
        super().__init__()
        self.function = function
        self.quiet = quiet
        self.signals = Signals()

    def run(self):
        try:
            self.signals.done.emit(self.function())
        except Exception as exc:
            if not self.quiet:
                logging.getLogger('手眼标定').exception('后台操作失败')
            self.signals.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, project, logger, log_path):
        super().__init__()
        self.project, self.logger, self.log_path = Path(project), logger, Path(log_path)
        self.camera_record = None
        self.samples = []
        self.result = self.robot = None
        self.robot_pose = None
        self.busy = self.closed = self.dirty = False
        self.robot_connecting = False
        self.can_configuring = False
        self.can_interfaces = []
        self.robot_generation = 0
        self.auto_running = self.capture_inflight = self.pose_reading = False
        self.solve_after_capture = False
        self.auto_checking = False
        self.auto_checked_count = 0
        self.auto_quality_note = ''
        self.capture_generation = 0
        self.next_capture_at = 0.0
        self.last_capture_serial = -1
        self.gate = CaptureGate()
        self.capture_defaults_path = self.project / '结果/采集默认参数.json'
        self.capture_defaults = (1.0, 20.0, 8.0)
        self.jobs = {}
        self.setWindowTitle('手眼标定 · Eye-in-hand 工作台')
        self.resize(1380, 1000)
        self.setMinimumSize(1060, 720)
        QApplication.instance().setStyle('Fusion')
        # 设在应用级别，保证提示框、对话框和无父窗口的浮层也用同一套配色。
        QApplication.instance().setStyleSheet(主题.生成样式(方案) + 补充样式(方案))
        self.build_ui()
        self.refresh_can_interfaces()
        self.update_state()
        self.robot_timer = QTimer(self)
        self.robot_timer.setInterval(80)
        self.robot_timer.timeout.connect(self.poll_robot)
        self.robot_timer.start()
        QTimer.singleShot(150, self.camera.scan)

    def build_ui(self):
        root = QWidget()
        root.setObjectName('页面')
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 14)
        outer.setSpacing(14)
        header = QFrame()
        header.setObjectName('header')
        row = QHBoxLayout(header)
        row.setContentsMargins(24, 16, 24, 16)
        titles = QVBoxLayout()
        titles.addWidget(label('手眼标定', 'title'))
        titles.addWidget(label('EYE-IN-HAND  ·  相机随末端运动，标定板固定在工作区', 'subtitle'))
        row.addLayout(titles, 1)
        row.addWidget(button('打开相机标定', self.open_camera_calibration))
        row.addWidget(button('打开日志', self.open_log))
        outer.addWidget(header)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_setup(), '01  准备参数')
        self.tabs.addTab(self.build_capture(), '02  自动采集')
        self.tabs.addTab(self.build_results(), '03  标定结果')
        outer.addWidget(self.tabs, 1)
        self.footer = label('就绪 · 未保存的数据只保留在本次运行中。', 'muted')
        outer.addWidget(self.footer)

    def build_setup(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        panel, left = card('相机内参与标定板')
        left.addWidget(label('直接读取本工作区“相机标定”导出的结果，沿用镜头模型、K / D 和真实板尺寸。', 'muted'))
        self.load_camera_button = button('读取相机标定 JSON…', self.load_camera, True)
        left.addWidget(self.load_camera_button)
        self.camera_summary = label('尚未读取参数\n\n支持普通针孔与鱼眼镜头；使用原始、未去畸变照片。', 'muted')
        left.addWidget(self.camera_summary)
        self.intrinsic_text = QPlainTextEdit()
        self.intrinsic_text.setReadOnly(True)
        self.intrinsic_text.setPlaceholderText('加载后显示相机内参 K 与畸变 D')
        self.intrinsic_text.setMinimumHeight(170)
        left.addWidget(self.intrinsic_text)
        self.identity = QCheckBox('已核对：实体相机、视频流、焦距及标定板\n与这份相机标定结果一致')
        self.identity.toggled.connect(self.update_state)
        left.addWidget(self.identity)
        self.board_warning = label('标定板固定不动；相机与所选末端坐标系保持刚性连接。', 'muted')
        left.addWidget(self.board_warning)
        left.addStretch()
        layout.addWidget(scroll(panel), 1)
        panel, right = card('参考坐标系 · 默认使用法兰')
        right.addWidget(label('手眼标定直接选“法兰”即可，无需针尖或接触固定点。机械臂位姿自动读取。', 'muted'))
        self.frame_combo = QComboBox()
        self.frame_combo.addItem('法兰', 'flange')
        self.frame_combo.addItem('夹爪中心（PiPER X 参考，可调整）', 'gripper')
        self.frame_combo.addItem('自定义 TCP', 'custom')
        right.addWidget(self.frame_combo)
        self.tool_name = QLineEdit('法兰')
        self.tool_name.setPlaceholderText('TCP 名称，例如夹爪中心、工具尖端')
        right.addWidget(self.tool_name)
        right.addWidget(label('TCP 相对法兰的固定偏移', 'section'))
        right.addWidget(label('位置沿法兰的 X / Y / Z 轴填写；角度定义 TCP 的朝向。只设置一次，采集时自动换算。', 'muted'))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.offsets = []
        for i, name in enumerate(('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw')):
            spin = QDoubleSpinBox()
            spin.setRange(-2000 if i < 3 else -180, 2000 if i < 3 else 180)
            spin.setDecimals(3)
            spin.setSuffix(' mm' if i < 3 else ' °')
            spin.setMinimumWidth(130)
            spin.valueChanged.connect(self.update_state)
            grid.addWidget(label(name), (i // 3) * 2, i % 3)
            grid.addWidget(spin, (i // 3) * 2 + 1, i % 3)
            self.offsets.append(spin)
        right.addLayout(grid)
        self.reset_tcp_button = button('恢复所选 TCP 的预设值', self.reset_tcp)
        right.addWidget(self.reset_tcp_button)
        self.tcp_note = label('', 'muted')
        right.addWidget(self.tcp_note)
        self.direction_label = label('输出：TCP → 相机 · T_cam_tcp', 'section')
        right.addWidget(self.direction_label)
        right.addWidget(label('采集开始后锁定 TCP 设置；需要更换时先保存并清空本轮样本。偏移只用于本程序计算。', 'muted'))
        right.addStretch()
        row = QHBoxLayout()
        self.import_bundle_button = button('读取已保存的采集…', self.load_samples)
        row.addWidget(self.import_bundle_button)
        row.addWidget(button('进入自动采集 →', lambda: self.tabs.setCurrentIndex(1), True))
        right.addLayout(row)
        layout.addWidget(scroll(panel), 1)
        self.frame_combo.currentIndexChanged.connect(self.tcp_changed)
        self.tool_name.textChanged.connect(self.update_state)
        self.tcp_changed()
        return page

    def build_capture(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        top = QSplitter(Qt.Horizontal)
        panel, left = card('相机 · 实时画面')
        self.camera = CameraPanel(self)
        left.addWidget(self.camera)
        top.addWidget(panel)
        panel, right = card('机械臂 · 自动读取位姿')
        right.addWidget(label('松灵 PiPER X', 'section'))
        row = QHBoxLayout()
        self.can_channel = QComboBox()
        self.can_channel.setEditable(True)
        self.can_channel.addItem('左臂 · can0', 'can0')
        self.can_channel.addItem('右臂 · can1', 'can1')
        self.can_channel.setToolTip('选择左／右臂，也可输入其他已配置的 CAN 接口名')
        row.addWidget(self.can_channel, 1)
        self.robot_connect = button('连接机械臂', self.toggle_robot)
        row.addWidget(self.robot_connect)
        right.addLayout(row)
        row = QHBoxLayout()
        self.refresh_can_button = button('刷新接口', self.refresh_can_interfaces)
        self.enable_can_button = button('启用所选 CAN', self.enable_selected_can)
        self.enable_can_button.setToolTip('将所选未启用接口设为 1 Mbps；需要系统管理员认证，仅本次接入生效。')
        row.addWidget(self.refresh_can_button)
        row.addWidget(self.enable_can_button)
        row.addStretch()
        right.addLayout(row)
        self.can_status = label('', 'muted')
        right.addWidget(self.can_status)
        self.can_channel.currentTextChanged.connect(self.update_can_status)
        self.robot_status = label('连接后自动读取关节角，通过 FK 获取法兰位姿。', 'muted')
        right.addWidget(self.robot_status)
        self.pose_title = label('当前 TCP 在基座中的位置与朝向', 'section')
        right.addWidget(self.pose_title)
        self.pose_display = label('尚未连接机械臂', 'muted')
        self.pose_display.setMinimumHeight(42)
        right.addWidget(self.pose_display)
        self.capture_parameters = QWidget()
        parameters_layout = QVBoxLayout(self.capture_parameters)
        parameters_layout.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        self.stable_seconds = QDoubleSpinBox()
        self.stable_seconds.setRange(0.6, 3.0)
        self.stable_seconds.setSingleStep(0.2)
        self.stable_seconds.setValue(1.0)
        self.stable_seconds.setSuffix(' 秒')
        self.move_threshold = QDoubleSpinBox()
        self.move_threshold.setRange(1, 200)
        self.move_threshold.setValue(20)
        self.move_threshold.setSuffix(' mm')
        self.turn_threshold = QDoubleSpinBox()
        self.turn_threshold.setRange(1, 90)
        self.turn_threshold.setValue(8)
        self.turn_threshold.setSuffix(' °')
        for i, (name, widget) in enumerate((('停稳等待', self.stable_seconds), ('换位距离', self.move_threshold), ('换姿角度', self.turn_threshold))):
            grid.addWidget(label(name), 0, i)
            grid.addWidget(widget, 1, i)
        parameters_layout.addLayout(grid)
        defaults_row = QHBoxLayout()
        self.set_capture_defaults_button = button('设为默认值', self.save_capture_defaults)
        self.set_capture_defaults_button.setToolTip('保存当前三个采集参数，重新打开程序时也使用这组值。')
        self.restore_capture_defaults_button = button('恢复默认值', self.restore_capture_defaults)
        self.restore_capture_defaults_button.setToolTip('恢复到你保存的默认值；尚未保存时使用 1 秒、20 mm、8°。')
        defaults_row.addStretch()
        defaults_row.addWidget(self.set_capture_defaults_button)
        defaults_row.addWidget(self.restore_capture_defaults_button)
        parameters_layout.addLayout(defaults_row)
        self.capture_defaults_label = label('', 'muted')
        parameters_layout.addWidget(self.capture_defaults_label)
        self.load_capture_defaults()
        right.addWidget(label('保持标定板固定，按下方提示手动调整机械臂。画面合格且停稳后自动采集。', 'muted'))
        finish_row = QHBoxLayout()
        self.auto_finish = QCheckBox('数据充分后自动标定')
        self.auto_finish.setChecked(True)
        self.target_samples = QSpinBox()
        self.target_samples.setRange(5, 80)
        self.target_samples.setValue(15)
        self.target_samples.setSuffix(' 组')
        self.target_samples.setToolTip('达到目标组数后，还须通过多轴旋转与固定板一致性检查。建议至少 15 组。')
        finish_row.addWidget(self.auto_finish, 1)
        finish_row.addWidget(label('目标样本'))
        finish_row.addWidget(self.target_samples)
        right.addLayout(finish_row)
        self.auto_button = button('开始自动采集', self.toggle_auto, True)
        right.addWidget(self.auto_button)
        self.capture_status = label('请先加载相机参数、打开相机并连接机械臂。', 'muted')
        right.addWidget(self.capture_status)
        guide = QFrame()
        guide.setObjectName('guidance')
        guide_layout = QVBoxLayout(guide)
        guide_layout.setContentsMargins(14, 12, 14, 12)
        guide_layout.setSpacing(7)
        guide_layout.addWidget(label('识别与下一步引导', 'section'))
        self.image_quality = label('等待开始 · 自动检查角点、画面边缘和识别误差。', 'muted')
        guide_layout.addWidget(self.image_quality)
        self.coverage_summary = label('已采集 0 / 15 组', 'muted')
        guide_layout.addWidget(self.coverage_summary)
        self.capture_progress = QProgressBar()
        self.capture_progress.setTextVisible(False)
        self.capture_progress.setRange(0, 15)
        self.capture_progress.setValue(0)
        guide_layout.addWidget(self.capture_progress)
        self.guidance_text = label('先让标定板完整出现在画面中央，再点击开始自动采集。')
        guide_layout.addWidget(self.guidance_text)
        self.auto_quality_status = label('自动完成：多轴姿态充分，固定板一致性 RMS ≤ 3 mm / 1°。', 'muted')
        self.auto_quality_status.setToolTip('同时要求最大偏差 ≤ 6 mm / 2°；这些是本轮数据一致性筛选门槛，不代表实际定位精度。')
        guide_layout.addWidget(self.auto_quality_status)
        right.insertWidget(right.indexOf(self.pose_display) + 1, guide)
        parameters_button = button('展开采集参数 · 停稳 / 换位 / 换姿', lambda: None)
        parameters_button.setCheckable(True)
        parameters_button.toggled.connect(self.capture_parameters.setVisible)
        parameters_button.toggled.connect(lambda expanded: parameters_button.setText(
            '收起采集参数' if expanded else '展开采集参数 · 停稳 / 换位 / 换姿'))
        right.addWidget(parameters_button)
        right.addWidget(self.capture_parameters)
        self.capture_parameters.hide()
        right.addWidget(label('画面箭头表示标定板在画面中的目标位置；程序只读机械臂，不控制运动。', 'muted'))
        self.target_samples.valueChanged.connect(self.update_guidance)
        self.auto_finish.toggled.connect(self.update_guidance)
        right.addStretch()
        top.addWidget(panel)
        top.setSizes([620, 530])
        top.setChildrenCollapsible(False)
        layout.addWidget(top, 3)
        panel, bottom = card('已采集样本')
        row = QHBoxLayout()
        self.count_label = label('已采集 0 组', 'section')
        row.addWidget(self.count_label, 1)
        self.remove_button = button('移除所选', self.remove_samples)
        self.clear_button = button('清空本轮', self.clear_all)
        self.save_samples_button = button('保存配对数据…', self.save_samples)
        self.save_images_button = button('保存采样照片…', self.save_images)
        for widget in (self.remove_button, self.clear_button, self.save_images_button, self.save_samples_button):
            row.addWidget(widget)
        bottom.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['序号', '采集时间／来源', '角点数', 'TCP 位置 X / Y / Z · mm'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(130)
        self.table.itemSelectionChanged.connect(self.update_state)
        bottom.addWidget(self.table)
        layout.addWidget(panel, 2)
        area = scroll(page)
        self.camera.attach_preview_layout(area, top)
        return area

    def build_results(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        panel, controls = card('计算手眼变换')
        row = QHBoxLayout()
        self.method = QComboBox()
        self.method.addItems(['PARK', 'TSAI', 'HORAUD', 'ANDREFF', 'DANIILIDIS'])
        row.addWidget(label('求解方法'))
        row.addWidget(self.method)
        self.solve_button = button('计算手眼变换', self.solve, True)
        row.addWidget(self.solve_button)
        row.addStretch()
        self.export_button = button('导出结果 JSON…', self.export_result)
        row.addWidget(self.export_button)
        controls.addLayout(row)
        self.result_status = label('采集充分后自动计算并显示结果，也可手动停止并计算。', 'muted')
        controls.addWidget(self.result_status)
        layout.addWidget(panel)
        panel, content = card('TCP → 相机 · 固定变换')
        self.matrix_title = label('等待求解', 'section')
        content.addWidget(self.matrix_title)
        self.matrix_text = QPlainTextEdit()
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setFont(QFont('DejaVu Sans Mono', 13))
        self.matrix_text.setStyleSheet('font-family:"DejaVu Sans Mono",monospace;font-size:16px;')
        self.matrix_text.setMinimumHeight(220)
        content.addWidget(self.matrix_text, 1)
        content.addWidget(label('p_cam = T_cam_tcp × p_tcp\n把 TCP 坐标系中的点转换到相机坐标系。矩阵平移单位：米 m。', 'muted'))
        self.copy_button = button('复制 TCP → 相机矩阵', self.copy_matrix)
        content.addWidget(self.copy_button)
        self.result_notes = label('TCP 选法兰时，输出的就是法兰 → 相机。', 'muted')
        content.addWidget(self.result_notes)
        layout.addWidget(panel, 1)
        return scroll(page)

    def run_job(self, function, done, failed=None, quiet=False):
        job = Job(function, quiet)
        key = id(job)
        self.jobs[key] = job

        def success(value):
            self.jobs.pop(key, None)
            if not self.closed:
                done(value)

        def failure(message):
            self.jobs.pop(key, None)
            if not self.closed:
                (failed or self.error)(message)

        job.signals.done.connect(success)
        job.signals.failed.connect(failure)
        QThreadPool.globalInstance().start(job)

    def notice(self, text):
        self.footer.setText(text)

    def error(self, text):
        self.logger.warning('操作提示：%s', text)
        QMessageBox.warning(self, '请检查', str(text))

    def open_log(self):
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_path))):
            self.error(f'无法打开日志，请查看：{self.log_path}')

    def open_camera_calibration(self):
        entry = self.project.parent / '相机标定/启动标定'
        ok, _ = QProcess.startDetached(str(entry), [], str(entry.parent))
        if not ok:
            self.error(f'无法启动相机标定：{entry}')

    def tcp_changed(self, *_):
        mode = self.frame_combo.currentData()
        self.tool_name.setText({'flange': '法兰', 'gripper': '夹爪中心', 'custom': '自定义 TCP'}[mode])
        if mode != 'custom':
            self.reset_tcp()
        self.update_state()

    def reset_tcp(self):
        values = [0, 0, 142.5 if self.frame_combo.currentData() == 'gripper' else 0, 0, 0, 0]
        for spin, value in zip(self.offsets, values):
            spin.setValue(value)

    def tcp_definition(self):
        if self.samples and getattr(self, 'loaded_tcp_definition', None):
            return self.loaded_tcp_definition
        name = self.tool_name.text().strip()
        if not name:
            raise ValueError('请填写 TCP 名称。')
        mode = self.frame_combo.currentData()
        pose = parse_pose(' '.join(str(s.value()) for s in self.offsets), 'rpy', 'mm', 'deg')
        return dict(mode=mode, name=name, T_flange_tcp=pose.tolist(),
                    source='PiPER X URDF 手指参考原点中点，偏移可调整' if mode == 'gripper' else '用户选择／输入')

    def tcp_transform(self):
        return np.asarray(self.tcp_definition()['T_flange_tcp'])

    def selected_channel(self):
        index = self.can_channel.currentIndex()
        if index >= 0 and self.can_channel.currentText() == self.can_channel.itemText(index):
            return self.can_channel.itemData(index)
        return self.can_channel.currentText().strip()

    def update_state(self, *_):
        if not hasattr(self, 'footer'):
            return
        operating = self.auto_running or self.capture_inflight or self.auto_checking or self.solve_after_capture or self.busy or self.robot_connecting or self.can_configuring
        locked = bool(self.samples) or operating
        mode = self.frame_combo.currentData()
        self.load_camera_button.setEnabled(not locked and not self.camera.active)
        self.identity.setEnabled(not operating)
        self.frame_combo.setEnabled(not locked)
        self.tool_name.setEnabled(not locked and mode != 'flange')
        for spin in self.offsets:
            spin.setEnabled(not locked and mode != 'flange')
        self.reset_tcp_button.setEnabled(not locked and mode != 'flange')
        self.tcp_note.setText({
            'flange': '法兰作为 TCP：位置与角度偏移均为 0。',
            'gripper': '参考值：Z = 142.5 mm，坐标轴与法兰平行。来自 PiPER X 官方模型中两指参考原点的中点；实际夹持点不同或更换手指时，请调整。',
            'custom': '填写你选定的工具点相对法兰的位置与朝向。角度采用固定轴 RPY：Rz(Yaw) × Ry(Pitch) × Rx(Roll)。',
        }[mode])
        name = self.tool_name.text().strip() or 'TCP'
        self.direction_label.setText(f'输出：{name} → 相机 · T_cam_tcp')
        self.pose_title.setText(f'当前 {name} 在基座中的位置与朝向')
        self.count_label.setText(f'已采集 {len(self.samples)} 组')
        self.auto_button.setText('停止采集并计算' if self.auto_running else '开始自动采集')
        ready = self.robot is not None and self.camera.active and self.camera_record and self.identity.isChecked() and name
        if self.capture_status.text() in ('请先加载相机参数、打开相机并连接机械臂。', '设备已就绪，可以开始自动采集。'):
            self.capture_status.setText('设备已就绪，可以开始自动采集。' if ready else '请先加载相机参数、打开相机并连接机械臂。')
        self.auto_button.setEnabled(self.auto_running or (bool(ready) and not operating))
        self.solve_button.setEnabled(len(self.samples) >= 5 and not operating)
        self.export_button.setEnabled(bool(self.result) and not operating)
        self.copy_button.setEnabled(bool(self.result) and not operating)
        self.solve_button.setText('重新计算' if self.result else '计算手眼变换')
        self.save_samples_button.setEnabled(bool(self.samples) and not operating)
        self.save_images_button.setEnabled(any('_image' in item for item in self.samples) and not operating)
        self.remove_button.setEnabled(bool(self.table.selectionModel().selectedRows()) and not operating)
        self.clear_button.setEnabled(bool(self.samples or self.result) and not operating)
        self.import_bundle_button.setEnabled(not operating and not self.camera.active)
        self.method.setEnabled(not operating)
        self.can_channel.setEnabled(self.robot is None and not self.robot_connecting and not operating)
        self.robot_connect.setEnabled(not self.robot_connecting and not self.can_configuring and not self.busy)
        self.robot_connect.setText('断开机械臂' if self.robot else '连接机械臂')
        self.refresh_can_button.setEnabled(not operating)
        interface = next((item for item in self.can_interfaces if item['channel'] == self.selected_channel()), None)
        self.enable_can_button.setEnabled(not operating and self.robot is None and interface is not None and not interface['is_up'])
        for spin in (self.stable_seconds, self.move_threshold, self.turn_threshold):
            spin.setEnabled(not operating)
        self.set_capture_defaults_button.setEnabled(not operating)
        self.restore_capture_defaults_button.setEnabled(not operating)
        self.auto_finish.setEnabled(not operating)
        self.target_samples.setEnabled(not operating)
        self.camera.controls()

    def load_capture_defaults(self):
        """只读取用户明确保存的默认值；启动不创建或重写设置。"""
        warning = ''
        if self.capture_defaults_path.exists():
            try:
                with self.capture_defaults_path.open('rb') as stream:
                    content = stream.read(8193)
                if len(content) > 8192:
                    raise ValueError('默认参数文件超过 8 KiB')
                record = json.loads(content.decode('utf-8'))
                if not isinstance(record, dict) or type(record.get('版本')) is not int or record['版本'] != 1:
                    raise ValueError('默认参数文件格式不受支持')
                values = tuple(record[key] for key in ('停稳等待_秒', '换位距离_mm', '换姿角度_度'))
                for value, spin in zip(values, (self.stable_seconds, self.move_threshold, self.turn_threshold)):
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not np.isfinite(value) or not spin.minimum() <= value <= spin.maximum()):
                        raise ValueError('默认参数超出可用范围')
                self.capture_defaults = values
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warning = ' · 读取失败，暂用初始值；详见日志'
                self.logger.warning('无法读取采集默认参数：%s', exc)
        for spin, value in zip((self.stable_seconds, self.move_threshold, self.turn_threshold), self.capture_defaults):
            spin.setValue(value)
        self.show_capture_defaults(warning)

    def show_capture_defaults(self, suffix=''):
        wait, distance, angle = self.capture_defaults
        self.capture_defaults_label.setText(f'默认值：{wait:g} 秒 · {distance:g} mm · {angle:g}°' + suffix)

    def save_capture_defaults(self):
        values = tuple(spin.value() for spin in (self.stable_seconds, self.move_threshold, self.turn_threshold))
        try:
            write_json(self.capture_defaults_path, dict(zip(
                ('版本', '停稳等待_秒', '换位距离_mm', '换姿角度_度'), (1, *values))))
        except Exception as exc:
            self.error('默认值保存失败，原默认值保持不变：' + str(exc))
            return
        self.capture_defaults = values
        self.show_capture_defaults(' · 已保存，下次打开也生效')
        self.notice('已将当前三个采集参数保存为默认值。误改后可点击“恢复默认值”。')
        self.logger.info('用户保存采集默认参数')

    def restore_capture_defaults(self):
        for spin, value in zip((self.stable_seconds, self.move_threshold, self.turn_threshold), self.capture_defaults):
            spin.setValue(value)
        self.show_capture_defaults(' · 已恢复')
        self.notice('已恢复采集默认值。')
        self.logger.info('用户恢复采集默认参数')

    def set_camera_record(self, record, title):
        normalized = load_camera_record(record)
        self.camera_record = normalized
        spec = BoardSpec(**self.camera_record['board'])
        c = self.camera_record['calibration']
        model = '鱼眼' if c['model'] == 'fisheye' else '普通针孔'
        self.camera_summary.setText(f'{title}\n相机：{self.camera_record.get("camera_name") or "未记录名称，请核对实体设备"}\n'
                                    f'{model} · {c["image_size"][0]} × {c["image_size"][1]}\n'
                                    f'{KIND_NAMES[spec.kind]} · {spec.columns} 列 × {spec.rows} 行\n'
                                    f'板面尺寸：{spec.size_mm[0]:g} × {spec.size_mm[1]:g} mm')
        self.intrinsic_text.setPlainText('K =\n' + np.array2string(np.array(c['K']), precision=6) + '\n\nD =\n' + str(c['D']))
        warnings = self.camera_record.get('warnings', [])
        self.board_warning.setText('\n'.join(warnings) if warnings else '带编码标定板可保持角点身份一致。采集期间，板必须固定不动。')
        self.identity.setChecked(False)
        self.update_state()

    def load_camera(self):
        path, _ = QFileDialog.getOpenFileName(self, '读取相机标定结果', str(self.project.parent / '相机标定/结果'), 'JSON 文件 (*.json)')
        if not path:
            return
        try:
            if Path(path).stat().st_size > 32 * 1024 * 1024:
                raise ValueError('参数文件超过 32 MiB，请选择相机标定结果 JSON。')
            self.set_camera_record(json.loads(Path(path).read_text(encoding='utf-8')), Path(path).name)
            self.camera.device_changed()
            self.logger.info('读取相机内参与标定板参数')
            self.notice('相机参数已加载，请核对实体设备与坐标约定。')
        except Exception as exc:
            self.error(str(exc))

    def sample_context(self):
        definition = self.tcp_definition()
        return ('flange' if definition['mode'] == 'flange' else 'tcp'), definition['name']

    def toggle_auto(self):
        if self.auto_running:
            self.stop_auto('已停止采集，准备计算结果…', calculate=True)
            return
        try:
            self.tcp_definition()
            if not self.camera_record or not self.identity.isChecked() or not self.camera.active or self.robot is None:
                raise ValueError('请先核对相机参数、打开相机并连接机械臂。')
            if any(not s.get('robot_channel') for s in self.samples):
                raise ValueError('旧样本没有机械臂来源记录，只能重新求解。继续采集请先保存并清空本轮。')
            if any(s.get('robot_channel') and s['robot_channel'] != self.robot.channel for s in self.samples):
                raise ValueError('本轮样本来自另一机械臂，请先保存并清空本轮。')
            if len(self.samples) >= 200:
                raise ValueError('本轮已达到 200 组，请先保存并清空本轮。')
            self.gate.clear()
            self.capture_generation += 1
            self.auto_running = True
            self.auto_checked_count = 0
            self.auto_quality_note = ''
            self.next_capture_at = time.monotonic()
            self.capture_status.setText('自动识别已开始，按引导调整机械臂，停稳后自动采集。')
            self.update_guidance()
            self.maybe_auto_solve()
            self.logger.info('用户开始自动采集')
            self.update_state()
        except Exception as exc:
            self.error(str(exc))

    def stop_auto(self, message='已停止自动采集。', calculate=False):
        was_running = self.auto_running
        self.auto_running = False
        self.solve_after_capture = calculate and was_running
        self.capture_generation += 1
        self.gate.clear()
        self.camera.set_guidance(None)
        self.capture_status.setText(message)
        if was_running:
            self.logger.info('自动采集停止：%s', message)
        self.update_state()
        self.finish_capture_result()

    def finish_capture_result(self):
        if not self.solve_after_capture or self.capture_inflight or self.auto_checking:
            return
        self.solve_after_capture = False
        self.tabs.setCurrentIndex(2)
        if len(self.samples) < 5:
            self.result = None
            self.render_result()
            message = f'尚不能计算：当前 {len(self.samples)} 组，至少还需 {5 - len(self.samples)} 组。请继续采集不同轴旋转的姿态。'
            self.result_status.setText(message)
            self.capture_status.setText(message)
            self.notice(message)
            return
        self.solve()

    def poll_robot(self):
        if self.robot is None or self.pose_reading or self.closed:
            return
        reader = self.robot
        generation = self.robot_generation
        self.pose_reading = True

        def done(feedback):
            self.pose_reading = False
            if generation != self.robot_generation or reader is not self.robot:
                return
            try:
                fresh = self.gate.push(feedback, time.monotonic())
                self.robot_pose = feedback
                transform = np.asarray(feedback['T_base_end']) @ self.tcp_transform()
                xyz = transform[:3, 3] * 1000
                rotation = transform[:3, :3]
                pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
                roll = np.arctan2(rotation[2, 1], rotation[2, 2])
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                self.pose_display.setText('X / Y / Z：' + ' / '.join(f'{v:.2f}' for v in xyz) + ' mm\n'
                    + 'Roll / Pitch / Yaw：' + ' / '.join(f'{v:.2f}' for v in np.degrees([roll, pitch, yaw])) + ' °')
                source = feedback.get('pose_source_label', feedback.get('pose_source', '法兰反馈'))
                self.robot_status.setText(f'已连接 · {reader.channel} · {source}')
                if fresh:
                    self.try_auto_capture()
            except Exception as exc:
                failed(str(exc))

        def failed(message):
            self.pose_reading = False
            if generation != self.robot_generation or reader is not self.robot:
                return
            self.robot_pose = None
            self.gate.clear()
            self.pose_display.setText('暂无有效位姿')
            self.robot_status.setText('位姿读取失败：' + message)
            if self.auto_running:
                self.stop_auto('位姿读取中断，自动采集已停止。恢复连接后重新开始。')

        self.run_job(reader.read_pose, done, failed, quiet=True)

    def try_auto_capture(self):
        now = time.monotonic()
        if not self.auto_running or self.capture_inflight or self.auto_checking or now < self.next_capture_at:
            return
        if not self.camera.active or self.camera.frame is None:
            return
        if self.camera.frame_serial == self.last_capture_serial:
            return
        accepted = [s.get('T_base_flange', s['T_base_end']) for s in self.samples]
        feedback, reason = self.gate.ready(now, self.camera.last_frame_at, self.stable_seconds.value(),
                                          accepted, self.move_threshold.value(), self.turn_threshold.value())
        if not 0 <= now - self.camera.last_frame_at <= 0.2:
            self.capture_status.setText('等待新的相机画面…')
            return
        frame = self.camera.frame.copy()
        frame_time = self.camera.last_frame_at
        self.last_capture_serial = self.camera.frame_serial
        record = self.camera_record
        token = self.capture_generation
        reader = self.robot
        definition = self.tcp_definition()
        flange = validate_transform(feedback['T_base_end']) if feedback is not None else None
        end = flange @ np.asarray(definition['T_flange_tcp']) if flange is not None else None
        captured_at = self.camera.frame_captured_at or datetime.now().astimezone().isoformat(timespec='milliseconds')
        self.capture_inflight = True
        self.capture_status.setText(reason or '已停稳，正在识别标定板并配对…')
        self.update_state()

        def work():
            measurement = estimate_board(frame, record)
            quality = assess_image(measurement, record['board'])
            if not quality['accepted'] or feedback is None:
                return measurement, quality, None
            ok, encoded = cv2.imencode('.png', frame)
            if not ok:
                raise ValueError('无法暂存采样图像。')
            digest = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
            candidate = dict(measurement, _image=encoded.tobytes(), image_sha256=digest,
                        T_base_flange=flange.tolist(), T_base_end=end.tolist(), role='calibration',
                        id=uuid.uuid4().hex, source='实时自动采集 · ' + reader.channel,
                        robot_channel=reader.channel, pose_source=feedback.get('pose_source', 'flange_feedback'),
                        pair_delta_ms=abs(feedback['monotonic'] - frame_time) * 1000, captured_at=captured_at)
            return measurement, quality, candidate

        def done(payload):
            self.capture_inflight = False
            if token != self.capture_generation or not self.auto_running or reader is not self.robot:
                self.update_state()
                self.finish_capture_result()
                return
            try:
                measurement, quality, candidate = payload
                self.next_capture_at = time.monotonic() + 0.45
                self.image_quality.setText(quality['message'])
                self.update_guidance(None if quality['accepted'] else quality['message'])
                self.camera.set_detection(quality['center'], frame_time)
                if candidate is None:
                    self.capture_status.setText(reason if quality['accepted'] else '本次未采集：' + quality['message'])
                    self.update_state()
                    return
                if sum(len(s.get('_image', b'')) for s in self.samples) + len(candidate['_image']) > 256 * 1024 * 1024:
                    self.stop_auto('采样图像内存达到 256 MiB，请保存后再清空本轮。')
                    return
                frame_kind, tool = self.sample_context()
                make_bundle(record, frame_kind, tool, self.samples + [candidate], tcp_definition=definition)
                self.samples.append(candidate)
                self.result = None
                self.dirty = True
                self.next_capture_at = time.monotonic() + 0.5
                self.refresh_table()
                self.render_result()
                self.capture_status.setText(f'已自动采集第 {len(self.samples)} 组 · {candidate["point_count"]} 个角点。请换一个姿态。')
                self.notice('采样数据暂存在内存，点击保存才写入磁盘。')
                self.logger.info('自动加入手眼样本，当前共%d组', len(self.samples))
                if len(self.samples) >= 200:
                    self.stop_auto('已采集 200 组，正在计算结果…', calculate=True)
                else:
                    self.maybe_auto_solve()
            except Exception as exc:
                failed(str(exc))

        def failed(message):
            self.capture_inflight = False
            if token == self.capture_generation and self.auto_running:
                self.next_capture_at = time.monotonic() + 1.0
                self.capture_status.setText('本次未采集：' + message)
                self.image_quality.setText('尚未得到合格图像：' + message)
                self.update_guidance('让标定板完整入镜，减少遮挡和反光，调整相机距离后停稳。')
                self.camera.set_detection(None, 0)
            self.update_state()
            self.finish_capture_result()

        self.run_job(work, done, failed, quiet=True)

    def update_guidance(self, issue=None):
        if not hasattr(self, 'guidance_text'):
            return
        state = capture_guidance(self.samples, self.target_samples.value())
        self.coverage_summary.setText(state['summary'])
        self.capture_progress.setMaximum(self.target_samples.value())
        self.capture_progress.setValue(min(len(self.samples), self.target_samples.value()))
        instruction = state['instruction']
        if self.result is not None and not self.auto_running:
            instruction = '本轮结果已生成，可在“标定结果”页查看、复制或导出。'
        elif state['ready'] and not self.auto_checking:
            if self.auto_quality_note:
                instruction = '补采不同方向的倾斜与距离，停稳后自动重试；若板移动过，请停止并清空本轮重新采集。'
            elif not self.auto_finish.isChecked():
                instruction = '数量与多轴姿态已满足求解条件，可以继续采集，或点击“停止采集并计算”。'
        self.guidance_text.setText(issue if isinstance(issue, str) else instruction)
        self.auto_quality_status.setText(self.auto_quality_note or
            '自动完成：多轴姿态充分，固定板一致性 RMS ≤ 3 mm / 1°。')
        self.camera.set_guidance(state['target'] if self.auto_running and not isinstance(issue, str) else None)

    @staticmethod
    def calculate_result(samples, method):
        result = solve_handeye(samples, method)
        camera_to_tcp = validate_transform(result.pop('T_end_cam'))
        result['consistency'] = evaluate_samples(samples, camera_to_tcp)
        result['T_cam_tcp'] = np.linalg.inv(camera_to_tcp).tolist()
        result['transform_direction'] = 'tcp_to_camera'
        return result

    def maybe_auto_solve(self):
        if (not self.auto_running or not self.auto_finish.isChecked() or self.auto_checking
                or len(self.samples) <= self.auto_checked_count):
            return
        state = capture_guidance(self.samples, self.target_samples.value())
        if not state['ready']:
            return
        self.auto_checked_count = len(self.samples)
        self.auto_checking = True
        token = self.capture_generation
        samples, method = list(self.samples), self.method.currentText()
        self.auto_quality_note = '样本与姿态已满足要求，正在自动求解并检查固定板一致性…'
        self.update_guidance()
        self.update_state()

        def done(result):
            self.auto_checking = False
            if token != self.capture_generation or not self.auto_running:
                self.update_state()
                self.finish_capture_result()
                return
            quality = result['consistency']
            passed = (quality['translation_rms_mm'] <= 3 and quality['rotation_rms_deg'] <= 1
                      and quality['translation_max_mm'] <= 6 and quality['rotation_max_deg'] <= 2)
            if not passed:
                self.auto_quality_note = (f"尚未自动完成：固定板一致性 RMS {quality['translation_rms_mm']:.2f} mm / "
                    f"{quality['rotation_rms_deg']:.2f}°，最大偏差 {quality['translation_max_mm']:.2f} mm / "
                    f"{quality['rotation_max_deg']:.2f}°。请检查板是否移动、图像反光及位姿配对，补采后重试。")
                self.update_guidance()
                self.update_state()
                return
            result['automatic_completion'] = True
            self.result = result
            self.auto_quality_note = '固定板一致性检查已通过，本轮手眼标定已自动完成。'
            self.stop_auto('数据充分且一致性检查通过，已自动完成手眼标定。')
            self.update_guidance()
            self.tabs.setCurrentIndex(2)
            self.render_result()
            self.notice('已自动完成手眼标定。结果与采样只在内存中，点击保存或导出才写入磁盘。')
            self.logger.info('自动完成手眼标定：%s，%d组，固定板一致性 %.3f mm / %.3f°',
                method, len(samples), quality['translation_rms_mm'], quality['rotation_rms_deg'])

        def failed(message):
            self.auto_checking = False
            if token == self.capture_generation and self.auto_running:
                self.auto_quality_note = '本次自动求解未通过：' + message
                self.update_guidance()
            self.update_state()
            self.finish_capture_result()

        self.run_job(lambda: self.calculate_result(samples, method), done, failed, quiet=True)

    def refresh_table(self):
        self.table.setRowCount(len(self.samples))
        for i, sample in enumerate(self.samples):
            xyz = np.asarray(sample['T_base_end'])[:3, 3] * 1000
            values = [str(i + 1), sample.get('captured_at', sample.get('source', '已保存数据')),
                      str(sample.get('point_count', '—')), ' / '.join(f'{v:.2f}' for v in xyz)]
            for j, value in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(value))
        self.update_guidance()
        self.update_state()

    def remove_samples(self):
        rows = sorted({x.row() for x in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.samples.pop(row)
        if not self.samples:
            self.loaded_tcp_definition = None
        if rows:
            self.result = None
            self.auto_quality_note = ''
            self.dirty = True
            self.refresh_table()
            self.render_result()
            self.notice('已移除所选样本，需重新计算结果。')

    def clear_all(self):
        if QMessageBox.question(self, '清空本轮数据', '清空内存中的配对样本和计算结果？\n尚未保存的数据将丢失，磁盘文件保持不变。',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.samples = []
        self.loaded_tcp_definition = None
        self.result = None
        self.dirty = False
        self.auto_quality_note = ''
        self.auto_checked_count = 0
        self.image_quality.setText('等待开始 · 自动检查角点、画面边缘和识别误差。')
        self.gate.clear()
        self.refresh_table()
        self.render_result()
        self.logger.info('用户清空本轮手眼数据')

    def load_samples(self):
        path, _ = QFileDialog.getOpenFileName(self, '读取已保存的采集', str(self.project / '结果'), 'JSON 文件 (*.json)')
        if not path:
            return
        try:
            record = read_bundle(path)
            definition = record.get('tcp_definition')
            if definition is None:
                if record['reference_frame'] != 'flange':
                    raise ValueError('旧 TCP 数据未记录相对法兰的偏移，无法恢复 TCP 设置，请使用包含 TCP 定义的采集文件。')
                definition = dict(mode='flange', name='法兰', T_flange_tcp=np.eye(4).tolist())
            if self.samples and QMessageBox.question(self, '替换当前样本', '读取将替换当前内存样本。是否继续？',
                                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            self.disconnect_robot()
            self.samples = []
            self.set_camera_record(record['camera'], Path(path).name)
            self.frame_combo.setCurrentIndex(self.frame_combo.findData(definition['mode']))
            self.tool_name.setText(definition['name'])
            transform = np.asarray(definition['T_flange_tcp'])
            r = transform[:3, :3]
            pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
            if abs(np.cos(pitch)) < 1e-8:
                roll, yaw = 0., np.arctan2(-r[0, 1], r[1, 1])
            else:
                roll, yaw = np.arctan2(r[2, 1], r[2, 2]), np.arctan2(r[1, 0], r[0, 0])
            for spin, value in zip(self.offsets, list(transform[:3, 3] * 1000) + list(np.degrees([roll, pitch, yaw]))):
                spin.setValue(value)
            # 回读显示精度不能改变已保存的偏移；矩阵精确值单独保留至清空。
            self.loaded_tcp_definition = definition
            self.samples = [item for item in record['samples'] if item['role'] == 'calibration']
            if record['reference_frame'] == 'flange':
                for item in self.samples:
                    item.setdefault('T_base_flange', item['T_base_end'])
            self.result = None
            self.dirty = False
            self.auto_quality_note = ''
            self.auto_checked_count = 0
            self.refresh_table()
            self.render_result()
            self.notice('已读取用于求解的样本。继续采集时须保持原相机、机械臂及标定板安装位置。')
            self.logger.info('读取已保存的采集，共%d组', len(self.samples))
        except Exception as exc:
            self.error(str(exc))


    def choose_file(self, title, name, filters):
        path, _ = QFileDialog.getSaveFileName(self, title, str(self.project / '结果' / name), filters)
        return Path(path) if path else None

    def save_samples(self):
        try:
            path = self.choose_file('保存配对数据', '手眼配对数据.json', 'JSON 文件 (*.json)')
            if path:
                frame, tool = self.sample_context()
                write_json(path, make_bundle(self.camera_record, frame, tool, self.samples, tcp_definition=self.tcp_definition()))
                self.dirty = False
                self.notice(f'已保存配对数据：{path}。原始照片需单独点击“保存采样照片”。')
                self.logger.info('用户保存配对数据')
        except Exception as exc:
            self.error(str(exc))

    def save_images(self):
        selected = QFileDialog.getExistingDirectory(self, '选择采样照片保存位置', str(self.project / '结果'))
        if not selected:
            return
        folder = None
        try:
            parent = Path(selected).resolve()
            protected = (self.project.parent / '鱼眼相机安装调试').resolve()
            if parent == protected or protected in parent.parents:
                raise ValueError('此项目不能写入受保护的鱼眼相机目录，请选择手眼标定结果目录。')
            folder = parent / ('手眼采样_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:4])
            folder.mkdir()
            count = 0
            for i, sample in enumerate(self.samples, 1):
                if '_image' not in sample:
                    continue
                photo = folder / f'样本_{i:03d}.png'
                photo.write_bytes(sample['_image'])
                write_json(photo.with_suffix('.json'), {k: v for k, v in sample.items() if not k.startswith('_')})
                count += 1
            frame, tool = self.sample_context()
            write_json(folder / '配对数据.json', make_bundle(self.camera_record, frame, tool, self.samples, tcp_definition=self.tcp_definition()))
            self.notice(f'已保存 {count} 张采样照片及配对数据至：{folder}')
            self.logger.info('用户保存%d张手眼采样照片', count)
        except Exception as exc:
            self.error(f'保存未完成：{exc}' + (f'\n已写入文件保留在：{folder}' if folder else ''))

    def solve(self):
        if self.auto_running or self.capture_inflight or self.auto_checking or self.busy:
            return
        self.busy = True
        self.result = None
        self.render_result()
        self.tabs.setCurrentIndex(2)
        self.matrix_title.setText('正在计算…')
        self.result_status.setText('正在计算 TCP → 相机的固定变换…')
        samples, method = list(self.samples), self.method.currentText()

        def work():
            return self.calculate_result(samples, method)

        def done(result):
            self.busy = False
            self.result = result
            self.auto_quality_note = ''
            self.update_guidance()
            self.render_result()
            self.update_state()
            self.capture_status.setText('标定结果已生成，可在“标定结果”页复制或导出。')
            self.notice('标定结果已生成 · TCP → 相机。点击导出才保存到磁盘。')
            self.logger.info('完成手眼求解：%s，样本%d组', method, result['calibration_count'])

        def failed(message):
            self.busy = False
            self.result = None
            self.render_result()
            self.matrix_title.setText('暂未得到标定结果')
            self.result_status.setText('无法求解：' + message)
            self.capture_status.setText('无法求解，请在结果页查看原因并补充采集。')
            self.update_state()
            self.notice('本轮未生成结果，请按提示补充样本后继续采集。')

        self.run_job(work, done, failed)




    def render_result(self, *_):
        if not hasattr(self, 'result_notes'):
            return
        if not self.result:
            self.matrix_title.setText('等待求解')
            self.matrix_text.clear()
            self.result_status.setText('采集充分后自动计算并显示结果，也可手动停止并计算。')
            self.result_notes.setText('TCP 选法兰时，输出的就是法兰 → 相机。')
        else:
            result = self.result
            self.matrix_title.setText(f'{self.tool_name.text()} → 相机 · T_cam_tcp')
            self.matrix_text.setPlainText(np.array2string(np.array(result['T_cam_tcp']), precision=8, suppress_small=True))
            prefix = '已自动完成手眼标定' if result.get('automatic_completion') else '标定结果已生成'
            self.result_status.setText(f'{prefix} · {result["method"]} · 已使用 {result["calibration_count"]} 组样本 · 平移单位 m')
            notes = list(result.get('warnings', []))
            quality = result.get('consistency')
            if quality:
                notes.append(f"固定板一致性 RMS：{quality['translation_rms_mm']:.3f} mm / {quality['rotation_rms_deg']:.3f}°；"
                    f"最大偏差：{quality['translation_max_mm']:.3f} mm / {quality['rotation_max_deg']:.3f}°。")
                notes.append('一致性使用本轮采集数据评估，不代表独立验证的实际定位精度。')
            self.result_notes.setText('\n'.join(notes))
        self.update_state()

    def copy_matrix(self):
        if self.result:
            QApplication.clipboard().setText(json.dumps(self.result['T_cam_tcp'], ensure_ascii=False, indent=2))
            self.notice('已复制 TCP → 相机矩阵，平移单位为米。')

    def result_record(self):
        frame, tool = self.sample_context()
        return dict(schema='handeye_result_v2', configuration='eye_in_hand', reference_frame=frame,
                    tool_name=tool, tcp_definition=self.tcp_definition(), translation_unit='m',
                    transform_direction='tcp_to_camera', convention='p_camera = T_cam_tcp @ p_tcp',
                    camera=self.camera_record, result=self.result,
                    exported_at=datetime.now().astimezone().isoformat(timespec='seconds'))

    def export_result(self):
        try:
            path = self.choose_file('导出手眼标定结果', '手眼标定结果.json', 'JSON 文件 (*.json)')
            if path:
                write_json(path, self.result_record())
                self.notice(f'已导出结果：{path}')
                self.logger.info('用户导出手眼结果')
        except Exception as exc:
            self.error(str(exc))



    def refresh_can_interfaces(self):
        try:
            self.can_interfaces = discover_can_interfaces()
        except OSError as exc:
            self.can_interfaces = []
            self.logger.warning('读取 CAN 接口失败：%s', exc)
        selected = self.selected_channel()
        self.can_channel.blockSignals(True)
        self.can_channel.clear()
        channels = ['can0', 'can1'] + [item['channel'] for item in self.can_interfaces if item['channel'] not in ('can0', 'can1')]
        if selected and selected not in channels:
            channels.append(selected)
        for channel in channels:
            title = {'can0': '左臂 · can0', 'can1': '右臂 · can1'}.get(channel, channel)
            self.can_channel.addItem(title, channel)
            interface = next((item for item in self.can_interfaces if item['channel'] == channel), None)
            self.can_channel.setItemData(self.can_channel.count() - 1,
                interface['label'] if interface else f'{channel} · 未检测到 USB-CAN', Qt.ToolTipRole)
        self.can_channel.setCurrentIndex(max(0, self.can_channel.findData(selected)))
        self.can_channel.blockSignals(False)
        self.update_can_status()

    def update_can_status(self, *_):
        channel = self.selected_channel()
        interface = next((item for item in self.can_interfaces if item['channel'] == channel), None)
        if self.can_configuring:
            message = f'正在启用 {channel} · 请在系统认证窗口完成管理员认证。'
        elif interface is None:
            message = f'{channel or "所选接口"} · 未检测到。请插好对应机械臂的 USB-CAN，再刷新接口。'
        elif not interface['is_up']:
            message = interface['label'] + '。请点击“启用所选 CAN”，完成系统认证后连接。'
        else:
            message = interface['label'] + ' · 可以连接机械臂读取位姿。'
        self.can_status.setText(message)
        self.update_state()

    def enable_selected_can(self):
        if self.can_configuring or self.robot_connecting or self.robot is not None or self.auto_running:
            return
        channel = self.selected_channel()
        if not shutil.which('pkexec'):
            self.error('系统未安装 pkexec，无法打开管理员认证。请由管理员启用所选 CAN 接口（1 Mbps）。')
            return
        self.can_configuring = True
        self.update_can_status()
        self.logger.info('用户请求启用 CAN 接口：%s（1 Mbps）', channel)

        def work():
            try:
                result = subprocess.run(['pkexec', '/bin/bash',
                    str(self.project / '程序/运行支持/启用CAN接口.sh'), channel],
                    capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError('系统认证超时，CAN 启用未完成；可再次点击“启用所选 CAN”。') from exc
            if result.returncode in (126, 127):
                raise RuntimeError('未完成系统管理员认证，CAN 接口未启用。请再次点击并在系统认证窗口完成认证。')
            if result.returncode:
                raise RuntimeError('CAN 启用失败：' + (result.stderr.strip() or result.stdout.strip() or '系统未返回原因'))
            return result.stdout.strip()

        def done(message):
            self.can_configuring = False
            self.refresh_can_interfaces()
            self.notice(message + ' · 现在可以点击“连接机械臂”。')
            self.logger.info('CAN 接口启用完成：%s', channel)

        def failed(message):
            self.can_configuring = False
            self.refresh_can_interfaces()
            self.error(message)

        self.run_job(work, done, failed)

    def toggle_robot(self):
        if self.robot_connecting or self.can_configuring:
            return
        if self.robot:
            self.disconnect_robot()
            return
        channel = self.selected_channel()
        if self.samples and any(item.get('robot_channel') and item['robot_channel'] != channel for item in self.samples):
            self.error('本轮样本来自另一机械臂，请先保存并清空本轮。')
            return
        self.robot_connecting = True
        self.robot_generation += 1
        generation = self.robot_generation
        self.robot_status.setText('正在连接机械臂，等待完整且新鲜的反馈…')
        self.update_state()

        def work():
            from 机械臂接入 import PiperReader
            reader = PiperReader(channel=channel)
            reader.connect()
            return reader

        def done(reader):
            if generation != self.robot_generation:
                reader.disconnect()
                return
            self.robot_connecting = False
            self.robot = reader
            self.gate.clear()
            self.robot_status.setText(f'已连接 PiPER X · {channel}')
            self.logger.info('连接PiPER X只读位姿接口：%s', channel)
            self.update_state()

        def failed(message):
            if generation != self.robot_generation:
                return
            self.robot_connecting = False
            self.refresh_can_interfaces()
            self.robot_status.setText('连接未完成：' + message)
            self.update_state()
            self.error(message)

        self.run_job(work, done, failed)

    def disconnect_robot(self):
        if self.auto_running:
            self.stop_auto('机械臂已断开，自动采集停止。')
        self.robot_generation += 1
        self.robot_connecting = False
        reader, self.robot = self.robot, None
        self.robot_pose = None
        self.gate.clear()
        if reader:
            try:
                reader.disconnect()
            except Exception:
                self.logger.exception('断开机械臂时异常')
            self.logger.info('断开机械臂只读连接')
        self.robot_status.setText('机械臂已断开。')
        self.pose_display.setText('尚未连接机械臂')
        self.update_state()

    def closeEvent(self, event):
        if self.auto_running:
            self.stop_auto('自动采集已停止。')
        if self.busy or self.capture_inflight or self.auto_checking or self.robot_connecting or self.can_configuring:
            QMessageBox.information(self, '操作进行中', '请等待当前采集或求解完成后关闭窗口。')
            event.ignore()
            return
        if self.dirty and QMessageBox.question(self, '尚有未保存的样本', '关闭将释放尚未保存的配对样本和照片。是否关闭？',
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            event.ignore()
            return
        self.closed = True
        self.robot_timer.stop()
        self.camera.stop()
        self.disconnect_robot()
        event.accept()
