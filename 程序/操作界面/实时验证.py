"""固定已有内参的实时标定板核验；画面与参考值只保存在本次运行内存。"""
from __future__ import annotations

from datetime import datetime
import json
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QCheckBox, QDoubleSpinBox, QFileDialog

from 核心算法.相机接入 import CaptureController, discover_devices, probe_device
from 核心算法.标定板 import BoardSpec, KIND_NAMES
from 核心算法.标定算法 import normalize_calibration, adapt_calibration_resolution, detect_board, validate_frame
from 预览组件 import Preview


class ValidationPage(QWidget):
    def __init__(self, host, label, button, card, scroll):
        super().__init__(host)
        self.host = host
        self.controller = CaptureController()
        self.source = None
        self.spec = None
        self.active = False
        self.hardware_busy = False
        self.generation = self.discovery_generation = 0
        self.analysis_token = None
        self.last_analysis = self.last_frame_at = self.measured_at = 0.0
        self.current = self.reference = self.effective = None
        self.actual = {}
        self.frame_size = None
        self.display_frame = self.display_detection = None
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)
        panel, left = card('验证参数与相机')
        self.load_button = button('读取标定结果 JSON…', self.load_file)
        left.addWidget(self.load_button)
        self.source_label = label('尚未选择参数。可读取已导出的标定 JSON，或在标定结果页使用当前结果。', 'muted')
        left.addWidget(self.source_label)
        left.addWidget(label('当前验证相机'))
        self.devices = QComboBox()
        left.addWidget(self.devices)
        self.refresh_button = button('刷新设备', self.scan)
        left.addWidget(self.refresh_button)
        left.addWidget(label('验证分辨率与帧率'))
        self.modes = QComboBox()
        left.addWidget(self.modes)
        self.identity = QCheckBox('已核对：当前物理相机及视频流\n与这份标定参数一致')
        left.addWidget(self.identity)
        left.addWidget(label('设备名称、video 编号不能证明是同一只相机；请通过实体编号和画面确认。', 'muted'))
        self.scaling = QCheckBox('确认视场、裁剪不变，\n分辨率仅作等比例缩放')
        left.addWidget(self.scaling)
        left.addWidget(label('改变焦距不能靠分辨率换算修正。本页始终固定已有 K、D，观察调焦后的误差。', 'muted'))
        self.start_button = button('开始实时验证', self.toggle, True)
        threshold = QHBoxLayout()
        threshold.addWidget(label('RMS 参考线'))
        self.limit = QDoubleSpinBox()
        self.limit.setRange(0.01, 100.0)
        self.limit.setDecimals(2)
        self.limit.setSingleStep(0.1)
        self.limit.setValue(1.0)
        self.limit.setSuffix(' px')
        threshold.addWidget(self.limit)
        left.addLayout(threshold)
        self.effective_label = label('有效参数：等待选择标定结果。', 'muted')
        left.addWidget(self.effective_label)
        left.addStretch()
        area = scroll(panel)
        area.setMinimumWidth(335)
        area.setMaximumWidth(390)
        outer.addWidget(area)
        workspace, right = card('')
        right.setSpacing(8)
        heading = QHBoxLayout()
        heading.addWidget(label('固定 K、D · 标定板实时验证', 'section'), 1)
        heading.addWidget(self.start_button)
        right.addLayout(heading)
        self.mean_label = label('平均像素误差  — px', 'metric')
        right.addWidget(self.mean_label)
        self.details = label('RMS — px    最大误差 — px    角点 —', 'section')
        right.addWidget(self.details)
        self.state_label = label('读取参数并核对设备后开始验证。', 'muted')
        right.addWidget(self.state_label)
        self.preview = Preview()
        self.preview.setMinimumSize(420, 220)
        self.preview.empty_text = '固定已有 K、D，实时观察标定板误差\n本页不会自动采样、重新标定或保存照片'
        right.addWidget(self.preview, 1)
        self.actual_label = label('实际画面：尚未打开相机', 'muted')
        left.insertWidget(left.count() - 1, self.actual_label)
        self.reference_label = label('参考尚未记录。调焦前保持标定板稳定，点击下方按钮记录当前帧。', 'muted')
        right.addWidget(self.reference_label)
        actions = QHBoxLayout()
        self.reference_button = button('记录当前帧为参考', self.record_reference)
        self.export_button = button('导出当前验证 JSON', self.export_current)
        actions.addWidget(self.reference_button)
        actions.addWidget(self.export_button)
        right.addLayout(actions)
        left.insertWidget(left.count() - 1, label('均值＝本帧每个角点二维距离的算术平均；RMS 单独计算。仅估计板位姿，不更新 K、D。\n调焦前后请保持相近位置、倾角和相同分辨率；正面姿态可能掩盖焦距变化，还需检查边缘及倾斜视角。参考线只作提示，不代表完成精度验收。', 'muted'))
        outer.addWidget(scroll(workspace), 1)
        self.devices.currentIndexChanged.connect(self.device_changed)
        self.modes.currentIndexChanged.connect(self.settings_changed)
        self.identity.toggled.connect(self.update_controls)
        self.scaling.toggled.connect(self.settings_changed)
        self.limit.valueChanged.connect(self.render_measurement)
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.poll)
        self.update_controls()

    def enter(self):
        if not self.active:
            self.scan()

    def stop(self):
        was_active = self.active
        self.active = False
        self.timer.stop()
        self.generation += 1
        self.discovery_generation += 1
        self.hardware_busy = False
        self.analysis_token = None
        self.controller.stop()
        self.current = self.reference = None
        self.frame_size = None
        self.display_frame = self.display_detection = None
        self.preview.set_frame(None)
        self.actual_label.setText('实际画面：相机已关闭')
        self.clear_measurement('验证已停止；当前误差和调焦参考已清空。')
        self.update_reference()
        self.update_controls()
        if was_active:
            self.host.logger.info('停止固定内参实时验证')

    def set_source(self, record, title):
        parameters = normalize_calibration(record)
        if not isinstance(record.get('board'), dict):
            raise ValueError('需要含标定板配置的本程序导出文件。')
        try:
            spec = BoardSpec(**record['board'])
            spec.validate()
        except (ValueError, TypeError) as exc:
            raise ValueError('文件中的标定板配置无效：' + str(exc)) from exc
        # 全部校验成功后才替换，错误文件不影响已有来源，也不碰主标定页。
        self.stop()
        self.spec = spec
        self.source = dict(title=str(title), calibration=parameters, board=spec.to_dict(),
                           camera_name=record.get('camera_name', ''), device=record.get('device', ''))
        self.identity.setChecked(False)
        self.scaling.setChecked(False)
        width, height = parameters['image_size']
        model = '鱼眼' if parameters['model'] == 'fisheye' else '普通针孔'
        geometry = (f'方格 {spec.square_mm:g} mm' if spec.kind in ('charuco', 'chessboard')
                    else f'标签 {spec.marker_mm:g} mm / 间距 {spec.gap_mm:g} mm')
        self.source_label.setText(f'来源：{title}\n相机：{self.source["camera_name"] or "未记录"}\n'
                                  f'模型：{model} · 原分辨率 {width} × {height}\n'
                                  f'标定板：{KIND_NAMES[spec.kind]}\n{spec.columns} 列 × {spec.rows} 行 · {geometry}')
        self.state_label.setText('参数已加载。请核对当前物理相机、视频流和纸质标定板。')
        self.show_parameters(parameters)
        self.update_controls()

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, '读取含标定板配置的标定结果',
                                                 str(self.host.project / '结果'), 'JSON 文件 (*.json)')
        if not filename:
            return
        try:
            with open(filename, encoding='utf-8') as stream:
                record = json.load(stream)
            from pathlib import Path
            self.set_source(record, Path(filename).name)
            self.host.logger.info('用户读取实时验证标定参数')
        except Exception as exc:
            self.host.error(str(exc))

    def scan(self):
        if self.active or self.hardware_busy:
            return
        self.hardware_busy = True
        self.discovery_generation += 1
        token = self.discovery_generation
        self.identity.setChecked(False)
        self.update_controls()
        self.state_label.setText('正在探测视频采集设备…')
        previous = self.devices.currentData()
        def done(devices):
            if token != self.discovery_generation:
                return
            self.hardware_busy = False
            self.devices.blockSignals(True)
            self.devices.clear()
            for device in devices:
                self.devices.addItem(self.host.camera_display_name(device), device.path)
                self.devices.setItemData(self.devices.count() - 1, device.path, Qt.ToolTipRole)
            found = self.devices.findData(previous)
            self.devices.setCurrentIndex(found if found >= 0 else (0 if devices else -1))
            self.devices.blockSignals(False)
            self.device_changed()
        self.host.run_job(discover_devices, done, lambda message: self.device_failed(message, token))

    def device_failed(self, message, token):
        if token != self.discovery_generation:
            return
        self.hardware_busy = False
        self.state_label.setText('设备读取失败：' + message)
        self.update_controls()

    def device_changed(self, *_):
        self.settings_changed()
        self.identity.setChecked(False)
        self.modes.blockSignals(True)
        self.modes.clear()
        self.modes.blockSignals(False)
        path = self.devices.currentData()
        self.discovery_generation += 1
        token = self.discovery_generation
        self.hardware_busy = bool(path)
        self.update_controls()
        if not path:
            self.state_label.setText('没有发现视频设备，请连接相机并刷新。')
            return
        def done(payload):
            if token != self.discovery_generation:
                return
            self.hardware_busy = False
            source_size = self.source['calibration']['image_size'] if self.source else (1280, 720)
            modes, _ = payload
            modes = sorted(modes, key=lambda mode: ((mode.width, mode.height) != tuple(source_size),
                           mode.pixel_format != 'MJPG', abs(mode.fps - 30)))
            self.modes.blockSignals(True)
            for mode in modes:
                self.modes.addItem(f'{mode.width} × {mode.height} · {mode.fps:g} fps · {mode.pixel_format}',
                                   (mode.width, mode.height, mode.fps, mode.pixel_format))
            self.modes.blockSignals(False)
            self.state_label.setText('请核对相机身份、标定板与分辨率，再开始验证。')
            self.settings_changed()
        self.host.run_job(lambda: probe_device(path), done, lambda message: self.device_failed(message, token))

    def settings_changed(self, *_):
        self.generation += 1
        self.reference = None
        self.clear_measurement('设置已改变，当前误差与参考已清空。')
        self.update_reference()
        self.effective = None
        if self.source and self.modes.currentData():
            try:
                parameters = adapt_calibration_resolution(self.source['calibration'],
                    self.modes.currentData()[:2], allow_uniform_scale=self.scaling.isChecked())
                self.show_parameters(parameters)
            except ValueError as exc:
                self.effective_label.setText(str(exc))
        self.update_controls()

    def show_parameters(self, parameters):
        self.effective = parameters
        K, D = parameters['K'], parameters['D']
        width, height = parameters['image_size']
        self.effective_label.setText(f'有效参数 · {width} × {height}\n'
            f'fx {K[0][0]:.3f}　fy {K[1][1]:.3f}\ncx {K[0][2]:.3f}　cy {K[1][2]:.3f}\n'
            + 'D [' + ', '.join(f'{value:.5g}' for value in D) + ']')

    def update_controls(self, *_):
        unlocked = not self.active and not self.hardware_busy
        for control in (self.load_button, self.devices, self.modes, self.refresh_button, self.identity, self.scaling):
            control.setEnabled(unlocked)
        self.start_button.setText('停止实时验证' if self.active else '开始实时验证')
        self.start_button.setEnabled(self.active or bool(unlocked and self.source and self.modes.currentData() and self.identity.isChecked()))
        fresh = self.current is not None and self.active and time.monotonic() - self.measured_at <= 1.0
        self.reference_button.setEnabled(fresh)
        self.export_button.setEnabled(fresh)

    def toggle(self):
        if self.active:
            self.stop()
            return
        if not self.source or not self.identity.isChecked() or not self.modes.currentData():
            self.host.error('请先加载标定结果、选择视频模式并确认相机身份。')
            return
        try:
            width, height, fps, pixel_format = self.modes.currentData()
            self.show_parameters(adapt_calibration_resolution(self.source['calibration'], (width, height),
                                 allow_uniform_scale=self.scaling.isChecked()))
            self.controller.start(self.devices.currentData(), width, height, fps, pixel_format)
        except Exception as exc:
            self.host.error(str(exc))
            return
        self.active = True
        self.generation += 1
        self.last_frame_at = time.monotonic()
        self.last_analysis = 0.0
        self.actual = {}
        self.clear_measurement('正在连接相机，等待视频画面…')
        self.update_controls()
        self.timer.start()
        self.host.logger.info('开始固定内参实时验证：%s', self.devices.currentText())

    def poll(self):
        if not self.active:
            return
        try:
            events = self.controller.poll()
        except Exception as exc:
            self.stop()
            self.host.error('相机取流失败：' + str(exc))
            return
        for event in events:
            if event['type'] == 'opened':
                self.actual = event['actual']
            elif event['type'] in ('closed', 'error'):
                self.stop()
                if event['type'] == 'error':
                    self.host.error(event['message'])
                return
            elif event['type'] == 'frame':
                self.receive_frame(event)
                if not self.active:
                    return
        now = time.monotonic()
        if now - self.last_frame_at > 1.0:
            self.clear_measurement('暂未收到新画面，当前误差无效。')
        elif self.current is not None and now - self.measured_at > 1.0:
            self.clear_measurement('当前检测结果已过期，等待新一帧验证。')

    def receive_frame(self, event):
        frame = event['frame']
        now = time.monotonic()
        captured = event.get('monotonic', now)
        self.last_frame_at = captured
        size = (frame.shape[1], frame.shape[0])
        if size != self.frame_size:
            try:
                parameters = adapt_calibration_resolution(self.source['calibration'], size,
                                                         allow_uniform_scale=self.scaling.isChecked())
            except ValueError as exc:
                self.stop()
                self.host.error('相机实际输出尺寸不匹配：' + str(exc))
                return
            self.frame_size = size
            self.generation += 1
            self.reference = None
            self.clear_measurement('已核对实际输出尺寸，等待验证。')
            self.update_reference()
            self.show_parameters(parameters)
        self.actual_label.setText(f'实际画面：{size[0]} × {size[1]} · {event.get("measured_fps", 0):.1f} 帧 / 秒 · 原图像素单位')
        pixels = frame[::16, ::16]
        if float(pixels.mean()) < 28 and float(pixels.std()) < 2.0:
            self.generation += 1
            self.clear_measurement('画面近乎纯黑，请检查遮挡、照明和相机选择。', frame)
            return
        if self.analysis_token is not None or now - self.last_analysis < 0.2:
            return
        self.last_analysis = now
        token = object()
        self.analysis_token = token
        generation = self.generation
        source, spec, scaling = self.source['calibration'], self.spec, self.scaling.isChecked()
        def work():
            try:
                detection = detect_board(frame, spec)
                if detection is None:
                    return None, None, '未识别到足够标定板角点，请检查板型、光照和视角。'
                return detection, validate_frame(detection, source, size, allow_uniform_scale=scaling), None
            except ValueError as exc:
                return None, None, str(exc)
        def done(payload):
            if self.analysis_token is token:
                self.analysis_token = None
            if not self.active or generation != self.generation:
                return
            if time.monotonic() - captured > 1.0:
                self.clear_measurement('本帧检测耗时较长，结果已过期，请等待新画面。', frame)
                return
            detection, result, message = payload
            if result is None:
                self.clear_measurement(message, frame)
                return
            self.current = dict(result, captured_at=event.get('captured_at'),
                                validated_at=datetime.now().astimezone().isoformat(timespec='seconds'))
            self.measured_at = captured
            self.display_frame, self.display_detection = frame, detection
            self.render_measurement()
        def failed(message):
            if self.analysis_token is token:
                self.analysis_token = None
            if self.active and generation == self.generation:
                self.clear_measurement('本帧检测失败：' + message, frame)
        self.host.run_job(work, done, failed)

    def clear_measurement(self, message, frame=None):
        self.current = None
        self.display_frame = self.display_detection = None
        self.mean_label.setText('平均像素误差  — px')
        self.details.setText('RMS — px    最大误差 — px    角点 —')
        self.state_label.setText(message)
        self.preview.set_frame(frame, guide=dict(color='amber', status='当前帧暂无有效误差', instruction=message,
                                               show_banner=False))
        self.update_reference()
        self.update_controls()

    def render_measurement(self, *_):
        if not self.current:
            return
        if time.monotonic() - self.measured_at > 1.0:
            self.clear_measurement('当前检测结果已过期，等待新画面。')
            return
        value = self.current
        self.mean_label.setText(f'平均像素误差  {value["mean_px"]:.3f} px')
        self.details.setText(f'RMS {value["rms_px"]:.3f} px    最大 {value["max_px"]:.3f} px    {value["point_count"]} 个角点')
        below = value['rms_px'] <= self.limit.value()
        status = '本帧 RMS 低于参考线' if below else '本帧 RMS 超过参考线'
        self.state_label.setText(f'{status} {self.limit.value():.2f} px；只提示当前视角，不能单独确认参数有效。')
        self.preview.set_frame(self.display_frame, self.display_detection,
            dict(color='green' if below else 'amber', status=status,
                 show_banner=False,
                 instruction='检测点与重投影十字属于同一帧；连线表示像素残差。',
                 projected=np.asarray(value['projected_points'])))
        self.update_reference()
        self.update_controls()

    def update_reference(self):
        if self.reference is None:
            self.reference_label.setText('参考尚未记录。调焦前保持标定板稳定，点击下方按钮记录当前帧。')
            return
        ref = self.reference
        text = f'参考帧：平均 {ref["mean_px"]:.3f} px / RMS {ref["rms_px"]:.3f} px'
        if self.current:
            text += f'　当前变化：平均 {self.current["mean_px"] - ref["mean_px"]:+.3f} px / RMS {self.current["rms_px"] - ref["rms_px"]:+.3f} px'
        self.reference_label.setText(text + '\n仅在相近位置、倾角及同分辨率下比较；参考只保存在本次运行。')

    def record_reference(self):
        if not self.current or time.monotonic() - self.measured_at > 1.0:
            self.clear_measurement('当前帧已过期，请等待新的有效误差。')
            return
        self.reference = {key: self.current[key] for key in ('mean_px', 'rms_px', 'max_px', 'point_count', 'image_size', 'validated_at', 'rvec', 'tvec')}
        self.update_reference()
        self.host.notice('已记录调焦前参考，未写入磁盘。保持相近姿态后观察误差变化。')

    def export_current(self):
        if not self.current or time.monotonic() - self.measured_at > 1.0:
            self.clear_measurement('当前帧已过期，请等待新的有效误差再导出。')
            return
        # 在弹出保存对话框前截取当前记录，避免对话框期间变化混入同一份报告。
        record = dict(schema_version=1, type='fixed_intrinsics_live_validation',
            exported_at=datetime.now().astimezone().isoformat(timespec='seconds'), source=self.source,
            effective_calibration=self.effective, current_frame=self.current, reference=self.reference,
            device=dict(name=self.devices.currentText(), path=self.devices.currentData(), actual=self.actual),
            identity_confirmed_by_user=self.identity.isChecked(),
            uniform_scale_confirmed_by_user=self.scaling.isChecked(), rms_reference_px=self.limit.value(),
            method='固定 K、D，仅以本帧标定板角点估计板位姿；均值为二维像素残差距离算术平均。',
            limitations='未确认样本独立性，不自动视为留出验证；参考线并非精度验收；正面姿态可能掩盖焦距变化。')
        try:
            path = self.host.choose_file('导出当前帧验证记录', f'实时验证_{datetime.now():%Y%m%d_%H%M%S}.json', 'JSON 文件 (*.json)')
            if path:
                self.host.write_atomic(path, json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
                self.host.notice(f'已导出当前帧验证记录：{path.name}')
                self.host.logger.info('用户导出固定内参实时验证记录')
        except Exception as exc:
            self.host.error(str(exc))
