"""手眼标定的连续相机预览；复用已有设备后端，不写入照片。"""
from __future__ import annotations

import json
from pathlib import Path
import time
import cv2
from PySide6.QtCore import Qt, QTimer, QEvent, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDialog, QFileDialog

# 画面覆盖层与预览占位同用「色盲友好橙」方案色，不另写色值。
import 界面主题 as 主题
from 核心算法.相机接入 import CaptureController, discover_devices, probe_device
from 标定界面 import label, button, MainWindow as CameraCalibrationWindow
from 预览组件 import image_from_array
from 画面区域 import RegionPreview, PolygonCropDialog, validate_region
from 配对数据 import write_json

方案 = 主题.取方案('色盲友好橙')


class CameraPanel(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.controller = CaptureController()
        self.active = self.busy = False
        self.generation = 0
        self.stream_generation = 0
        self.display_regions = {}
        self.frame = None
        self.last_frame_at = 0.0
        self.frame_serial = 0
        self.frame_captured_at = ''
        self.display_image = None
        self.guide_target = None
        self.detected_center = None
        self.detected_at = 0.0
        self.preview_area = self.preview_splitter = None
        self._preview_layout_key = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        row = QHBoxLayout()
        self.devices = QComboBox()
        self.devices.setMinimumWidth(200)
        self.refresh_button = button('刷新相机', self.scan)
        row.addWidget(self.devices, 1)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.modes = QComboBox()
        self.open_button = button('打开相机', self.toggle)
        row.addWidget(self.modes, 1)
        row.addWidget(self.open_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.crop_button = button('圈选保留区域', self.edit_display_region)
        self.restore_crop_button = button('恢复完整画面', self.restore_display_region)
        self.save_crop_button = button('保存区域', self.save_display_region)
        self.load_crop_button = button('加载区域', self.load_display_region)
        for item in (self.crop_button, self.restore_crop_button, self.save_crop_button, self.load_crop_button):
            row.addWidget(item)
        layout.addLayout(row)
        self.crop_status = label('打开相机后，可沿轮廓逐点圈选保留区域。', 'muted')
        layout.addWidget(self.crop_status)
        self.view = RegionPreview(self)
        self.view.setText('打开相机，预览标定板\n相机安装在机械臂上，标定板保持固定')
        self.view.setFixedSize(360, 230)
        layout.addWidget(self.view, 0, Qt.AlignHCenter | Qt.AlignTop)
        self.status = label('尚未打开相机。开始自动采集后，停稳即可采样。', 'muted')
        layout.addWidget(self.status)
        layout.addStretch()
        self.devices.currentIndexChanged.connect(self.device_changed)
        self.timer = QTimer(self)
        self.timer.setInterval(45)
        self.timer.timeout.connect(self.poll)
        self.controls()

    def controls(self):
        available = self.configuration_available()
        can_select = not self.active and not self.busy and available and not self.host.samples
        self.devices.setEnabled(can_select)
        self.modes.setEnabled(can_select)
        self.refresh_button.setEnabled(can_select)
        self.open_button.setText('关闭相机' if self.active else '打开相机')
        self.open_button.setEnabled(self.active or (available and not self.busy and bool(self.modes.currentData())))
        self.refresh_display_region()

    def configuration_available(self):
        return not any(getattr(self.host, name, False) for name in (
            'busy', 'robot_connecting', 'can_configuring', 'auto_running',
            'auto_checking', 'capture_inflight', 'solve_after_capture', 'closed'))

    def display_region_key(self):
        if (not self.active or self.frame is None or not self.devices.currentData()
                or not 0 <= time.monotonic() - self.last_frame_at <= 2):
            return None
        return (self.devices.currentData(), self.frame.shape[1], self.frame.shape[0])

    def region_context(self):
        return self.generation, self.stream_generation, self.display_region_key()

    def refresh_display_region(self):
        key = self.display_region_key()
        region = self.display_regions.get(key)
        available = self.configuration_available() and not self.busy
        self.crop_button.setEnabled(key is not None and available)
        self.load_crop_button.setEnabled(key is not None and available)
        self.restore_crop_button.setEnabled(region is not None and available)
        self.save_crop_button.setEnabled(region is not None and available)
        if key is None:
            text = '收到实时画面后可圈选；保存区域后可在下次加载。'
        elif region is None:
            text = '当前显示完整画面 · 圈选只改变预览，标定与采样仍用完整原图。'
        else:
            text = f'已保留 {len(region["points"])} 点轮廓 · 完整等比显示 · 标定与采样仍用完整原图。'
        self.crop_status.setText(text)

    def edit_display_region(self):
        context = self.region_context()
        key = context[2]
        if key is None or not self.configuration_available() or self.busy:
            return
        try:
            dialog = PolygonCropDialog(image_from_array(self.frame), self.display_regions.get(key), self)
            if dialog.exec() != QDialog.Accepted:
                return
            if context != self.region_context() or not self.configuration_available():
                raise ValueError('相机画面或采集状态已变化，请在实时画面上重新圈选。')
            self.display_regions[key] = validate_region(dialog.region, key[1:])
            self.refresh_display_region()
            self.paint_image()
            self.host.notice('已应用保留区域；仅本次运行有效，需要保留时请点击“保存区域”。')
            self.host.logger.info('应用手眼相机显示区域')
        except Exception as exc:
            self.host.error(str(exc))

    def restore_display_region(self):
        key = self.display_region_key()
        if key is None or not self.configuration_available() or self.busy:
            return
        self.display_regions.pop(key, None)
        self.refresh_display_region()
        self.paint_image()
        self.host.notice('已恢复完整画面；已保存的区域文件保持不变。')
        self.host.logger.info('恢复手眼相机完整预览')

    def save_display_region(self):
        context = self.region_context()
        key = context[2]
        region = self.display_regions.get(key)
        if region is None or not self.configuration_available() or self.busy:
            return
        record = {'type': '相机预览显示区域', 'version': 1, 'device': key[0],
                  'camera_name': self.devices.currentText(), 'region': validate_region(region)}
        try:
            path = self.host.choose_file('保存相机显示区域', '相机显示区域.json', 'JSON 文件 (*.json)')
            if not path:
                return
            if context != self.region_context() or not self.configuration_available():
                raise ValueError('相机画面或采集状态已变化，请重新保存区域。')
            write_json(path, record)
            self.host.notice(f'显示区域已保存：{Path(path).name}。下次打开相机后可点击“加载区域”。')
            self.host.logger.info('用户保存手眼相机显示区域')
        except Exception as exc:
            self.host.logger.exception('保存显示区域失败')
            self.host.error(str(exc))

    def load_display_region(self):
        context = self.region_context()
        key = context[2]
        if key is None or not self.configuration_available() or self.busy:
            return
        value, _ = QFileDialog.getOpenFileName(self, '加载相机显示区域',
                                              str(self.host.project / '结果'), 'JSON 文件 (*.json)')
        if not value:
            return
        try:
            path = Path(value)
            if path.stat().st_size > 1024 * 1024:
                raise ValueError('区域文件超过 1 MiB，请选择“保存区域”生成的 JSON 文件。')
            record = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(record, dict) or record.get('type') != '相机预览显示区域' or record.get('version') != 1:
                raise ValueError('请选择相机标定或手眼标定通过“保存区域”保存的文件。')
            if record.get('device') != key[0]:
                raise ValueError('区域文件记录的相机与当前设备不同，请选择对应相机或重新圈选。')
            region = validate_region(record.get('region'), key[1:])
            if context != self.region_context() or not self.configuration_available():
                raise ValueError('相机画面或采集状态已变化，请在实时画面上重新加载区域。')
            self.display_regions[key] = region
            self.refresh_display_region()
            self.paint_image()
            self.host.notice('已加载保留区域；仅改变预览，标定与采样仍用完整原图。')
            self.host.logger.info('用户加载手眼相机显示区域')
        except Exception as exc:
            self.host.logger.exception('加载显示区域失败')
            self.host.error(str(exc))

    def scan(self):
        if self.active or self.busy or self.host.samples or getattr(self.host, 'auto_running', False):
            return
        self.busy = True
        self.generation += 1
        token = self.generation
        self.controls()
        self.status.setText('正在查找视频设备…')

        def done(devices):
            if token != self.generation:
                return
            self.busy = False
            self.devices.blockSignals(True)
            self.devices.clear()
            for device in devices:
                self.devices.addItem(CameraCalibrationWindow.camera_display_name(device), device.path)
                self.devices.setItemData(self.devices.count() - 1, f'{device.name}\n{device.path}', Qt.ToolTipRole)
            self.devices.blockSignals(False)
            self.device_changed()

        self.host.run_job(discover_devices, done, lambda msg: self.failed(msg, token))

    def failed(self, message, token):
        if token != self.generation:
            return
        self.busy = False
        self.status.setText('设备读取失败：' + message)
        self.controls()

    def device_changed(self, *_):
        self.generation += 1
        token = self.generation
        self.modes.clear()
        self.host.identity.setChecked(False)
        path = self.devices.currentData()
        self.busy = bool(path)
        self.controls()
        if not path:
            self.status.setText('未发现视频设备，请连接相机后刷新。')
            return

        def done(payload):
            if token != self.generation:
                return
            self.busy = False
            size = self.host.camera_record['calibration']['image_size'] if self.host.camera_record else [1280, 720]
            modes, _ = payload
            modes = sorted(modes, key=lambda m: ((m.width, m.height) != tuple(size), m.pixel_format != 'MJPG', abs(m.fps - 30)))
            for mode in modes:
                self.modes.addItem(f'{mode.width} × {mode.height} · {mode.fps:g} fps · {mode.pixel_format}',
                                   (mode.width, mode.height, mode.fps, mode.pixel_format))
            self.status.setText('请选择与内参标定相同的相机、视频流和分辨率。')
            self.controls()

        self.host.run_job(lambda: probe_device(path), done, lambda msg: self.failed(msg, token))

    def toggle(self):
        if self.active:
            self.stop()
            return
        try:
            if not self.host.camera_record:
                raise ValueError('请先在“准备参数”读取相机标定结果。')
            if not self.host.identity.isChecked():
                raise ValueError('请先核对并勾选相机、镜头状态及标定板一致。')
            mode = self.modes.currentData()
            if not mode:
                raise ValueError('请选择可用视频模式。')
            if tuple(mode[:2]) != tuple(self.host.camera_record['calibration']['image_size']):
                raise ValueError('当前视频分辨率与内参不一致，请选择原标定分辨率。')
            self.controller.start(self.devices.currentData(), *mode)
            self.stream_generation += 1
            self.active = True
            self.frame = None
            self.frame_captured_at = ''
            self.last_frame_at = time.monotonic()
            self.timer.start()
            self.status.setText('正在连接相机…')
            self.host.logger.info('打开手眼标定相机')
        except Exception as exc:
            self.host.error(str(exc))
        self.controls()
        self.host.update_state()

    def stop(self):
        self.stream_generation += 1
        self.active = False
        if getattr(self.host, 'auto_running', False):
            self.host.stop_auto('相机已关闭')
        self.timer.stop()
        self.controller.stop()
        self.frame = None
        self.last_frame_at = 0.0
        self.frame_captured_at = ''
        self.display_image = None
        self.view.clear()
        self.view.setText('相机已关闭')
        self.fit_preview()
        self.status.setText('相机已关闭；已采集的样本仍保留在本次运行中。')
        self.host.logger.info('关闭手眼标定相机')
        self.controls()
        self.host.update_state()

    def poll(self):
        try:
            for event in self.controller.poll():
                if event['type'] in ('error', 'closed'):
                    self.stop()
                    self.status.setText(event.get('message', '视频流已关闭。'))
                    return
                if event['type'] == 'frame':
                    frame = event['frame']
                    size = [frame.shape[1], frame.shape[0]]
                    if size != self.host.camera_record['calibration']['image_size']:
                        self.stop()
                        self.host.error('相机实际输出分辨率与标定内参不一致，已停止取流。')
                        return
                    self.frame = frame
                    # 后端在 camera.read() 返回后记录主机时间，不代表硬件曝光同步。
                    self.last_frame_at = event.get('monotonic', time.monotonic())
                    self.frame_captured_at = event.get('captured_at', '')
                    self.frame_serial += 1
                    self.show_frame(frame)
                    self.status.setText(f'实时画面 {size[0]} × {size[1]} · 请保持标定板固定')
            if time.monotonic() - self.last_frame_at > 2:
                self.frame = None
                self.frame_captured_at = ''
                self.display_image = None
                self.view.clear()
                self.view.setText('暂未收到新画面')
                self.fit_preview()
                if getattr(self.host, 'auto_running', False):
                    self.host.stop_auto('相机暂时无新画面')
                self.status.setText('视频暂时无新画面，请检查相机连接后重新开始采集。')
            self.controls()
        except Exception as exc:
            self.stop()
            self.host.error('取流失败：' + str(exc))

    def show_frame(self, frame, corners=None):
        if corners:
            frame = frame.copy()
            for x, y in corners:
                cv2.circle(frame, (round(x), round(y)), 4, (80, 220, 100), -1)
        self.display_image = image_from_array(frame)
        self.paint_image()

    def set_guidance(self, target):
        self.guide_target = target
        self.paint_image()

    def set_detection(self, center, sampled_at):
        self.detected_center, self.detected_at = center, sampled_at
        self.paint_image()

    def paint_image(self):
        self.view.image = self.display_image
        self.view.display_region = self.display_regions.get(self.display_region_key())
        self.view.guide_target = self.guide_target
        self.view.detected_center, self.view.detected_at = self.detected_center, self.detected_at
        self.fit_preview()
        self.view.update()

    def attach_preview_layout(self, area, splitter):
        self.preview_area, self.preview_splitter = area, splitter
        self._camera_minimum_width = splitter.widget(0).minimumSizeHint().width()
        area.viewport().installEventFilter(self)
        QTimer.singleShot(0, self.fit_preview)

    def fit_preview(self):
        """画布随圈选比例伸展；按窗口可见高度分配左右宽度，整幅显示。"""
        if self.preview_area is None:
            return
        viewport = self.preview_area.viewport()
        splitter = self.preview_splitter
        camera_card = splitter.widget(0)
        # 使用内容原点，整页滚动查看样本时不改变图像尺寸。
        top = self.view.mapTo(viewport, QPoint()).y() + self.preview_area.verticalScrollBar().value()
        status_height = max(self.status.sizeHint().height(), self.status.heightForWidth(max(1, self.width())))
        max_height = max(80, viewport.height() - top - status_height - 36)
        bounds, _ = self.view._content_bounds()
        aspect = bounds.width() / bounds.height()
        margin = max(0, camera_card.width() - self.width())
        points = tuple(tuple(point) for point in (self.view.display_region or {}).get('points', []))
        key = (viewport.width(), viewport.height(), bounds.width(), bounds.height(), points, self.view.image is not None)
        available_width = self.width()
        sizes = None
        if self.view.image is not None and key != self._preview_layout_key:
            self._preview_layout_key = key
            right_minimum = splitter.widget(1).minimumSizeHint().width()
            total = max(1, viewport.width() - splitter.handleWidth())
            ideal = round(24 + max(1, max_height - 24) * aspect) + margin
            camera_width = max(self._camera_minimum_width, min(ideal, total - right_minimum))
            available_width = max(25, camera_width - margin)
            sizes = [camera_width, max(right_minimum, total - camera_width)]
        if self.view.image is None:
            width, height = max(25, available_width), min(230, max_height)
        else:
            width = max(25, min(available_width, round(24 + max(1, max_height - 24) * aspect)))
            height = self.view.heightForWidth(width)
        if self.view.width() != width or self.view.height() != height:
            self.view.setFixedSize(width, height)
        if sizes is not None:
            # 先释放旧画布带来的最小宽度，再调整分栏，避免圈选后留下旧列宽。
            self.layout().activate()
            self.updateGeometry()
            camera_card.layout().activate()
            camera_card.updateGeometry()
            splitter.setSizes(sizes)

    def eventFilter(self, watched, event):
        if self.preview_area is not None and event.type() == QEvent.Resize and watched is self.preview_area.viewport():
            QTimer.singleShot(0, self.fit_preview)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.fit_preview)
