"""同一右臂、同一相机上的标定板对点验证；只有用户执行才驱动机械臂。"""
import copy
from datetime import datetime
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk
from scipy.spatial.transform import Rotation

from 标定板验证 import (BOARD_KINDS, DEFAULT_SPEC, SYSTEM_SCOPE, board_geometry,
                       estimate_board, load_intrinsics, target_point,
                       image_to_board, project_board_point, base_board_chain)
from 目标变换 import transform, tool_pose, travel_targets
from 探针运动 import (MotionExecutor, plan_tcp_path, MOTION_EXECUTION_HOLD_REASON,
                    BOARD_MIN_GAP_M)
from 控制诊断 import AutomaticControlDiagnostics


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


class BoardPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.colors['bg'], padx=18, pady=14)
        self.app, self.c = app, app.colors
        self.events = queue.Queue()
        self.detect_events = queue.Queue(maxsize=2)
        self.detect_stop = threading.Event()
        self.detect_generation = self.generation = 0
        self.detecting = self.planning = self.motion_busy = False
        self.plan_worker = self.control_worker = None
        self.preparing_control = False
        self.control_ready = self.direction_ready = False
        self.motion_kind = None
        self.result = self.locked_target = self.plan = self.executor = None
        self.camera_record = self.render_map = self.shown_result = None
        self.last_good = 0.0
        self.last_goal = None
        self.last_commanded_goal = None
        self.last_motion_state = '未执行'
        self.automatic = AutomaticControlDiagnostics(self.app.project_dir / '日志' / '运动诊断')
        self.diagnostic_exporting = False
        self.diagnostic_events = queue.Queue()
        self.diagnostic_path = None
        self.export_job = None
        self.finish_reason = None
        self.save_error = None
        self.save_retry_after = 0.0
        self.selected_point = None
        self.final_result = None
        self.board_name = tk.StringVar(value=next(n for n, k in BOARD_KINDS.items() if k == 'aprilgrid'))
        self.fields = {k: tk.StringVar(value=str(DEFAULT_SPEC[k])) for k in
                       ('columns', 'rows', 'square_mm', 'marker_mm', 'gap_mm', 'first_id')}
        self.dictionary = tk.StringVar(value='DICT_4X4_1000')
        self.dictionary_display = tk.StringVar(value='DICT_APRILTAG_36h11')
        self.approach = tk.StringVar(value='10')
        self.gap = tk.StringVar(value='5')
        self.step = tk.StringVar(value='0.5')
        self.clearance = tk.StringVar(value='50')
        self._build()
        for var in (self.board_name, self.dictionary, *self.fields.values()):
            var.trace_add('write', self._board_changed)
        for var in (self.approach, self.gap, self.step, self.clearance):
            var.trace_add('write', self._motion_changed)
        self._field_states()
        self.after(120, self._poll)

    @property
    def busy(self):
        return (self.preparing_control or self.planning or self.motion_busy
                or bool(self.executor and self.executor.running)
                or bool(self.plan_worker and self.plan_worker.is_alive())
                or bool(self.control_worker and self.control_worker.is_alive()))

    def _row(self, parent):
        row = tk.Frame(parent, bg=self.c['panel'])
        row.pack(fill='x', pady=4)
        return row

    def _note(self, parent, text, color='muted'):
        label = self.app._label(parent, text, 10, color, justify='left', anchor='w', wraplength=440)
        label.pack(fill='x', pady=4)
        label.bind('<Configure>', lambda e: label.configure(wraplength=max(180, e.width-6)))
        return label

    def _build(self):
        self.columnconfigure(0, weight=3, uniform='board_columns')
        self.columnconfigure(1, weight=2, minsize=475, uniform='board_columns')
        self.rowconfigure(1, weight=1)
        head = tk.Frame(self, bg=self.c['bg'])
        head.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 12))
        self.app._label(head, '标定板对点验证', 18, bold=True).pack(side='left')
        connection_text = getattr(self.app, 'robot_connection_text', lambda: '连接右臂 can1')()
        self.robot_button = self.app._button(head, connection_text, self.app.toggle_robot, compact=True)
        self.robot_button.pack(side='right')
        self.app._button(head, '打开相机', self.app.camera.start, compact=True).pack(side='right', padx=10)
        outer, left = self.app._card(self, 14)
        outer.grid(row=1, column=0, sticky='nsew', padx=(0, 14))
        self.app._label(left, '实时画面 · 点击板面选点', 13, bold=True).pack(anchor='w')
        row = self._row(left)
        self.detect_button = self.app._button(row, '开始识别标定板', self._start_detection, primary=True, compact=True)
        self.detect_button.pack(side='left')
        self.app._button(row, '停止识别', self._stop_detection, compact=True).pack(side='left', padx=8)
        self.canvas = tk.Canvas(left, bg=self.c['selected'], highlightthickness=1,
                                highlightbackground=self.c['muted'], height=330)
        self.canvas.pack(fill='both', expand=True, pady=8)
        self.canvas.bind('<Button-1>', self._click)
        self.video_note = self._note(left, '打开相机，填写实际板型和尺寸，再开始识别。')
        self.quality_label = self._note(left, '审查：至少 16 个角点、重投影 RMS ≤ 1 px，并检查特征分布。')
        self.target_label = self._note(left, '请在画面点击目标。', 'primary')
        self.state_label = self._note(left, '点选目标后，规划路径，再执行运动。', 'primary')
        outer, right = self.app._card(self, 12)
        outer.grid(row=1, column=1, sticky='nsew')
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill='both', expand=True)
        motion, settings = [tk.Frame(self.tabs, bg=self.c['panel'], padx=12, pady=12) for _ in range(2)]
        self.tabs.add(motion, text='对点验证')
        self.tabs.add(settings, text='标定设置')
        self.tabs.select(motion)
        from 界面主题 import Tk滚动区
        settings = Tk滚动区(settings, width=410)
        row = self._row(settings)
        self.app._button(row, '载入 TCP', self._choose_tcp, compact=True).pack(side='left', fill='x', expand=True)
        self.app._button(row, '载入手眼', self._choose_handeye, compact=True).pack(side='left', fill='x', expand=True, padx=(8, 0))
        self.calibration_label = self._note(settings, '与“选孔与运动”页共享 TCP、手眼结果。')
        row = self._row(settings)
        self.app._button(row, '载入相机内参', self._choose_intrinsics, compact=True).pack(side='left')
        self.app._button(row, '使用设备内参', self._device_intrinsics, compact=True).pack(side='left', padx=8)
        self.intrinsics_label = self._note(settings, '使用当前 RealSense 彩色流内参；外部内参须匹配相机序列号与分辨率。')
        self.app._label(settings, '实际标定板', 12, bold=True).pack(anchor='w', pady=(12, 4))
        ttk.Combobox(settings, values=list(BOARD_KINDS), textvariable=self.board_name,
                     state='readonly').pack(fill='x', pady=4)
        grid = tk.Frame(settings, bg=self.c['panel']); grid.pack(fill='x', pady=6)
        self.field_widgets = {}
        for i, (name, key) in enumerate((('列数', 'columns'), ('行数', 'rows'), ('方格边 mm', 'square_mm'),
                                       ('标签边 mm', 'marker_mm'), ('标签间隙 mm', 'gap_mm'), ('起始 ID', 'first_id'))):
            self.app._label(grid, name, 10).grid(row=i//2, column=(i%2)*2, sticky='w', pady=5)
            entry = ttk.Entry(grid, textvariable=self.fields[key], width=8)
            entry.grid(row=i//2, column=(i%2)*2+1, padx=(6, 12), pady=5)
            self.field_widgets[key] = entry
        self.dictionary_combo = ttk.Combobox(settings, textvariable=self.dictionary, state='readonly',
            values=['DICT_4X4_1000', 'DICT_5X5_1000', 'DICT_6X6_1000', 'DICT_7X7_1000', 'DICT_APRILTAG_36h11'])
        self.dictionary_combo.pack(fill='x', pady=4)
        self.spec_note = self._note(settings, '')
        self._note(settings, '尺寸以实物为准，单位 mm。棋盘格行列填写方格数；网格板填写标签数。\n原点由板模型定义，画面会标出 O、X、Y。普通棋盘格需核对 180° 朝向。')
        self._note(settings, '当前 TCP 即使拟合误差较大也可以进入验证。此页不会把“标定通过”当成整机精度通过。', 'primary')

        self.app._label(settings, '路径参数', 12, bold=True).pack(anchor='w', pady=(12, 4))
        grid = tk.Frame(settings, bg=self.c['panel']); grid.pack(fill='x')
        for i, (label, var) in enumerate((('调姿间距 mm', self.approach), ('最终间距 mm', self.gap),
                                         ('抬升间距 mm', self.clearance))):
            self.app._label(grid, label, 10).grid(row=i, column=0, sticky='w', pady=4)
            ttk.Entry(grid, textvariable=var, width=9).grid(row=i, column=1, padx=10, pady=4)
        self._note(settings, '默认在板前 10 mm 调成垂直，最终停在 5 mm；不接触板面。')

        actions = tk.Frame(motion, bg=self.c['panel'])
        actions.pack(side='bottom', fill='x', pady=(18, 0))
        self.stop_button = self.app._button(actions, '停止运动（电子急停）', self.stop_motion, danger=True)
        self.stop_button.pack(fill='x')
        self._note(actions, '整条路径保持非接触，最小板距 5 mm。')
        self.app._label(motion, '选点 → 规划 → 执行', 16, bold=True).pack(anchor='w', pady=(4, 12))
        self._note(motion, '点击左侧画面选择目标，规划完成后即可执行整条路径。')
        row = self._row(motion)
        self.preview_button = self.app._button(row, '规划路径', self._lock_and_preview, primary=True)
        self.preview_button.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.execute_button = self.app._button(row, '执行运动', self._execute, primary=True)
        self.execute_button.pack(side='left', fill='x', expand=True)
        self.path_text = self._note(motion, '尚未规划路径。')
        result_box = tk.Frame(motion, bg=self.c['selected'], padx=14, pady=12)
        result_box.pack(fill='x', pady=(18, 12))
        result_line = tk.Frame(result_box, bg=self.c['selected'])
        result_line.pack(fill='x')
        self.app._label(result_line, '板面投影偏差', 13, bold=True).pack(side='left')
        self.result_value = self.app._label(result_line, '— mm', 20, 'text', bold=True)
        self.result_value.pack(side='right')
        self.result_note = self._note(result_box, '执行结束后自动显示 · 反馈估算')
        self._note(motion, '针尖在板面上的垂直投影与所选点的距离。\n由反馈及标定换算，不是独立实测。')
        self.diagnostic_status = self._note(motion, '选点后自动记录，结束或异常时自动保存。')
        self.app._button(motion, '打开日志', self.app.open_log, compact=True).pack(anchor='w', pady=4)
        note = self.app._label(self, '板距和投影偏差按标定坐标计算；刚性针没有接触检测。', 10, 'muted')
        note.grid(row=2, column=0, columnspan=2, pady=(10, 0))

    def _lock_and_preview(self):
        try:
            self._idle()
            if self.selected_point is None:
                raise ValueError('请先在画面上选择目标点。')
            self._ensure_recording()
            self._lock_target()
            if self.locked_target is not None:
                self._prepare()
        except Exception as exc:
            self._tell(str(exc), True)
            self._end_recording('规划失败：' + str(exc))

    def _diagnostic_context(self):
        reader = self.app.robot_reader
        detection = ({k: v for k, v in self.result.items() if k != 'image_rgb'}
                     if self.result is not None else None)
        try: board = self._spec()
        except Exception as exc: board = {'parameter_error': str(exc)}
        try:
            reader_context = reader.diagnostic_context() if callable(getattr(reader, 'diagnostic_context', None)) else None
        except Exception as exc:
            reader_context = {'read_error': str(exc)}
        return serializable(dict(robot=self.app.robot_identity(), selected_point_board_m=self.selected_point,
            board=board, detection=detection, locked_target=self.locked_target,
            plan=self.plan, final_result=self.final_result, last_motion_state=self.last_motion_state,
            feedback=self.app.live_snapshot,
            reader_context=reader_context))

    def _ensure_recording(self):
        if self.diagnostic_exporting or self.finish_reason:
            raise ValueError('上一轮过程记录正在保存，请稍候。')
        self.automatic.begin(self.app.robot_reader, self._diagnostic_context())
        self.save_error = None

    def _end_recording(self, reason):
        if self.automatic.active:
            self.automatic.event('流程结束请求', {'reason': reason})
            self.finish_reason = self.finish_reason or reason

    def _save_automatic(self):
        if self.diagnostic_exporting or self.busy or not self.events.empty() or time.monotonic() < self.save_retry_after:
            return
        if self.finish_reason:
            try:
                self.export_job = self.automatic.finish(self.finish_reason, self._diagnostic_context())
            except Exception as exc:
                self.save_error = str(exc)
                self.save_retry_after = time.monotonic() + 5
                self.app.logger.exception('结束自动诊断失败')
                return
            self.finish_reason = None
        job = self.export_job
        if job is None or job.saved:
            return
        self.diagnostic_exporting = True
        def worker():
            try:
                path = job.export()
                self.diagnostic_events.put((job, path, None))
            except Exception as exc:
                self.diagnostic_events.put((job, None, str(exc)))
        threading.Thread(target=worker, name='运动过程自动保存', daemon=True).start()

    def _refresh_diagnostics(self):
        while True:
            try: job, path, error = self.diagnostic_events.get_nowait()
            except queue.Empty: break
            self.diagnostic_exporting = False
            if error:
                self.save_error = error
                self.save_retry_after = time.monotonic() + 5
                self.app.logger.error('自动诊断保存失败：%s', error)
            else:
                self.diagnostic_path = path
                self.save_error = None
                self.save_retry_after = 0.0
                self.app.logger.info('运动过程已自动保存：%s', path)
        state = self.automatic.status()
        if self.automatic.active and not state['running'] and not self.finish_reason:
            reason = state['error'] or '诊断接收已中断，无法继续完整记录。'
            if self.busy: self.stop_motion(reason)
            else:
                self.plan = None
                self._tell(reason, True)
                self._end_recording(reason)
        self._save_automatic()
        if self.save_error:
            text = '过程记录保存失败：' + self.save_error + '；内存记录保留，正在自动重试。'
        elif self.diagnostic_exporting or self.finish_reason:
            text = '正在自动保存本轮过程记录…'
        elif self.automatic.active:
            text = f"自动记录中 · {state['retained_frames']:,} 帧 · {state['retained_events']:,} 个事件"
        elif self.diagnostic_path:
            issues = self.export_job.context.get('capture_issues', []) if self.export_job else []
            text = '过程记录已自动保存到 日志/运动诊断。' + ('记录存在缺口，详见日志。' if issues else '')
        else:
            text = '选点后自动记录，结束或异常时自动保存。'
        self.diagnostic_status.configure(text=text, fg=self.c['error'] if self.save_error else self.c['muted'])

    @property
    def diagnostic_unsaved(self):
        return self.automatic.active or bool(self.export_job and not self.export_job.saved)

    def prepare_close(self):
        self._end_recording('关闭窗口')
        self._refresh_diagnostics()
        return not (self.busy or self.finish_reason or self.diagnostic_exporting or self.diagnostic_unsaved)

    def _tell(self, text, error=False):
        self.state_label.configure(text=text, fg=self.c['error'] if error else self.c['text'])
        self.app._status(text, error)

    def _idle(self):
        if self.busy: raise ValueError('正在规划或执行，请等待完成。')
        reason = self.app.task_blocked('board')
        if reason: raise ValueError(reason)

    def invalidate(self, reason='条件变化，请重新规划。'):
        if self.busy: self.stop_motion(reason)
        self.automatic.event('规划条件变化', {'reason': reason})
        self.generation += 1
        self.locked_target = self.plan = self.last_goal = self.last_commanded_goal = None
        self.last_motion_state = '未执行'
        self.target_label.configure(text='已选目标，请规划路径。' if self.selected_point is not None else '请在画面选择目标。')
        self._path_message('条件变化，请重新规划。')
        self._tell(reason)


    def _board_changed(self, *_):
        self._stop_detection()
        self.selected_point = None
        self._field_states()
        self.invalidate('板参数已变化，请重新识别并核对原点。')

    def _field_states(self):
        kind = BOARD_KINDS[self.board_name.get()]
        enabled = {'columns', 'rows'}
        if kind in ('chessboard', 'charuco'): enabled.add('square_mm')
        if kind != 'chessboard': enabled.update(('marker_mm', 'first_id'))
        if kind in ('aruco', 'aprilgrid'): enabled.add('gap_mm')
        for key, widget in self.field_widgets.items():
            widget.configure(state='normal' if key in enabled else 'disabled')
        uses_dictionary = kind in ('aruco', 'charuco')
        self.dictionary_display.set('DICT_APRILTAG_36h11' if kind == 'aprilgrid' else '普通棋盘格不使用标签字典')
        self.dictionary_combo.configure(state='readonly' if uses_dictionary else 'disabled',
            textvariable=self.dictionary if uses_dictionary else self.dictionary_display)
        try:
            geometry = board_geometry(self._spec())
            self.spec_note.configure(text=f"图案尺寸 {geometry['size_mm'][0]:g} × {geometry['size_mm'][1]:g} mm\n"+geometry['origin_note'])
        except ValueError as exc:
            self.spec_note.configure(text=str(exc))

    def _motion_changed(self, *_):
        self.invalidate('路径参数已变化，请重新规划。')

    def _choose_tcp(self):
        self.app.hole_panel._choose_tcp()

    def _choose_handeye(self):
        self.app.hole_panel._choose_handeye()

    def _choose_intrinsics(self):
        try:
            self._idle()
            path = filedialog.askopenfilename(parent=self, title='载入当前相机内参',
                initialdir=self.app.project_dir.parent/'相机标定'/'结果', filetypes=[('JSON', '*.json')])
            if not path: return
            record = load_intrinsics(path)
            self._stop_detection(); self.camera_record = record
            self.invalidate('内参已更换，请重新识别标定板。')
            self.intrinsics_label.configure(text=f"外部内参 · 相机 {record['serial'] or '缺少序列号，不能用于定位'}\n分辨率 {record['calibration']['image_size']} · {record['calibration']['model']}")
        except Exception as exc: self._tell(str(exc), True)

    def _device_intrinsics(self):
        try:
            self._idle(); self._stop_detection(); self.camera_record = None
            self.invalidate('已切换设备内参，请重新识别标定板。')
            self.intrinsics_label.configure(text='使用当前 RealSense 彩色流内参；这是设备内参，不代表验证已通过。')
        except Exception as exc: self._tell(str(exc), True)

    def _spec(self):
        try:
            spec = {k: int(v.get()) if k in ('rows', 'columns', 'first_id') else float(v.get()) for k, v in self.fields.items()}
        except ValueError:
            raise ValueError('行列数、起始 ID 需为整数，尺寸需为毫米数值。')
        spec.update(kind=BOARD_KINDS[self.board_name.get()], dictionary=self.dictionary.get())
        return spec

    def _start_detection(self):
        try:
            self._idle()
            if self.detecting: return
            spec = self._spec(); board_geometry(spec)
            if not self.app.camera.is_live(): raise ValueError('请先打开相机。')
            self._stop_detection()
            generation = self.detect_generation
            stop = self.detect_stop = threading.Event()
            camera_record = copy.deepcopy(self.camera_record)
            self.detecting = True
            def worker():
                last_id = None
                while not stop.is_set():
                    frame = self.app.camera.get_frame_snapshot()
                    if frame is not None and frame['frame_id'] != last_id:
                        last_id = frame['frame_id']
                        try: data = estimate_board(frame, spec, camera_record)
                        except Exception as exc: data = str(exc)
                        try: self.detect_events.put_nowait((generation, data))
                        except queue.Full: pass
                    stop.wait(.15)
            threading.Thread(target=worker, name='标定板识别', daemon=True).start()
            self._tell('正在识别标定板；可以在画面选点。')
        except Exception as exc: self._tell(str(exc), True)

    def _stop_detection(self):
        self.detect_stop.set(); self.detect_generation += 1; self.detecting = False
        self.result = None; self.last_good = 0
        self.invalidate('识别已停止，运动目标已解锁。')

    def _snapshot(self):
        robot_kind = getattr(self.app, 'robot_kind', None)
        if robot_kind is not None and robot_kind.get() == 'UR（网络连接）':
            raise ValueError('UR 当前支持 TCP 标定采集；本页的定位链与路径仍仅支持 PiPER X。')
        value = self.app.live_snapshot
        if value is None or not 0 <= time.monotonic()-value['monotonic'] <= .3:
            raise ValueError('需要右臂 can1 的新鲜反馈。')
        return copy.deepcopy(value)

    def _fresh_result(self):
        value = self.result
        if not self.detecting or value is None or not value.get('accepted') or time.monotonic()-value['monotonic'] > 1:
            raise ValueError('需要一秒内通过审查的标定板画面。请保持板完整可见。')
        return value

    def _point(self, result):
        if self.selected_point is None: raise ValueError('请在画面上选择目标点。')
        return target_point(result, 'custom', xy_mm=np.asarray(self.selected_point[:2])*1000)

    def _chain(self, result):
        shared = self.app.hole_panel
        if shared.tcp is None or shared.handeye is None:
            raise ValueError('请先载入 TCP 和当前安装的手眼结果。')
        serial = shared.handeye['serial']
        if not serial:
            raise ValueError('手眼文件缺少相机序列号，无法自动匹配；请载入带相机身份的手眼结果。')
        if serial != result['color_serial']:
            raise ValueError('手眼结果的相机序列号与当前相机不同。')
        pose = shared.history.at(result['monotonic'])
        if pose is None: raise ValueError('标定板图像没有匹配到同一时刻的机械臂反馈。')
        return base_board_chain(pose, shared.handeye, shared.tcp['T_flange_tcp'], result['T_cam_board'])

    def _lock_target(self):
        try:
            self._idle(); self._snapshot()
            result = self._fresh_result()
            if not self.app.hole_panel.history.stable(result['monotonic']):
                raise ValueError('请让右臂停稳至少 0.7 秒，再锁定目标。')
            chain = self._chain(result)
            point = self._point(result)
            base_point = (chain['T_base_board'] @ np.r_[point, 1])[:3]
            normal = chain['T_base_camera'][:3, :3] @ np.asarray(result['normal_camera_outward'])
            normal /= np.linalg.norm(normal)
            snapshot = self._snapshot()
            T_tool = self.app.hole_panel.tcp['T_flange_tcp'].copy()
            current = transform(snapshot['T_base_flange']) @ T_tool
            base_goal = tool_pose(base_point, normal, current[:3, :3])
            self.generation += 1
            self.locked_target = dict(result={k: copy.deepcopy(v) for k, v in result.items() if k != 'image_rgb'},
                chain=serializable(chain), point_board_m=point.tolist(), point_base_m=base_point.tolist(),
                normal_base=normal.tolist(), surface_pose=base_goal.tolist(), T_flange_tcp=T_tool.tolist(),
                tcp=copy.deepcopy(self.app.hole_panel.tcp['payload']),
                handeye=copy.deepcopy(self.app.hole_panel.handeye['payload']),
                intrinsics_source=self.camera_record['path'] if self.camera_record else '当前彩色设备内参',
                locked_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                timing='停稳期间按主机收帧时间配对；非曝光硬同步')
            self.plan = self.last_goal = self.last_commanded_goal = None
            self.last_motion_state = '未执行'
            self.target_label.configure(text='目标已选定')
            self.automatic.event('目标锁定', self._diagnostic_context())
        except Exception as exc:
            self.locked_target = None
            self._tell(str(exc), True)
            self._end_recording('目标检查失败：' + str(exc))

    def _parameters(self):
        a, g, s, c = [float(v.get()) for v in (self.approach, self.gap, self.step, self.clearance)]
        if not np.isfinite([a, g, s, c]).all() or not (10 <= a <= 150 and BOARD_MIN_GAP_M*1000 <= g <= a and .1 <= s <= .5 and 20 <= c <= 150):
            raise ValueError('非接触验证：目标间距须为 5 mm 至观察间距，不能戳板；观察间距 10–150 mm；步长 0.1–0.5 mm；抬升 20–150 mm。')
        return dict(approach=a/1000, gap=g/1000, step=s/1000, clearance=c/1000)

    @staticmethod
    def _protect_noncontact_plan(plan, target):
        plan['metadata']['board_noncontact'] = dict(
            point_base_m=list(target['point_base_m']), normal_base=list(target['normal_base']),
            minimum_gap_m=BOARD_MIN_GAP_M)
        MotionExecutor.check_board_noncontact(plan)

    def _unchanged_board(self):
        if self.locked_target is None: raise ValueError('请先锁定目标与坐标链。')
        result = self._fresh_result()
        if result['color_serial'] != self.locked_target['result']['color_serial']:
            raise ValueError('相机发生变化，请重新锁定目标。')
        if not self.app.hole_panel.history.stable(result['monotonic']):
            raise ValueError('请等待右臂停稳，再继续验证。')
        current = self._chain(result)['T_base_board']
        old = transform(self.locked_target['chain']['T_base_board'])
        delta = np.linalg.inv(old) @ current
        if np.linalg.norm(delta[:3, 3]) > .005 or Rotation.from_matrix(delta[:3, :3]).magnitude() > np.deg2rad(2):
            raise ValueError('当前板定位与锁定结果相差超过 5 mm / 2°；请检查板是否移动及标定误差，重新锁定后再运动。')

    def _prepare(self):
        try:
            self._idle(); self._unchanged_board()
            params, snapshot = self._parameters(), self._snapshot()
            frozen = self.locked_target
            tool = transform(frozen['T_flange_tcp'])
            current = transform(snapshot['T_base_flange']) @ tool
            normal = np.asarray(frozen['normal_base'])
            surface = transform(frozen['surface_pose'])
            delta = current[:3, 3] - surface[:3, 3]
            current_gap = float(delta @ normal)
            aligned = (np.linalg.norm(delta-current_gap*normal) <= .0005
                       and Rotation.from_matrix(surface[:3, :3].T@current[:3, :3]).magnitude() <= np.deg2rad(.3))
            normal_only = aligned and BOARD_MIN_GAP_M <= current_gap <= params['approach']
            reference_goal = surface.copy(); reference_goal[:3, 3] += normal*params['gap']
            if normal_only:
                observation = current.copy()
                approach_gap = current_gap
                targets = []
            else:
                observation = surface.copy(); observation[:3, 3] += normal*params['approach']
                approach_gap = params['approach']
                targets = travel_targets(current, observation, params['clearance'], align_after_approach=True)
            # 小段由程序自动连续推进，不再要求用户逐段点击。
            count = max(1, int(np.ceil(abs(approach_gap-params['gap'])/params['step'])))
            for gap in np.linspace(approach_gap, params['gap'], count+1)[1:]:
                goal = observation.copy(); goal[:3, 3] += normal*(gap-approach_gap)
                targets.append(goal)
            free_segments = 0 if normal_only else 4
            self.planning = True; self.generation += 1
            generation = self.generation; self.plan = None; self.final_result = None
            self.result_value.configure(text='— mm'); self.result_note.configure(text='规划中 · 完成执行后自动显示')
            self.automatic.event('开始规划', dict(snapshot=snapshot, target=frozen, parameters=params))
            self._tell('正在规划整条路径…')
            def worker():
                try:
                    plan = plan_tcp_path(snapshot, tool, targets, free_orientation_segments=free_segments,
                                         align_free_endpoint=False)
                    plan['metadata']['approach_then_align'] = not normal_only
                    self._protect_noncontact_plan(plan, frozen)
                    self.events.put(('plan', generation, dict(plan=plan, kind='complete', gap_m=params['gap'],
                        approach_m=params['approach'], goal=goal.tolist(), reference_goal=reference_goal.tolist(),
                        free_transit=bool(free_segments), targets=[T.tolist() for T in targets])))
                except Exception as exc: self.events.put(('plan_error', generation, str(exc)))
            self.plan_worker = threading.Thread(target=worker, name='标定板完整路径', daemon=True)
            self.plan_worker.start()
        except Exception as exc:
            self._tell(str(exc), True)
            self._end_recording('规划失败：' + str(exc))

    def _check_execution_start(self):
        self._unchanged_board()
        if self.plan is None: raise ValueError('请先规划路径。')
        snapshot = self._snapshot()
        plan = self.plan['plan']
        self._parameters()
        self._protect_noncontact_plan(plan, self.locked_target)
        MotionExecutor.check_board_noncontact(plan, snapshot['T_base_flange'])
        tool = transform(self.locked_target['T_flange_tcp'])
        current = transform(snapshot['T_base_flange']) @ tool
        start = transform(plan['start_snapshot']['T_base_flange']) @ tool
        if (np.max(np.abs(np.asarray(snapshot['joint_angles'])-np.asarray(plan['start_snapshot']['joint_angles']))) > np.deg2rad(.1)
                or np.linalg.norm(current[:3, 3]-start[:3, 3]) > .0002
                or Rotation.from_matrix(start[:3, :3].T@current[:3, :3]).magnitude() > np.deg2rad(.1)):
            self.plan = None
            raise ValueError('机械臂已离开规划起点，请重新规划。')
        return plan


    def _execute(self):
        try:
            if MOTION_EXECUTION_HOLD_REASON: raise ValueError(MOTION_EXECUTION_HOLD_REASON)
            self._idle(); self._ensure_recording()
            self._check_execution_start()
            reader = self.app.robot_reader
            if reader is None: raise ValueError('请先连接右臂。')
            if self.executor and self.executor.reader is reader and self.executor.locked:
                raise ValueError('运动已急停锁定，请处理后重新连接。')
            generation = self.generation
            self.preparing_control = True
            self.automatic.event('用户执行整条路径', self._diagnostic_context())
            self._tell('正在检查控制状态，随后执行路径…')
            def worker():
                try:
                    status = reader.motion_status()
                    if status.get('locked'):
                        raise ValueError('控制器运动锁尚未解除，请处理后重新连接。')
                    if status['ctrl_mode'] == 0 and status.get('teach_status') in (0, 2):
                        reader.prepare_motion_control(operator_confirmed=True)
                    self.events.put(('execution_ready', generation, reader))
                except Exception as exc: self.events.put(('execution_error', generation, str(exc)))
            self.control_worker = threading.Thread(target=worker, name='标定板执行准备', daemon=True)
            self.control_worker.start()
        except Exception as exc:
            self._tell(str(exc), True)
            self._end_recording('执行未开始：' + str(exc))

    def _start_execution(self, reader):
        self.preparing_control = False
        if reader is not self.app.robot_reader or not self.app.robot_connected:
            raise ValueError('机械臂连接变化，执行已取消。')
        plan = self._check_execution_start()
        if self.executor is None or self.executor.reader is not reader:
            self.executor = MotionExecutor(reader, lambda: self.app.live_snapshot, lambda *_: None)
        generation = self.generation
        self.executor.on_event = lambda kind, data: self.events.put(('motion', generation, (kind, data)))
        self.motion_busy = True
        self.last_goal = copy.deepcopy(self.plan['reference_goal'])
        self.last_commanded_goal = copy.deepcopy(self.plan['goal'])
        self.last_motion_state = '正在执行'
        try:
            self.executor.start(plan, speed_percent=1, tolerance_mm=.1, tolerance_deg=.05)
        except Exception:
            self.motion_busy = False
            raise
        self._tell('正在执行，完成后自动显示投影偏差。')


    def stop_motion(self, reason=None):
        robot_kind = getattr(self.app, 'robot_kind', None)
        if robot_kind is not None and robot_kind.get() == 'UR（网络连接）':
            self._tell('本程序对 UR 仅提供只读采集，停止机械臂请使用 UR 示教器或实体急停。', True)
            return
        if self.busy: self.last_motion_state = '已取消或请求停止'
        preparing = self.preparing_control
        self.preparing_control = False; self.control_ready = False
        self.generation += 1; self.planning = self.motion_busy = False
        self.plan = None
        reader = self.app.robot_reader
        if preparing and reader is not None:
            reader.cancel_teaching_recovery()
        manual = not isinstance(reason, str)
        if manual and self.app.teaching_recovery_busy:
            self.app.teaching_recovery_cancelled = True
        result = None
        if reader is not None and (manual or (self.executor and self.executor.running)):
            if self.executor is None or self.executor.reader is not reader:
                self.executor = MotionExecutor(reader, lambda: self.app.live_snapshot, lambda *_: None)
            result = self.executor.stop()
        if result and not result.get('sent'):
            self._tell('停止发送失败：'+str(result.get('error'))+'；请立即使用实体急停。', True)
        elif manual:
            self._tell('已发送电子急停请求；确认可靠支撑后，可点窗口底部“解除急停并恢复示教”。' if reader else '右臂未连接，无法发送停止请求。', True)
        else: self._tell(reason, True)
        self.app.logger.info('标定板任务停止/取消：%s；停止请求 %s', reason or '用户电子急停', result)
        self._end_recording(reason or '用户电子急停')

    def _path_message(self, text):
        self.path_text.configure(text=text)

    def _show_result(self):
        try:
            snapshot = self._snapshot()
            target = self.locked_target
            tcp = transform(snapshot['T_base_flange']) @ transform(target['T_flange_tcp'])
            normal = np.asarray(target['normal_base'])
            delta = tcp[:3, 3] - np.asarray(target['point_base_m'])
            gap = float(delta @ normal)
            projected = delta - normal*gap
            error_mm = float(np.linalg.norm(projected)*1000)
            self.final_result = dict(projection_error_mm=error_mm, gap_mm=gap*1000,
                source='机械臂反馈与标定坐标换算，非独立实测', feedback=snapshot,
                point_base_m=target['point_base_m'], normal_base=target['normal_base'],
                projected_point_base_m=(tcp[:3, 3]-normal*gap).tolist())
            self.result_value.configure(text=f'{error_mm:.3f} mm')
            self.result_note.configure(text='运动已结束 · 板面投影偏差（反馈估算）')
            self.automatic.event('最终投影偏差', self.final_result)
        except Exception as exc:
            self.result_value.configure(text='— mm')
            self.result_note.configure(text='无法计算最终偏差：'+str(exc))
            self.automatic.event('结果计算失败', {'error': str(exc)}, trigger=True)


    def _click(self, event):
        try:
            self._idle()
            if self.finish_reason or self.diagnostic_exporting:
                raise ValueError('上一轮记录正在自动保存，请稍候。')
            if self.render_map is None or self.shown_result is None:
                raise ValueError('请先识别标定板。')
            result = self.shown_result
            if not result['accepted']: raise ValueError('当前标定板尚未通过审查。')
            x, y, scale = self.render_map
            point = image_to_board(((event.x-x)/scale, (event.y-y)/scale), result['K'], result['D'],
                                   result['T_cam_board'], result['geometry'], result['model'])
            self.selected_point = point.tolist()
            self.invalidate('目标已选择，请点击“规划路径”。')
            self.final_result = None
            self.result_value.configure(text='— mm'); self.result_note.configure(text='等待执行 · 反馈估算')
            self._ensure_recording()
            self.automatic.event('画面选点', {'pixel': [(event.x-x)/scale, (event.y-y)/scale], 'point_board_m': point})
            self.tabs.select(0)
        except Exception as exc:
            self._tell(str(exc), True)
            self._end_recording('选点未完成：'+str(exc))


    def _render(self):
        result = self.result
        frame = self.app.camera.get_frame_snapshot()
        use_result = result is not None and 0 <= time.monotonic()-result['monotonic'] <= 1
        self.shown_result = result if use_result else None
        pixels = result['image_rgb'] if use_result else (frame['rgb'] if frame else None)
        self.canvas.delete('all'); self.render_map = None
        if pixels is None:
            self.canvas.create_text(max(50, self.canvas.winfo_width()/2), 90, text='等待相机画面', fill=self.c['muted'])
            return
        h, w = pixels.shape[:2]
        cw, ch = max(10, self.canvas.winfo_width()), max(10, self.canvas.winfo_height())
        scale = min(cw/w, ch/h); x, y = (cw-w*scale)/2, (ch-h*scale)/2
        self.photo = ImageTk.PhotoImage(Image.fromarray(pixels).resize((max(1, int(w*scale)), max(1, int(h*scale)))))
        self.canvas.create_image(x, y, image=self.photo, anchor='nw'); self.render_map = (x, y, scale)
        def pos(uv): return x+uv[0]*scale, y+uv[1]*scale
        def annotation(u, v, text, color):
            item = self.canvas.create_text(u, v, text=text, fill=color, font=('Noto Sans CJK SC', 11, 'bold'))
            box = self.canvas.bbox(item)
            background = self.canvas.create_rectangle(box[0]-3, box[1]-1, box[2]+3, box[3]+1,
                                                      fill=self.c['text'], outline=self.c['text'])
            self.canvas.tag_raise(item, background)
        # 检测到角点与位姿审查通过是两件事；未通过也显示本帧实际检测点。
        if use_result:
            for uv in result['pixels']:
                u, v = pos(uv)
                if result['accepted']:
                    self.canvas.create_oval(u-3, v-3, u+3, v+3, fill=self.c['panel'], outline=self.c['panel'])
                    self.canvas.create_oval(u-2, v-2, u+2, v+2, outline=self.c['green'])
                else:
                    self.canvas.create_oval(u-4, v-4, u+4, v+4, fill=self.c['panel'], outline=self.c['text'])
                    self.canvas.create_oval(u-2, v-2, u+2, v+2, fill=self.c['warning'], outline=self.c['warning'])
        if use_result and result['accepted']:
            try:
                if self.plan is not None and self.locked_target is not None:
                    inverse = np.linalg.inv(transform(self.locked_target['chain']['T_base_board']))
                    steps = self.plan['plan']['steps']
                    samples = steps[::max(1, len(steps)//120)]
                    if samples[-1] is not steps[-1]: samples = samples + [steps[-1]]
                    line = []
                    for step in samples:
                        local = (inverse @ np.r_[np.asarray(step['T_base_tcp'])[:3, 3], 1])[:3]
                        camera_point = transform(result['T_cam_board']) @ np.r_[local, 1]
                        if camera_point[2] <= 0: continue
                        u, v = pos(project_board_point(result, local))
                        if np.isfinite([u, v]).all(): line.extend((u, v))
                    if len(line) >= 4:
                        self.canvas.create_line(*line, fill=self.c['text'], width=5, arrow='last')
                        self.canvas.create_line(*line, fill=self.c['primary'], width=3, arrow='last')
                point = np.asarray(self.locked_target['point_board_m']) if self.locked_target else self._point(result)
                u, v = pos(project_board_point(result, point))
                self.canvas.create_oval(u-7, v-7, u+7, v+7, outline=self.c['text'], width=6)
                self.canvas.create_oval(u-7, v-7, u+7, v+7, outline=self.c['primary'], width=3)
                self.canvas.create_line(u-12, v, u+12, v, fill=self.c['text'], width=5)
                self.canvas.create_line(u, v-12, u, v+12, fill=self.c['text'], width=5)
                self.canvas.create_line(u-12, v, u+12, v, fill=self.c['primary'], width=2)
                self.canvas.create_line(u, v-12, u, v+12, fill=self.c['primary'], width=2)
            except (ValueError, TypeError): pass
        if use_result and result['accepted']:
            legend = '圆点为检测角点，十字圆环为所选目标。'
        elif use_result and len(result['pixels']):
            legend = '带底圆点为已检测角点；审查未通过，暂不能选点或锁定目标。'
        else:
            legend = '等待本帧标定板角点，不显示旧检测标记。'
        if result is not None and not use_result:
            self.quality_label.configure(text='识别结果已过期，等待新的角点检测。', fg=self.c['warning'])
        self.video_note.configure(text=f"{'识别画面' if use_result else '实时画面'} · {w}×{h} · "+legend)

    def _poll(self):
        if self.app._closing: return
        for _ in range(2):
            try: generation, data = self.detect_events.get_nowait()
            except queue.Empty: break
            if generation != self.detect_generation: continue
            if isinstance(data, str):
                self.result = None; self.quality_label.configure(text='识别未完成：'+data, fg=self.c['warning'])
                self.automatic.event('标定板识别异常', {'error': data})
            else:
                self.result = data
                self.automatic.event('标定板定位反馈', {key: data.get(key) for key in
                    ('frame_id', 'monotonic', 'T_cam_board', 'normal_camera_outward', 'accepted',
                     'reason', 'point_count', 'rms_px', 'color_serial')})
                if data['accepted']:
                    self.last_good = data['monotonic']
                    self.quality_label.configure(text='标定板识别通过，可以点选目标。', fg=self.c['green'])
                else:
                    self.quality_label.configure(text=data['reason'], fg=self.c['warning'])
        for _ in range(20):
            try: kind, generation, data = self.events.get_nowait()
            except queue.Empty: break
            if generation != self.generation:
                self.automatic.event('已取消任务的后台结果', {'event': kind, 'generation': generation, 'data': data if kind != 'execution_ready' else None})
                continue
            if kind == 'plan':
                self.planning = False
                reason = self.app.task_blocked('board')
                if reason:
                    self.plan = None; self._tell(reason, True); self._end_recording(reason); continue
                self.plan = data
                text = (f"先到板前 {data['approach_m']*1000:g} mm → 调成垂直 → 停在 {data['gap_m']*1000:g} mm。"
                        if data['plan']['metadata'].get('approach_then_align') else
                        f"保持垂直，沿板法线到板前 {data['gap_m']*1000:g} mm。")
                self._path_message('路径已规划。'+text)
                self.automatic.event('完整路径规划完成', data)
                self.result_note.configure(text='等待执行 · 完成后自动显示')
                self._tell('规划完成，可点击“执行运动”。')
            elif kind in ('plan_error', 'execution_error'):
                self.planning = self.preparing_control = self.motion_busy = False
                self.plan = None
                self.automatic.event('任务失败', {'stage': kind, 'error': data}, trigger=True)
                self._tell(str(data), True); self._end_recording(str(data))
            elif kind == 'execution_ready':
                try: self._start_execution(data)
                except Exception as exc:
                    self.preparing_control = self.motion_busy = False
                    self.plan = None
                    self.automatic.event('执行启动失败', {'error': str(exc)}, trigger=True)
                    self._tell(str(exc), True); self._end_recording(str(exc))
            elif kind == 'motion':
                event, info = data
                if event in ('done', 'paused', 'error', 'stopped'):
                    self.motion_busy = False; self.plan = None
                    self.last_motion_state = {'done': '反馈已到位', 'paused': '已暂停', 'error': '执行失败', 'stopped': '已请求停止'}[event]
                    if event == 'done':
                        self._show_result()
                        self._tell('运动已完成，投影偏差已显示。')
                    else:
                        self.result_value.configure(text='— mm')
                        self.result_note.configure(text='本次未完成：'+str(info.get('message') or event))
                        self._tell('运动结束：'+str(info.get('message') or event), True)
                    self._end_recording(self.last_motion_state)
        if self.motion_busy:
            if not self.app.robot_connected or not self.app.camera.is_live():
                self.stop_motion('机械臂或相机连接中断，已请求停止。')
            elif time.monotonic()-self.last_good > 2:
                self.stop_motion('标定板超过 2 秒未被可靠识别，已请求停止。')
            elif self.result and self.result.get('accepted') and self.app.hole_panel.history.stable(self.result['monotonic']):
                try: self._unchanged_board()
                except ValueError as exc: self.stop_motion(str(exc))
        self._refresh_diagnostics()
        active = (self.busy or bool(self.app.task_blocked('board')) or self.diagnostic_exporting
                  or bool(self.finish_reason) or bool(self.save_error))
        self.robot_button.configure(text=self.app.robot_connection_text())
        self.app._enable(self.preview_button, not active and self.selected_point is not None, primary=True)
        self.app._enable(self.execute_button, not active and self.plan is not None and not MOTION_EXECUTION_HOLD_REASON, primary=True)
        self.app._enable(self.detect_button, not active and not self.detecting, primary=True)
        shared = self.app.hole_panel
        text = 'TCP 尚未载入' if shared.tcp is None else 'TCP 已载入'
        text += '\n手眼：'+('尚未载入' if shared.handeye is None else '已载入，规划时自动核对相机')
        self.calibration_label.configure(text=text)
        self._render()
        self.after(120, self._poll)

    def close(self):
        if self.busy: self.stop_motion('关闭窗口，已请求停止运动。')
        self.detect_stop.set(); self.detect_generation += 1; self.detecting = False
