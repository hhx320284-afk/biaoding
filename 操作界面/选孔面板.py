"""同一相机与右臂连接上的选孔、分阶段定位和探针运动。"""
import json
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk
from scipy.spatial.transform import Rotation

from 目标变换 import (PoseHistory, load_handeye, hole_target, travel_targets,
                    transform, precision_problem)
from 探针运动 import load_tcp, plan_tcp_path, MotionExecutor
from 选孔视觉 import VisionClient, FollowVisionGate, load_holes
from 孔位跟随 import HoleFollower
from 跟随诊断 import FollowDiagnosticsMixin


class HolePanel(FollowDiagnosticsMixin, tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.colors['bg'], padx=18, pady=14)
        self.app, self.c = app, app.colors
        self.events = queue.Queue()
        self.history = PoseHistory()
        self.tcp = self.handeye = self.pose = self.plan = None
        self.executor = self.client = None
        self.follower = None
        self.follow_reference_R = None
        self.follow_vision_gate = None
        self.follow_vision_request = None
        self._init_follow_diagnostics()
        self.vision_stop = threading.Event()
        self.vision_generation = 0
        self.task_generation = 0
        self.vision_running = self.motion_busy = self.planning = False
        self.cycle = None
        self.stage = '待准备'
        self.refine_requested = False
        self.stable_results = self.fine_results = 0
        self.last_base_pose = None
        self.bbox = self.drag_start = self.display_frame = self.render_map = None
        self.select_box = False
        self.holes = load_holes()
        self.hole_id = tk.StringVar()
        self.approach = tk.StringVar(value='30')
        self.depth = tk.StringVar(value='1')
        self.clearance = tk.StringVar(value='50')
        self.speed = tk.StringVar(value='5')
        self.mount_checked = tk.BooleanVar(value=False)
        self._mount_state = False
        self.route_checked = tk.BooleanVar(value=False)
        self._build()
        for variable in (self.approach, self.depth, self.clearance, self.speed):
            variable.trace_add('write', self._settings_changed)
        self.hole_id.trace_add('write', self._hole_changed)
        default = app.project_dir / '结果' / 'TCP标定结果.json'
        if default.exists():
            self._load_tcp(default)
        self.after(100, self._poll)

    @property
    def busy(self):
        return (self.motion_busy or self.planning or self.cycle is not None
                or (self.executor is not None and self.executor.running)
                or (self.follower is not None and self.follower.running))

    def _build(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2, minsize=450)
        self.rowconfigure(1, weight=1)
        head = tk.Frame(self, bg=self.c['bg']); head.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 12))
        self.app._label(head, '选孔与探针运动', 18, bold=True).pack(side='left')
        connection_text = getattr(self.app, 'robot_connection_text', lambda: '连接右臂 can1')()
        self.robot_button = self.app._button(head, connection_text, self.app.toggle_robot, compact=True)
        self.robot_button.pack(side='right')
        self.app._button(head, '打开相机', self.app.camera.start, compact=True).pack(side='right', padx=10)
        outer, left = self.app._card(self, 14); outer.grid(row=1, column=0, sticky='nsew', padx=(0, 14))
        self.app._label(left, '工件画面 · 选孔即自动跟随', 13, bold=True).pack(anchor='w')
        row = tk.Frame(left, bg=self.c['panel']); row.pack(fill='x', pady=10)
        self.box_button = self.app._button(row, '① 框选整个工件', self._begin_box, compact=True); self.box_button.pack(side='left')
        self.locate_button = self.app._button(row, '② 定位 / 跟踪', self._start_vision, primary=True, compact=True); self.locate_button.pack(side='left', padx=8)
        self.app._button(row, '停止定位', self._stop_vision, compact=True).pack(side='left')
        self.canvas = tk.Canvas(left, bg=self.c['selected'], highlightthickness=1, highlightbackground=self.c['muted'], height=360)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<ButtonPress-1>', self._mouse_down)
        self.canvas.bind('<B1-Motion>', self._mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._mouse_up)
        self.video_note = self.app._label(left, '先打开相机；框选完整工件用于 SAM3，定位后点击孔。', 10, 'muted', wraplength=690, justify='left')
        self.video_note.pack(fill='x', pady=(8, 10))
        row = tk.Frame(left, bg=self.c['panel']); row.pack(fill='x')
        self.app._label(row, '目标孔').pack(side='left', padx=(0, 8))
        self.hole_combo = ttk.Combobox(row, textvariable=self.hole_id, values=[h['id'] for h in self.holes], state='readonly', width=20)
        self.hole_combo.pack(side='left')
        self.hole_detail = self.app._label(left, '已载入 origin 工件 CAD 的 47 个孔；孔编号以 CAD 孔表为准。', 10, 'muted', justify='left', wraplength=680)
        self.hole_detail.pack(fill='x', pady=8)
        self.quality_label = self.app._label(left, '尚未定位', 10, 'muted', justify='left', wraplength=680)
        self.quality_label.pack(fill='x')
        outer, right = self.app._card(self, 14); outer.grid(row=1, column=1, sticky='nsew')
        # 设置区可滚动；急停固定在卡片底部，缩小窗口也不会被挤掉。
        self.stop_button = self.app._button(right, '停止运动（电子急停）', self.stop_motion, danger=True)
        self.stop_button.pack(side='bottom', fill='x', pady=(12, 0))
        from 界面主题 import Tk滚动区
        right = Tk滚动区(right)
        self.app._label(right, '自动跟随 · 孔外 5 mm', 15, bold=True).pack(anchor='w')
        self.app._label(right, '选定孔后自动接近；移动工件，针尖持续对准同一个孔。\n保持垂直和孔外间距，不自动插孔。',
                        10, 'text', justify='left', wraplength=470).pack(fill='x', pady=(8, 10))
        self.state_label = self.app._label(right, '准备定位；选孔后自动开始跟随', 12, 'text', bold=True,
                                         justify='left', anchor='w', wraplength=470)
        self.state_label.pack(fill='x')
        self.follow_result_label = self.app._label(right, '跟随偏差：—', 11, 'muted', anchor='w', wraplength=470)
        self.follow_result_label.pack(fill='x', pady=6)
        self.follow_diagnostic_label = self.app._label(right, '跟随全程自动记录，停止后自动保存。', 10,
                                                      'muted', anchor='w', wraplength=470)
        self.follow_diagnostic_label.pack(fill='x', pady=(0, 12))
        self.app._label(right, '标定与运动参数', 13, bold=True).pack(anchor='w')
        row = tk.Frame(right, bg=self.c['panel']); row.pack(fill='x', pady=9)
        self.tcp_button = self.app._button(row, '载入 TCP 结果', self._choose_tcp, compact=True)
        self.tcp_button.pack(side='left', fill='x', expand=True)
        self.handeye_button = self.app._button(row, '载入手眼结果', self._choose_handeye, compact=True)
        self.handeye_button.pack(side='left', fill='x', expand=True, padx=(8, 0))
        self.calibration_label = self.app._label(right, '未载入 TCP', 10, 'muted', justify='left', anchor='w', wraplength=470)
        self.calibration_label.pack(fill='x')
        self.handeye_label = self.app._label(right, '手眼未载入 · 选孔运动需要当前安装的相机→法兰矩阵', 10, 'warning', justify='left', anchor='w', wraplength=470)
        self.handeye_label.pack(fill='x', pady=8)
        params = tk.Frame(right, bg=self.c['panel']); params.pack(fill='x', pady=10)
        for i, (label, variable) in enumerate((('抬升 / 退针 mm', self.clearance), ('跟随速度 1–5%', self.speed))):
            self.app._label(params, label).grid(row=i//2, column=(i%2)*2, sticky='w', pady=5, padx=(0, 6))
            ttk.Entry(params, textvariable=variable, width=7).grid(row=i//2, column=(i%2)*2+1, sticky='w', padx=(0, 12))
        manual = tk.Frame(right, bg=self.c['panel'])
        def toggle_manual():
            opened = bool(manual.winfo_manager())
            if opened:
                manual.pack_forget()
            else:
                manual.pack(fill='both', expand=True, pady=(8, 0))
            manual_button.configure(text='其他运动 ▸' if opened else '其他运动 ▾')
            # Canvas 固定内容高度；展开/收起后按新的请求尺寸刷新滚动范围。
            self.after_idle(lambda: right.master.event_generate('<Configure>'))
        manual_button = self.app._button(right, '其他运动 ▸', toggle_manual, compact=True)
        manual_button.pack(fill='x', pady=(12, 0))
        self.app._label(manual, '以下手动路径在自动跟随停止后使用。',
                        10, 'muted', justify='left', wraplength=470).pack(fill='x')
        self.mount_button = ttk.Checkbutton(manual, text='手动路径：已核对手眼安装', variable=self.mount_checked,
                                            command=self._mount_changed)
        self.mount_button.pack(fill='x')
        params = tk.Frame(manual, bg=self.c['panel']); params.pack(fill='x', pady=8)
        for i, (label, variable) in enumerate((('观察间距 mm', self.approach), ('浅入深度 mm', self.depth))):
            self.app._label(params, label).grid(row=0, column=i*2, sticky='w', pady=5, padx=(0, 6))
            ttk.Entry(params, textvariable=variable, width=7).grid(row=0, column=i*2+1, sticky='w', padx=(0, 12))
        row = tk.Frame(manual, bg=self.c['panel']); row.pack(fill='x', pady=9)
        self.preview_button = self.app._button(row, '预览选孔路径', lambda: self._prepare('hole', False), compact=True); self.preview_button.pack(side='left', fill='x', expand=True)
        self.return_preview = self.app._button(row, '预览回点路径', lambda: self._prepare('pivot', False), compact=True); self.return_preview.pack(side='left', fill='x', expand=True, padx=(8, 0))
        self.path_text = tk.Text(manual, height=6, wrap='word', font=('Noto Sans CJK SC', 10), bg=self.c['bg'], fg=self.c['text'], relief='flat', padx=8, pady=6)
        self.path_text.pack(fill='both', expand=True)
        self._path_message('预览将列出针尖接近路径及可达检查。\n先退针、抬升，再调姿接近；这里只检查运动学，不检测周围障碍物。')
        ttk.Checkbutton(manual, text='已核对预览路径，机械臂和探针沿途无遮挡',
                        variable=self.route_checked).pack(fill='x', pady=(8, 4))
        self.coarse_button = self.app._button(manual, '仅接近观察位', lambda: self._execute('observe')); self.coarse_button.pack(fill='x', pady=3)
        self.insert_button = self.app._button(manual, '接近并自动浅插', lambda: self._execute('insert'), primary=True); self.insert_button.pack(fill='x', pady=3)
        self.return_button = self.app._button(manual, '一键回标定固定点', lambda: self._execute('pivot')); self.return_button.pack(fill='x', pady=3)
        self.note = self.app._label(self, '自动跟随保持孔外 5 mm。定位丢失、工件移动过快或目标不可达时停止；不自动恢复。偏差由视觉与机械臂反馈估算，非独立精度测量。', 10, 'muted', wraplength=1400, justify='left')
        self.note.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        for label in (self.calibration_label, self.handeye_label, self.state_label, self.video_note, self.hole_detail, self.quality_label,
                      self.follow_result_label, self.follow_diagnostic_label):
            label.bind('<Configure>', lambda event, widget=label: widget.configure(wraplength=max(200, event.width - 5)))

    def _tell(self, text, error=False):
        self.state_label.configure(text=text, fg=self.c['error'] if error else self.c['text'])
        self.app._status(text, error)

    def _entry_blocked(self):
        reason = self.app.task_blocked('hole')
        if reason is None and self.busy:
            reason = '选孔与运动任务尚未结束，请先停止或等待完成。'
        if reason is None and self.diagnostic_unsaved:
            reason = '上一轮跟随记录正在保存，请稍候。'
        if reason:
            self._tell(reason, True)
        return reason

    def _notify_board(self, reason):
        panel = getattr(self.app, 'board_panel', None)
        if panel is not None:
            panel.invalidate(reason)

    def _mount_changed(self):
        value = self.mount_checked.get()
        if value == self._mount_state:
            return
        if self._entry_blocked():
            self.mount_checked.set(self._mount_state)
            return
        self._mount_state = value
        self._settings_changed()
        self._notify_board('相机安装核对状态已变更，请重新锁定标定板并预览路径。')

    def _settings_changed(self, *_):
        self.task_generation += 1
        self.plan = None
        self.route_checked.set(False)
        if self.busy:
            self.stop_motion('参数或标定发生变化，已停止任务。')
        hole = next((h for h in self.holes if h['id'] == self.hole_id.get()), None)
        if hole:
            xyz = np.asarray(hole['center_cad_m']) * 1000
            self.hole_detail.configure(text=f"{hole['id']} · 孔径 {2*hole['radius_m']*1000:g} mm · CAD 中心 " + ' / '.join(f'{v:.1f}' for v in xyz) + ' mm')

    def _hole_changed(self, *_):
        was_busy = self.busy
        self._settings_changed()
        # 只有这一次选孔动作可以启动；停止、恢复识别、设备重连均不重新启动。
        if was_busy or not self.hole_id.get():
            return
        generation = self.task_generation
        context = (self.app.robot_reader, self.app.robot_generation, self.vision_generation)
        self.after_idle(lambda: self._start_follow_selected(generation, context))

    def _follow_target(self, pose=None):
        pose = pose or self.pose
        if pose is None or not pose.get('quality', {}).get('accepted'):
            raise ValueError('所选孔尚无合格定位，不能自动跟随。')
        hole = next((item for item in pose.get('projected_holes', [])
                     if item['id'] == self.hole_id.get()), None)
        if hole is None or not hole['visible'] or hole.get('diameter_px', 0) < 5:
            raise ValueError('所选孔不可见或太小，自动跟随已停止。')
        base = self._base_cad(pose, require_mount=False)
        reference = self.follow_reference_R
        if reference is None:
            raise ValueError('自动跟随尚未取得针尖初始朝向。')
        return dict(hole_id=hole['id'], T_base_hole=hole_target(base, hole, reference, 0),
                    T_base_tcp=hole_target(base, hole, reference, .005), monotonic=pose['monotonic'],
                    quality=pose['quality'], valid=True,
                    vision_ready=bool(self.follow_vision_gate and self.follow_vision_gate.ready))

    def _clear_follow_vision(self):
        # 整份替换普通字典供工作线程读取，线程不接触 Tk 变量。
        self.follow_vision_request = None
        self.follow_vision_gate = None

    def _refresh_follow_vision_request(self, target=None):
        if self.cycle != 'follow' or self.follow_vision_gate is None:
            return
        target = target or self.follower.latest_target
        snapshot = self.follower.reader.read_pose()
        if not 0 <= time.monotonic() - snapshot['monotonic'] <= .25:
            raise ValueError('机械臂反馈已过期，无法核对近孔定位方式。')
        tcp = transform(snapshot['T_base_flange']) @ self.tcp['T_flange_tcp']
        distance = np.linalg.norm(tcp[:3, 3] - np.asarray(target['T_base_hole'])[:3, 3])
        request = self.follow_vision_gate.update_distance(distance)
        if request != self.follow_vision_request:
            self._follow_diagnostic_event('自动切换孔位测量', dict(request, distance_mm=distance*1000))
            self.follow_vision_request = request

    def _follow_vision_record(self, pose, target=None):
        frame = pose.get('_frame', {})
        # 图像数组不进入诊断；保留曝光时间、位姿、孔身份和质量以复盘坐标链。
        return dict(operation=pose.get('_op'), monotonic=pose['monotonic'],
                    processed_at=time.monotonic(), T_cam_cad=pose['T_cam_cad'], quality=pose.get('quality'),
                    projected_holes=pose.get('projected_holes'), target=target,
                    processing_seconds=pose.get('processing_seconds'),
                    follow_context={key: pose.get(key) for key in
                                    ('selected_hole_id', 'follow_generation', 'follow_epoch', 'near_rgb')},
                    frame={key: frame[key] for key in ('frame_id', 'monotonic', 'color_serial', 'intrinsics')
                           if key in frame})

    def _start_follow_selected(self, generation, selection_context):
        if generation != self.task_generation or self.app._closing:
            return
        reader, robot_generation, vision_generation = selection_context
        if (reader is not self.app.robot_reader or robot_generation != self.app.robot_generation
                or vision_generation != self.vision_generation):
            return
        if self.app.pages.select() != str(self):
            return
        # 共用 CAN 只能绑定一轮记录；闲置的标定板会话先自动归档，再接管。
        board = getattr(self.app, 'board_panel', None)
        if board is not None and not board.busy and board.diagnostic_unsaved:
            board._end_recording('切换至选孔自动跟随')
            board._refresh_diagnostics()
            self._tell('正在归档上一轮标定板记录，完成后自动开始跟随。' if not board.save_error
                       else '上一轮记录保存失败：'+board.save_error, bool(board.save_error))
            self.after(100, lambda: self._start_follow_selected(generation, selection_context))
            return
        if self._entry_blocked():
            return
        try:
            if not self.vision_running or not self.app.camera.is_live():
                raise ValueError('自动跟随需要实时工件定位和相机画面。')
            problem = precision_problem(self.tcp)
            if problem:
                raise ValueError(problem)
            reader = self.app.robot_reader
            if not self.app.robot_connected or reader is None:
                raise ValueError('自动跟随需要已连接的 PiPER X 右臂 can1。')
            if self.executor is not None and self.executor.reader is reader and self.executor.locked:
                raise ValueError('运动已锁定；处理停止原因并恢复控制后才能开始新任务。')
            snapshot = self._snapshot()
            try:
                clearance, speed = float(self.clearance.get()) / 1000, int(self.speed.get())
            except ValueError:
                raise ValueError('自动跟随的退针/抬升距离和速度须为有效数值。')
            if not np.isfinite(clearance) or not .02 <= clearance <= .15 or not 1 <= speed <= 5:
                raise ValueError('自动跟随：退针/抬升 20–150 mm，速度 1–5%。')
            params = dict(clearance=clearance, speed=speed)
            self.follow_reference_R = (transform(snapshot['T_base_flange']) @ self.tcp['T_flange_tcp'])[:3, :3].copy()
            target = self._follow_target()
            self._begin_follow_recording(dict(mode='自动孔位跟随', gap_mm=5,
                selected_hole=self.hole_id.get(), tcp=self.tcp['payload'], handeye=self.handeye['payload'],
                parameters=params, initial_feedback=snapshot,
                initial_vision=self._follow_vision_record(self.pose, target)))
            self._follow_diagnostic_event('选孔自动跟随请求', target)
            self.task_generation += 1
            generation = self.task_generation
            self.follow_vision_gate = FollowVisionGate(self.hole_id.get(), generation)
            def event(kind, data):
                self._follow_diagnostic_event('跟随事件：'+kind, data, trigger=kind in ('error', 'stopped'))
                self.events.put(('follow', generation, (kind, data)))
            self.follower = HoleFollower(reader, lambda: self.app.live_snapshot, event)
            self.cycle = 'follow'
            self.motion_busy = True
            self.stage = '自动接近并跟随'
            self._refresh_follow_vision_request(target)
            target['vision_ready'] = self.follow_vision_gate.ready
            self.follower.start(self.tcp['T_flange_tcp'], target, gap_m=.005,
                                clearance_m=params['clearance'], speed_percent=params['speed'])
            self._tell('正在自动接近所选孔，之后持续保持孔外 5 mm 跟随。')
            self._path_message('自动跟随中：目标随工件更新，路径在后台逐段检查。\n'
                               '无需重复规划或执行；点击红色停止按钮结束。')
        except Exception as exc:
            if self.cycle == 'follow' and self.follower is not None and self.follower.running:
                self.follower.stop('自动跟随启动未完成：' + str(exc))
            self._clear_follow_vision()
            self.motion_busy = False
            self.cycle = None
            self._end_follow_recording('跟随未开始：'+str(exc))
            self._tell('自动跟随未开始：'+str(exc), True)
            self.app.logger.exception('自动跟随启动未完成')

    def _handle_follow(self, kind, data):
        if kind in ('error', 'stopped'):
            self._clear_follow_vision()
            self.motion_busy = False
            self.cycle = None
            reason = data.get('message', '自动跟随已停止。')
            self._end_follow_recording(reason)
            self._tell(reason, True)
        elif kind in ('started', 'state', 'step') and self.cycle == 'follow':
            self.stage = data.get('stage', self.stage)
            self.state_label.configure(text=f'{self.hole_id.get()} · {self.stage}', fg=self.c['text'])
            error, gap = data.get('projection_error_mm'), data.get('gap_mm')
            if error is not None and gap is not None:
                self.follow_result_label.configure(text=f'对孔偏差 {error:.2f} mm · 孔外间距 {gap:.2f} mm（估算）')

    def _load_tcp(self, path):
        if self._entry_blocked():
            return
        try:
            value = load_tcp(path)
            changed = self.tcp is None or self.tcp['payload'] != value['payload']
            self._settings_changed()
            self.tcp = value
            point = value['pivot_base']*1000
            self.calibration_label.configure(text=(f"TCP RMS {value['rms_mm']:.3f} / 最大 {value['max_mm']:.3f} mm\n固定点 base：" + ' / '.join(f'{v:.2f}' for v in point) + ' mm'), fg=self.c['warning'] if precision_problem(value) else self.c['green'])
            self._tell(precision_problem(value) or 'TCP 已载入。运动前请预览路径。', bool(precision_problem(value)))
            if changed:
                self._notify_board('TCP 标定已变更，请重新锁定标定板并预览路径。')
        except Exception as exc:
            self._tell(f'TCP 未载入：{exc}', True)

    def _choose_tcp(self):
        if self._entry_blocked(): return
        path = filedialog.askopenfilename(parent=self, title='载入 TCP 标定结果', initialdir=self.app.project_dir/'结果', filetypes=[('JSON', '*.json')])
        if path: self._load_tcp(path)

    def _choose_handeye(self):
        if self._entry_blocked(): return
        path = filedialog.askopenfilename(parent=self, title='载入当前安装的手眼标定', initialdir=self.app.project_dir.parent/'手眼标定'/'结果', filetypes=[('JSON', '*.json')])
        if not path or self._entry_blocked(): return
        try:
            value = load_handeye(path)
            changed = (self.handeye is None or self.handeye['payload'] != value['payload']
                       or self.mount_checked.get())
            self._settings_changed()
            self.handeye = value
            self.mount_checked.set(False)
            self._mount_state = False
            self.handeye_label.configure(text=f"手眼已载入 · 相机 {value['serial'] or '文件无设备身份，仅能由你核对当前安装'}\n方向已换算为 相机 → 法兰，单位 m", fg=self.c['green'] if value['serial'] else self.c['warning'])
            self.last_base_pose = None
            if changed:
                self._notify_board('手眼标定或安装核对已变更，请重新核对安装并锁定标定板。')
        except Exception as exc:
            self._tell(f'手眼未载入：{exc}', True)

    def _parameters(self):
        try:
            approach, depth, clearance = [float(v.get()) / 1000 for v in (self.approach, self.depth, self.clearance)]
            speed = int(self.speed.get())
        except ValueError:
            raise ValueError('请填写有效的毫米数值和整数速度百分比。')
        if not all(np.isfinite(v) for v in (approach, depth, clearance)) or not .01 <= approach <= .15 or not 0 < depth <= .005 or not .02 <= clearance <= .15 or not 1 <= speed <= 10:
            raise ValueError('观察间距 10–150 mm，浅入深度 0–5 mm，退针/抬升 20–150 mm，速度 1–10%。')
        return dict(approach=approach, depth=depth, clearance=clearance, speed=speed)

    def _snapshot(self):
        robot_kind = getattr(self.app, 'robot_kind', None)
        if robot_kind is not None and robot_kind.get() == 'UR（网络连接）':
            raise ValueError('UR 当前支持 TCP 标定采集；本页的定位链与路径仍仅支持 PiPER X。')
        value = self.app.live_snapshot
        if value is None or not 0 <= time.monotonic() - value['monotonic'] <= .3:
            raise ValueError('需要右臂 can1 的新鲜反馈，请先连接右臂。')
        return value

    def _frame(self):
        frame = self.app.camera.get_frame_snapshot()
        if frame is None:
            raise ValueError(self.app.camera.frame_snapshot_error or '没有新鲜 RGB-D 画面。')
        return frame

    def _base_cad(self, pose=None, *, require_mount=True):
        pose = pose or self.pose
        if pose is None: raise ValueError('请先定位工件。')
        if time.monotonic() - pose['monotonic'] > 2:
            raise ValueError('定位结果已过期，等待新的跟踪画面。')
        if not self.handeye or (require_mount and not self.mount_checked.get()):
            raise ValueError('请载入当前右臂安装的手眼结果，并核对安装对应关系。')
        serial = pose['_frame']['color_serial']
        if not require_mount and not self.handeye['serial']:
            raise ValueError('手眼结果缺少相机序列号，不能自动核对跟随坐标链。')
        if self.handeye['serial'] and serial != self.handeye['serial']:
            raise ValueError(f'手眼相机 {self.handeye["serial"]} 与当前相机 {serial} 不同。')
        return self.history.at(pose['monotonic']) @ self.handeye['T_flange_camera'] @ transform(pose['T_cam_cad'])

    def _selected_hole(self):
        if self.pose is None: raise ValueError('请先完成工件定位。')
        if not self.pose.get('quality',{}).get('accepted'):
            raise ValueError('当前定位未通过图像 / 深度检查，请等待有效定位结果。')
        hole = next((h for h in self.pose['projected_holes'] if h['id'] == self.hole_id.get()), None)
        if not hole or not hole['visible']:
            raise ValueError('请在定位画面中选择一个完整可见的孔。')
        return hole

    def _begin_box(self):
        if self._entry_blocked(): return
        self._stop_vision()
        self.select_box = True
        self.bbox = None
        self._tell('在画面中拖出包围整个工件的矩形，然后点“定位 / 跟踪”。')

    def _pixel(self, event):
        if self.render_map is None: return None
        x, y, scale, width, height = self.render_map
        px, py = (event.x-x)/scale, (event.y-y)/scale
        if not 0 <= px < width or not 0 <= py < height: return None
        return [px, py]

    def _mouse_down(self, event):
        point = self._pixel(event)
        if point is None or self._entry_blocked(): return
        if self.select_box:
            self.drag_start = point
        elif self.pose is not None:
            visible = [h for h in self.pose['projected_holes'] if h['visible']]
            if visible:
                hole = min(visible, key=lambda h: np.linalg.norm(np.asarray(h['uv'])-point))
                if np.linalg.norm(np.asarray(hole['uv'])-point) <= max(12, hole['diameter_px']*.75):
                    self.hole_id.set(hole['id'])

    def _mouse_drag(self, event):
        if self.busy or self.app.task_blocked('hole'):
            self.drag_start = None
            return
        point = self._pixel(event)
        if self.select_box and self.drag_start and point:
            a, b = self.drag_start, point
            self.bbox = [min(a[0],b[0]), min(a[1],b[1]), max(a[0],b[0]), max(a[1],b[1])]

    def _mouse_up(self, event):
        if self.busy or self.app.task_blocked('hole'):
            self.drag_start = None
            return
        self._mouse_drag(event)
        if self.bbox:
            self.select_box = False
            self.drag_start = None
            self._tell('整件框已选定。点击“定位 / 跟踪”运行 SAM3 和 FoundationPose。')

    def _start_vision(self):
        if self.vision_running or self._entry_blocked(): return
        try:
            frame = self._frame()
            bbox = list(self.bbox or [])
            if len(bbox) != 4 or min(bbox[2]-bbox[0], bbox[3]-bbox[1]) < 20:
                raise ValueError('请先框选整个工件，框内应包含完整外轮廓。')
        except Exception as exc:
            self._tell(str(exc), True); return
        self.pose = None
        self.plan = None
        self.stable_results = self.fine_results = 0
        self.last_base_pose = None
        self.vision_generation += 1
        generation = self.vision_generation
        stop = self.vision_stop = threading.Event()
        self.vision_running = True
        self.app.logger.info('用户启动整件 SAM3 / FoundationPose 定位')
        self._tell('正在启动 SAM3 / FoundationPose；首次加载需要等待。')
        def worker():
            client = VisionClient(logger=self.app.logger)
            if stop.is_set(): return
            self.client = client
            try:
                client.start()
                if stop.is_set(): return
                result = client.request('init', frame, bbox=bbox)
                self.events.put(('vision', generation, dict(result, _frame=frame, _op='init')))
                last_id = frame['frame_id']
                while not stop.is_set():
                    current = self.app.camera.get_frame_snapshot()
                    if current is None:
                        raise ValueError('RGB-D 画面中断，定位已停止。')
                    if current['frame_id'] == last_id:
                        stop.wait(.03); continue
                    context = self.follow_vision_request
                    op = 'follow' if context is not None else 'refine' if self.refine_requested else 'track'
                    last_id = current['frame_id']
                    result = client.request(op, current, **(context or {}))
                    if not stop.is_set():
                        self.events.put(('vision', generation, dict(result, _frame=current, _op=op)))
            except Exception as exc:
                if not stop.is_set(): self.events.put(('vision_error', generation, str(exc)))
            finally:
                client.close()
                self.events.put(('vision_end', generation, None))
        threading.Thread(target=worker, name='工件定位', daemon=True).start()

    def _stop_vision(self):
        if self.cycle is not None: self.stop_motion('定位已停止，运动任务取消。')
        self.vision_stop.set()
        self.vision_generation += 1
        self.vision_running = False
        self.pose = None
        self.refine_requested = False
        self._clear_follow_vision()
        self.plan = None
        self.task_generation += 1
        self.planning = False
        client, self.client = self.client, None
        if client: threading.Thread(target=client.close, daemon=True).start()

    def _path_message(self, text):
        self.path_text.configure(state='normal'); self.path_text.delete('1.0', 'end')
        self.path_text.insert('1.0', text); self.path_text.configure(state='disabled')

    def _prepare(self, kind, execute, mode=None):
        if self._entry_blocked(): return
        try:
            if self.tcp is None: raise ValueError('请先载入 TCP 结果。')
            snapshot, params = self._snapshot(), self._parameters()
            current = transform(snapshot['T_base_flange']) @ self.tcp['T_flange_tcp']
            if kind == 'pivot':
                target = self.tcp['reference_T_base_tcp']
                base_cad = None
            else:
                hole = self._selected_hole()
                base_cad = self._base_cad()
                target = hole_target(base_cad, hole, current[:3,:3], params['approach'])
            targets = travel_targets(current, target, params['clearance'])
            if execute:
                old_targets = self.plan['targets']
                for old, new in zip(old_targets, targets):
                    delta = np.linalg.inv(transform(old)) @ new
                    if np.linalg.norm(delta[:3,3]) > .002 or np.rad2deg(Rotation.from_matrix(delta[:3,:3]).magnitude()) > 1:
                        self.plan = None
                        self.route_checked.set(False)
                        raise ValueError('当前起点或目标已偏离预览路径，请重新预览并核对。')
            self.planning = True
            self.task_generation += 1
            generation = self.task_generation
            self._tell('正在计算完整针尖路径、逆解和关节连续性…')
            T_tool = self.tcp['T_flange_tcp'].copy()
            def worker():
                try:
                    plan = plan_tcp_path(snapshot, T_tool, targets)
                    self.events.put(('plan', generation, dict(plan=plan, kind=kind, execute=execute, mode=mode, params=params, base_cad=base_cad, targets=[t.tolist() for t in targets])))
                except Exception as exc: self.events.put(('plan_error', generation, str(exc)))
            threading.Thread(target=worker, name='针尖路径检查', daemon=True).start()
        except Exception as exc:
            self._tell(str(exc), True)

    def _execute(self, mode):
        if self._entry_blocked(): return
        try:
            if self.plan is None or self.plan['kind'] != ('pivot' if mode == 'pivot' else 'hole'):
                raise ValueError('请先预览对应路径。')
            if not self.route_checked.get(): raise ValueError('请先核对预览路径及现场障碍物，再勾选路径确认。')
            if mode != 'observe' and precision_problem(self.tcp): raise ValueError(precision_problem(self.tcp))
            if self.executor is not None and self.executor.reader is self.app.robot_reader and self.executor.locked:
                raise ValueError('自动运动已锁定；恢复手动拖动请点窗口底部“解除急停并恢复示教”。')
            if mode != 'pivot' and not self.vision_running: raise ValueError('请先启动实时跟踪。')
            if not self.app.camera.is_live(): raise ValueError('需要实时相机画面观察运动。')
            self._prepare('pivot' if mode == 'pivot' else 'hole', True, mode)
        except Exception as exc: self._tell(str(exc), True)

    def _start_plan(self, plan, mode, params):
        blocked = self.app.task_blocked('hole')
        if blocked:
            raise ValueError(blocked)
        reader = getattr(self.app, 'robot_reader', None)
        if reader is None: raise ValueError('右臂连接不可用。')
        fresh = self._snapshot()
        if mode != 'pivot':
            self._base_cad()
        if mode in ('pivot', 'insert', 'fine_insert') and precision_problem(self.tcp):
            raise ValueError(precision_problem(self.tcp))
        if not self.app.camera.is_live():
            raise ValueError('相机未就绪，不能执行运动。')
        start = plan['start_snapshot']
        if np.max(np.abs(np.asarray(fresh['joint_angles']) - np.asarray(start['joint_angles']))) > np.deg2rad(.5):
            raise ValueError('规划期间机械臂已移动，请重新预览路径。')
        if self.executor is None or self.executor.reader is not reader:
            self.executor = MotionExecutor(reader, lambda: self.app.live_snapshot, lambda kind,data: None)
        generation = self.task_generation
        self.executor.on_event = lambda kind,data: self.events.put(('motion',generation,(kind,data)))
        self.cycle = mode
        if mode != 'pivot':
            self.motion_anchor = self._base_cad()
        self.active_params = params
        self.motion_busy = True
        self.stage = '回标定固定点' if mode == 'pivot' else '修正并浅插' if mode == 'fine_insert' else '粗定位接近'
        try:
            self.executor.start(plan, speed_percent=params['speed'], tolerance_mm=.1 if mode in ('pivot','fine_insert') else .5,
                                tolerance_deg=.05 if mode in ('pivot','fine_insert') else .2)
        except Exception:
            self.motion_busy = False; self.cycle = None
            raise
        self.app.logger.info('用户执行右臂任务 %s，路径 %d 小段，速度 %d%%', mode, len(plan['steps']), params['speed'])
        self._tell(self.stage)

    def stop_motion(self, reason=None):
        robot_kind = getattr(self.app, 'robot_kind', None)
        if robot_kind is not None and robot_kind.get() == 'UR（网络连接）':
            self._tell('本程序对 UR 仅提供只读采集，停止机械臂请使用 UR 示教器或实体急停。', True)
            return
        was_active = self.busy
        self.task_generation += 1
        self.cycle = None
        self.motion_busy = False
        self.planning = False
        self.refine_requested = False
        self._clear_follow_vision()
        self.plan = None
        result = None
        # 显式停止按钮始终发送；内部取消只有实际执行器正在运动才发急停。
        manual = not isinstance(reason, str)
        if manual and self.app.teaching_recovery_busy:
            self.app.teaching_recovery_cancelled = True
        reader = getattr(self.app, 'robot_reader', None)
        if self.follower is not None and self.follower.running:
            result = self.follower.stop(reason if isinstance(reason, str) else '用户停止自动跟随')
        elif reader is not None and (manual or (self.executor and self.executor.running)):
            if self.executor is None or self.executor.reader is not reader:
                self.executor = MotionExecutor(reader, lambda:self.app.live_snapshot, lambda *_:None)
            result = self.executor.stop()
        if result and not result.get('sent'):
            self._tell('停止命令发送失败：'+str(result.get('error'))+'。请立即使用实体急停。',True)
        elif manual:
            self._tell('已发送电子急停请求；确认可靠支撑后，可点窗口底部“解除急停并恢复示教”。' if reader else '右臂未连接，无法发送停止命令；请使用实体急停。',True)
        else:
            self._tell(reason,True)
        self.app.logger.info('右臂任务停止/取消：%s；停止请求 %s', reason or '用户电子急停', result)
        self._end_follow_recording(reason if isinstance(reason, str) else '用户电子急停')

    def _observation_ready(self, pose):
        visible = [h for h in pose.get('projected_holes', []) if h['visible'] and h['diameter_px'] >= 5]
        selected = next((h for h in visible if h['id'] == self.hole_id.get()), None)
        return len(visible) >= 8 and selected is not None and pose.get('quality', {}).get('view_cosine', 0) >= .6 and self.stable_results >= 3

    def _handle_pose(self, pose):
        if self.cycle == 'follow':
            gate = self.follow_vision_gate
            if pose.get('_op') != 'follow' or gate is None or not gate.matches(pose):
                return
        elif pose.get('_op') == 'follow':
            return  # 停止或换任务后的在途结果不回写界面和手动流程。
        quality = pose.get('quality', {})
        if self.cycle == 'follow':
            try:
                if not self.follow_vision_gate.observe(pose):
                    return
                self.pose = pose
                quality['follow_confirmed_frames'] = self.follow_vision_gate.confirmed
                quality['follow_near_rgb'] = self.follow_vision_gate.near
                target = self._follow_target(pose)
                self._follow_diagnostic_event('跟随视觉更新', self._follow_vision_record(pose, target))
                if not self.follower.update(target):
                    return
                label = ('近孔实测 · 已核对所选孔' if self.follow_vision_gate.ready else
                         f'近孔实测 · 正在确认 {self.follow_vision_gate.confirmed}/3') if self.follow_vision_gate.near else '整件跟踪 · 靠近后自动核对孔位'
                self.quality_label.configure(text=label)
                self._refresh_follow_vision_request(target)
            except Exception as exc:
                self._follow_diagnostic_event('跟随视觉失效', self._follow_vision_record(pose), trigger=True)
                self.stop_motion(str(exc))
            return
        self.pose = pose
        self.quality_label.configure(text=f"{pose['_op']} · {quality.get('reason', '')}\n实测匹配 {quality.get('matches', 0)} · 像素残差 {quality.get('residual_px', '—')}")
        if pose['_op'] in ('init','track') and not quality.get('accepted'):
            self.stable_results = 0
            if self.cycle not in (None,'pivot'):
                self.stop_motion('工件跟踪未通过粗深度一致性检查，已停止任务。')
            return
        try:
            base = self._base_cad(pose)
            if self.last_base_pose is not None:
                delta = np.linalg.inv(self.last_base_pose) @ base
                dp = np.linalg.norm(delta[:3,3])
                dr = np.rad2deg(Rotation.from_matrix(delta[:3,:3]).magnitude())
                self.stable_results = self.stable_results+1 if dp < .003 and dr < 2 else 0
                if self.motion_busy and self.cycle != 'pivot' and (dp > .01 or dr > 5):
                    self.stop_motion('跟踪位姿突变，已停止运动，请重新核对工件。'); return
            if self.motion_busy and self.cycle != 'pivot':
                drift = np.linalg.inv(self.motion_anchor) @ base
                limit_m, limit_deg = (.002, 1) if self.cycle == 'fine_insert' else (.01, 5)
                if np.linalg.norm(drift[:3,3]) > limit_m or np.rad2deg(Rotation.from_matrix(drift[:3,:3]).magnitude()) > limit_deg:
                    self.stop_motion('工件累计偏离本次运动目标，已停止，请重新定位和预览。'); return
            self.last_base_pose = base
        except ValueError:
            self.stable_results = 0
            if self.cycle not in (None, 'pivot'): self.stop_motion('图像与右臂坐标无法同步，已停止运动。')
            return
        if self.cycle == 'insert' and self.motion_busy and self._observation_ready(pose):
            current = transform(self._snapshot()['T_base_flange']) @ self.tcp['T_flange_tcp']
            target = hole_target(base, self._selected_hole(), current[:3,:3], 0)
            if np.linalg.norm(current[:3,3]-target[:3,3]) <= self.active_params['approach'] + .005:
                self.executor.pause_after_step()
        if self.stage == '停稳精定位' and self.cycle == 'insert' and pose['_op'] == 'refine' and pose['monotonic'] >= self.fine_started+.7:
            selected = self.hole_id.get()
            observed = pose.get('observed_hole_ids', quality.get('observed_hole_ids', []))
            accepted = bool(quality.get('accepted')) and quality.get('matches',0) >= 8
            residual = quality.get('residual_px')
            accepted = accepted and residual is not None and np.isfinite(residual) and residual <= 1.0 and selected in observed
            if accepted and self._robot_stable():
                if self.fine_results:
                    previous = self.fine_base
                    delta = np.linalg.inv(previous) @ base
                    if np.linalg.norm(delta[:3,3]) > .0005 or np.rad2deg(Rotation.from_matrix(delta[:3,:3]).magnitude()) > .2:
                        self.fine_results = 0
                self.fine_base = base
                self.fine_results += 1
                if self.fine_results >= 2: self._fine_plan(base)
            else:
                self.fine_results = 0
                self._tell('停稳精定位：等待两帧合格结果；须含选中孔、≥8 实测孔、残差≤1 px。')

    def _robot_stable(self):
        try:
            now = self._snapshot()
            return self.history.stable(now['monotonic'])
        except ValueError: return False

    def _fine_plan(self, base):
        if self.planning: return
        try:
            if precision_problem(self.tcp): raise ValueError(precision_problem(self.tcp))
            snap, hole = self._snapshot(), self._selected_hole()
            delta = np.linalg.inv(self.motion_anchor) @ base
            if np.linalg.norm(delta[:3,3]) > .005 or np.rad2deg(Rotation.from_matrix(delta[:3,:3]).magnitude()) > 2:
                raise ValueError('RGB 修正超过 5 mm / 2°，已停在观察位。请重新定位、预览后再接近。')
            current = transform(snap['T_base_flange']) @ self.tcp['T_flange_tcp']
            targets = [hole_target(base,hole,current[:3,:3],.005), hole_target(base,hole,current[:3,:3],-self.active_params['depth'])]
            self.planning = True
            generation = self.task_generation
            self.refine_requested = False
            self.stage = '精定位路径检查'
            params = dict(self.active_params, speed=min(2,self.active_params['speed']))
            def worker():
                try:
                    plan = plan_tcp_path(snap,self.tcp['T_flange_tcp'],targets)
                    self.events.put(('fine_plan',generation,dict(plan=plan,params=params,base=base)))
                except Exception as exc: self.events.put(('plan_error',generation,str(exc)))
            threading.Thread(target=worker,daemon=True).start()
        except Exception as exc: self.stop_motion(str(exc))

    def _handle_motion(self, kind, data):
        if kind in ('error','stopped'):
            self.motion_busy = False; self.cycle = None; self.refine_requested = False
            self._tell(data.get('message',str(data)),True)
        elif kind == 'step':
            self._tell(f"{self.stage} · {data.get('index',0)} / {data.get('count',0)} 段")
        elif kind in ('done','paused'):
            self.motion_busy = False
            if self.cycle == 'insert':
                self.stage = '停稳精定位'
                self.fine_results = 0
                self.fine_started = time.monotonic()
                self.fine_deadline = time.monotonic()+30
                self.refine_requested = False
                self._tell('已结束当前运动段，停稳后进行 RGB 孔特征精定位。')
                self.app.logger.info('接近阶段结束，等待停稳后 RGB 精定位')
            else:
                mode = self.cycle
                self.cycle = None
                self._tell('已到达浅插目标，按几何位置停止。' if mode == 'fine_insert' else '已到达标定固定点。' if mode == 'pivot' else '已到达观察位。可检查画面后再执行精定位插孔。')

    def _render(self):
        frame = self.pose.get('_frame') if self.pose else self.app.camera.get_frame_snapshot()
        if frame is None:
            self.canvas.delete('all'); self.canvas.create_text(max(100,self.canvas.winfo_width()/2),150,
                text='等待实时 RGB-D 画面\n'+str(self.app.camera.frame_snapshot_error or ''),
                width=max(240,self.canvas.winfo_width()-40),fill=self.c['muted'],font=('Noto Sans CJK SC',12)); return
        self.display_frame = frame
        rgb = frame['rgb']; h,w = rgb.shape[:2]
        cw,ch = max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        scale = min(cw/w,ch/h); iw,ih = max(1,int(w*scale)),max(1,int(h*scale)); x,y=(cw-iw)/2,(ch-ih)/2
        self.render_map = (x,y,scale,w,h)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((iw,ih)))
        self.canvas.delete('all'); self.canvas.create_image(x,y,image=self.photo,anchor='nw')
        if self.bbox and (self.select_box or self.pose is None):
            a,b,c,d=self.bbox; self.canvas.create_rectangle(x+a*scale,y+b*scale,x+c*scale,y+d*scale,outline=self.c['green'],width=2)
        if self.pose:
            for hole in self.pose.get('projected_holes',[]):
                if not hole['visible']: continue
                u,v=hole['uv']; px,py=x+u*scale,y+v*scale
                chosen=hole['id']==self.hole_id.get(); radius=max(4,hole['diameter_px']*.5*scale)
                self.canvas.create_oval(px-radius,py-radius,px+radius,py+radius,
                    outline=self.c['text'] if chosen else self.c['panel'],width=5 if chosen else 3)
                self.canvas.create_oval(px-radius,py-radius,px+radius,py+radius,
                    outline=self.c['primary'] if chosen else self.c['green'],width=3 if chosen else 1)
                if chosen:
                    for color, width in ((self.c['text'], 4), (self.c['primary'], 2)):
                        self.canvas.create_line(px-7,py,px+7,py,fill=color,width=width)
                        self.canvas.create_line(px,py-7,px,py+7,fill=color,width=width)
                    label = self.canvas.create_text(px+12,py-14,text=hole['id'],fill=self.c['text'],
                        anchor='w',font=('Noto Sans CJK SC',11,'bold'))
                    bounds = self.canvas.bbox(label)
                    backdrop = self.canvas.create_rectangle(bounds[0]-3,bounds[1]-1,bounds[2]+3,bounds[3]+1,
                        fill=self.c['selected'],outline=self.c['input'])
                    self.canvas.tag_lower(backdrop, label)
        age=time.monotonic()-frame['monotonic']
        self.video_note.configure(text=f"{'定位使用的画面' if self.pose else '实时 RGB-D'} · {w}×{h} · {age:.2f} 秒前 · 相机 {frame['color_serial']}\n"+('带十字的圆环为目标孔，绿色圆环为 CAD 投影；精定位还会测量真实 RGB 孔轮廓。' if self.pose else '框选整个工件后定位，再点孔选择。'))

    def _poll(self):
        if self.app._closing: return
        for _ in range(20):
            try: kind,generation,data=self.events.get_nowait()
            except queue.Empty: break
            if kind.startswith('vision') and generation != self.vision_generation: continue
            if kind in ('plan','plan_error','fine_plan','motion','follow') and generation != self.task_generation: continue
            try:
                if kind == 'vision': self._handle_pose(data)
                elif kind == 'vision_error':
                    self.pose=None; self.plan=None
                    self._tell('定位失败：'+data,True)
                    if self.cycle: self.stop_motion('定位失败：'+data)
                elif kind == 'vision_end': self.vision_running=False
                elif kind == 'plan_error':
                    self.planning=False; self.cycle=None; self._tell('路径不可执行：'+data,True)
                elif kind == 'plan':
                    self.planning=False; self.plan=data
                    text='回标定固定点' if data['kind']=='pivot' else f"目标孔 {self.hole_id.get()} · 观察位"
                    text+='\n针尖 base XYZ（mm）：\n'
                    text+='\n'.join(f"{i+1}. "+' / '.join(f'{v:.1f}' for v in np.asarray(T)[:3,3]*1000) for i,T in enumerate(data['targets']))
                    text+=f"\n{len(data['plan']['steps'])} 段可达检查通过；不含障碍物检查。"
                    self._path_message(text)
                    if data['execute']:
                        if data['base_cad'] is not None:
                            delta=np.linalg.inv(data['base_cad'])@self._base_cad()
                            if np.linalg.norm(delta[:3,3])>.002 or np.rad2deg(Rotation.from_matrix(delta[:3,:3]).magnitude())>1:
                                self.plan=None; self.route_checked.set(False)
                                raise ValueError('规划期间工件目标发生变化，请重新预览路径。')
                        self._start_plan(data['plan'],data['mode'],data['params'])
                    else:
                        self.route_checked.set(False); self._tell('路径已预览，请核对现场后选择执行。')
                elif kind == 'fine_plan':
                    self.planning=False
                    if self.cycle=='insert':
                        if not self.pose.get('quality',{}).get('accepted'):
                            raise ValueError('最新画面未通过定位检查，请重新精定位。')
                        delta=np.linalg.inv(data['base'])@self._base_cad()
                        if np.linalg.norm(delta[:3,3])>.0005 or np.rad2deg(Rotation.from_matrix(delta[:3,:3]).magnitude())>.2:
                            raise ValueError('精定位规划期间目标发生变化，请重新精定位。')
                        self._start_plan(data['plan'],'fine_insert',data['params'])
                elif kind == 'motion': self._handle_motion(*data)
                elif kind == 'follow': self._handle_follow(*data)
            except Exception as exc:
                self.planning=False
                if self.busy: self.stop_motion(str(exc))
                else: self._tell(str(exc),True)
                self.app.logger.exception('选孔流程未完成')
        if self.cycle:
            if not self.app.robot_connected or not self.app.camera.is_live():
                self.stop_motion('右臂或相机反馈中断，任务已停止。')
            elif self.cycle == 'follow':
                if self.follower.reader is not self.app.robot_reader:
                    self.stop_motion('机械臂连接已变化，自动跟随已停止。')
                elif not self.vision_running:
                    self.stop_motion('工件定位已结束，自动跟随已停止。')
                else:
                    self.follower.update_heartbeat()
                    try:
                        self._refresh_follow_vision_request()
                    except Exception as exc:
                        self.stop_motion(str(exc))
            elif self.cycle!='pivot' and (self.pose is None or time.monotonic()-self.pose['monotonic']>2):
                self.stop_motion('工件定位超过 2 秒未更新，任务已停止。')
            elif self.stage=='停稳精定位' and time.monotonic()>self.fine_deadline:
                self.cycle=None; self.refine_requested=False
                self._tell('30 秒内未通过精定位复核，已停留观察位。请调整视角、光照或检查孔对应。',True)
            elif self.stage=='停稳精定位' and time.monotonic()-self.fine_started>=.7 and self._robot_stable():
                self.refine_requested=True
        self._poll_follow_diagnostics()
        self.follow_diagnostic_label.configure(text=self._follow_diagnostic_text(),
                                              fg=self.c['error'] if self.follow_save_error else self.c['muted'])
        active=self.busy or self.diagnostic_unsaved or self.app.task_blocked('hole') is not None
        self.robot_button.configure(text=getattr(self.app, 'robot_connection_text',
            lambda: '取消连接' if self.app.robot_connecting else 'can1 已连接 · 断开' if self.app.robot_connected else '连接右臂 can1')())
        for button in (self.box_button,self.preview_button,self.return_preview,self.coarse_button,
                       self.tcp_button,self.handeye_button): self.app._enable(button,not active)
        self.mount_button.configure(state='disabled' if active else 'normal')
        precise = not active and precision_problem(self.tcp) is None
        self.app._enable(self.return_button,precise)
        self.app._enable(self.locate_button,not active and not self.vision_running,primary=True)
        self.app._enable(self.insert_button,precise,primary=True)
        self.hole_combo.configure(state='disabled' if active else 'readonly')
        self._render()
        self.after(100,self._poll)

    def close(self):
        if self.busy: self.stop_motion('窗口关闭，已请求停止运动。')
        self._stop_vision()

    def prepare_close(self):
        self.task_generation += 1  # 关闭等待期间，撤销尚未开始的选孔回调。
        return super().prepare_close()
