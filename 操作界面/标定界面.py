"""右臂 TCP 触点确认采集、相机观察与批量数据筛选界面。"""

import copy
import json
import logging
import queue
import re
import subprocess
import threading
import tkinter as tk
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from zoneinfo import ZoneInfo

from 四点标定 import DEFAULT_CRITERIA, validate_criteria
from 批量标定 import review_dataset
from 采点引导 import DEFAULT_STABILITY
from 自动采集面板 import AutoCaptureMixin, FIRMWARES, ROBOT_KINDS
from 相机画面 import CameraPanel
from 选孔面板 import HolePanel
from 标定板面板 import BoardPanel
from 界面主题 import TK_COLORS, Tk卡片, Tk按钮, Tk控件样式, Tk滚动区

COLORS = TK_COLORS
FONT = 'Noto Sans CJK SC'


class CalibrationApp(AutoCaptureMixin, tk.Tk):
    def __init__(self, project_dir, logger):
        super().__init__()
        self.project_dir, self.logger = Path(project_dir), logger
        self.colors = COLORS
        self.title('TCP标定与视觉验证 · PiPER X / UR')
        self.configure(bg=COLORS['bg'])
        self.tk.call('tk', 'scaling', 1.333333)
        self.option_add('*Font', (FONT, 10))
        self.geometry(f'1580x{min(1110, self.winfo_screenheight() - 100)}')
        self.minsize(1440, 970)
        self.records = []
        self.next_record_id = 1
        self.audit = self.result = None
        self.audit_busy = False
        self.audit_generation = 0
        self.review_events = queue.Queue()
        self.teaching_recovery_busy = False
        self.teaching_recovery_cancelled = False
        self.teaching_recovery_events = queue.Queue()
        self.criteria_after = None
        self.suppress_changes = False
        # 本机 can1 已通过固件查询确认 S-V1.9-0；仍可在连接前手动选择。
        self.robot_firmware = tk.StringVar(value=next(name for name, value in FIRMWARES.items() if value == 'v189'))
        self.robot_kind = tk.StringVar(value=ROBOT_KINDS[0])
        self.ur_host = tk.StringVar()
        self.ur_model = tk.StringVar()
        self._selected_robot_kind = self.robot_kind.get()
        self.criteria_vars = {key: tk.StringVar(value=f'{value:g}') for key, value in DEFAULT_CRITERIA.items()}
        self.filter_value = tk.StringVar(value='全部数据')
        self._init_capture()
        self._styles()
        self._build()
        self._robot_fields()
        self.robot_kind.trace_add('write', self._robot_selection_changed)
        for variable in self.criteria_vars.values():
            variable.trace_add('write', self._criteria_changed)
        self.protocol('WM_DELETE_WINDOW', self._close)
        self.bind('<Control-s>', self._save_current)
        self._controls()
        self._refresh_shell()
        self.after(80, self._poll)

    def _styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', font=(FONT, 10), foreground=COLORS['text'], background=COLORS['panel'])
        for name in ('TEntry', 'TCombobox', 'TSpinbox'):
            style.configure(name, padding=(8, 5), fieldbackground=COLORS['panel'],
                            background=COLORS['panel'], foreground=COLORS['text'],
                            insertcolor=COLORS['text'], arrowcolor=COLORS['text'],
                            selectbackground=COLORS['selected'], selectforeground=COLORS['text'],
                            bordercolor=COLORS['input'], lightcolor=COLORS['panel'], darkcolor=COLORS['panel'])
            style.map(name, fieldbackground=[('disabled', COLORS['bg']), ('readonly', COLORS['panel'])],
                      foreground=[('disabled', COLORS['muted'])],
                      bordercolor=[('focus', COLORS['text']), ('!focus', COLORS['input'])],
                      arrowcolor=[('disabled', COLORS['muted'])],
                      selectbackground=[('readonly', COLORS['selected'])],
                      selectforeground=[('readonly', COLORS['text'])])
        self.option_add('*TCombobox*Listbox.background', COLORS['panel'])
        self.option_add('*TCombobox*Listbox.foreground', COLORS['text'])
        self.option_add('*TCombobox*Listbox.selectBackground', COLORS['selected'])
        self.option_add('*TCombobox*Listbox.selectForeground', COLORS['text'])
        style.configure('Treeview', font=(FONT, 10), rowheight=34, background=COLORS['panel'],
                        fieldbackground=COLORS['panel'], foreground=COLORS['text'],
                        bordercolor=COLORS['input'], lightcolor=COLORS['panel'], darkcolor=COLORS['panel'])
        style.configure('Treeview.Heading', font=(FONT, 10, 'bold'), background=COLORS['bg'],
                        foreground=COLORS['text'], bordercolor=COLORS['input'], padding=(8, 8))
        style.map('Treeview', background=[('selected', COLORS['selected'])],
                  foreground=[('selected', COLORS['text'])], bordercolor=[('focus', COLORS['text'])])
        style.map('Treeview.Heading', background=[('active', COLORS['selected'])])
        style.configure('TNotebook', background=COLORS['bg'], bordercolor=COLORS['border'], tabmargins=(8, 8, 8, 0))
        style.configure('TNotebook.Tab', font=(FONT, 10, 'bold'), background=COLORS['bg'],
                        foreground=COLORS['muted'], padding=(16, 9), focuscolor=COLORS['text'],
                        bordercolor=COLORS['muted'], lightcolor=COLORS['panel'], darkcolor=COLORS['panel'])
        style.map('TNotebook.Tab', background=[('selected', COLORS['selected']), ('active', COLORS['panel'])],
                  foreground=[('selected', COLORS['text']), ('active', COLORS['text'])])
        style.layout('Workbench.TNotebook.Tab', [])
        style.configure('Workbench.TNotebook', tabmargins=0, borderwidth=0,
                        bordercolor=COLORS['bg'], lightcolor=COLORS['bg'], darkcolor=COLORS['bg'])
        for name in ('Horizontal.TScrollbar', 'Vertical.TScrollbar'):
            style.configure(name, background=COLORS['input'], troughcolor=COLORS['bg'],
                            bordercolor=COLORS['panel'], arrowcolor=COLORS['text'])
            style.map(name, background=[('active', COLORS['muted']), ('pressed', COLORS['muted'])],
                      arrowcolor=[('active', COLORS['panel']), ('pressed', COLORS['panel'])])
        Tk控件样式(self)

    def _label(self, parent, text='', size=10, color='text', bold=False, **options):
        if color == 'primary':
            color = 'text'  # 灰褐主色只作填充，浅色主色不可当正文。
        return tk.Label(parent, text=text, bg=parent.cget('bg'), fg=COLORS.get(color, color),
                        font=(FONT, size, 'bold' if bold else 'normal'), **options)

    def _button(self, parent, text, command, primary=False, compact=False, danger=False):
        return Tk按钮(parent, text, command, primary=primary, compact=compact, danger=danger)

    def _enable(self, button, enabled, primary=False):
        button.configure(state='normal' if enabled else 'disabled', cursor='hand2' if enabled else 'arrow')

    def _card(self, parent, padding=16):
        return Tk卡片(parent, padding)

    def _robot_fields(self):
        is_ur = self.robot_kind.get() == ROBOT_KINDS[1]
        self.piper_fields.pack_forget()
        self.ur_fields.pack_forget()
        (self.ur_fields if is_ur else self.piper_fields).pack(fill='x')
        self.title('TCP标定与视觉验证 · '+self.robot_label())
        self.capture_heading.configure(text='TCP 标定  /  '+self.robot_label())
        for panel in self._motion_panels():
            self.pages.tab(panel, state='disabled' if is_ur else 'normal')
        if is_ur:
            self.pages.select(0)
        self._refresh_navigation()
        self.teaching_note.configure(text='UR 当前仅读取位姿；拖动示教与急停复位请使用 UR 示教器。'
                                    if is_ur else '先可靠支撑机械臂，再确认恢复手动拖动；恢复示教不会自动执行路径。')

    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_shell()
        self.pages = ttk.Notebook(self, style='Workbench.TNotebook')
        self.pages.grid(row=1, column=1, sticky='nsew')
        calibration_page = tk.Frame(self.pages, bg=COLORS['bg'])
        self.pages.add(calibration_page, text='TCP 标定')
        calibration = Tk滚动区(calibration_page, width=1200, background='bg')
        calibration.columnconfigure(0, weight=1)
        calibration.rowconfigure(2, weight=1, minsize=340)
        header = tk.Frame(calibration, bg=COLORS['bg'], padx=24, pady=14)
        header.grid(row=0, column=0, sticky='ew')
        actions = tk.Frame(header, bg=COLORS['bg'])
        actions.pack(side='right')
        self._button(actions, '打开日志', self.open_log, compact=True).pack(side='left', padx=8)
        self.reset_button = self._button(actions, '清空本轮', self.reset, compact=True)
        self.reset_button.pack(side='left')
        self.capture_heading = self._label(header, 'TCP 标定', 18, bold=True)
        self.capture_heading.pack(anchor='w')
        self._label(header, '打开相机 → 选择并连接机械臂 → 开始采集 → 每次确认触点 → 停止审查 → 保存结果', 10, 'muted').pack(anchor='w', pady=(5, 0))

        top = tk.Frame(calibration, bg=COLORS['bg'])
        top.grid(row=1, column=0, sticky='ew', padx=24, pady=(0, 14))
        top.columnconfigure(0, weight=1, uniform='capture')
        top.columnconfigure(1, weight=1, uniform='capture')
        outer, holder = self._card(top, 14)
        outer.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        self.camera = CameraPanel(holder, self.logger, on_status=self._status)
        self.camera.pack(fill='both', expand=True)
        outer, controls = self._card(top, 18)
        outer.grid(row=0, column=1, sticky='nsew')
        self._label(controls, '机械臂与触点确认', 14, bold=True).pack(anchor='w')
        line = tk.Frame(controls, bg=COLORS['panel'])
        line.pack(fill='x', pady=(8, 6))
        self._label(line, '机械臂', 10).pack(side='left', padx=(0, 8))
        self.robot_combo = ttk.Combobox(line, textvariable=self.robot_kind, values=ROBOT_KINDS, state='readonly', width=24)
        self.robot_combo.pack(side='left', padx=(0, 10))
        self.connect_button = self._button(line, '连接机械臂', self.toggle_robot, compact=True)
        self.connect_button.pack(side='left')
        parameters = tk.Frame(controls, bg=COLORS['panel'])
        parameters.pack(fill='x', pady=(0, 6))
        self.piper_fields = tk.Frame(parameters, bg=COLORS['panel'])
        self._label(self.piper_fields, '固件', 10).pack(side='left', padx=(0, 8))
        self.firmware_combo = ttk.Combobox(self.piper_fields, textvariable=self.robot_firmware, values=tuple(FIRMWARES), state='readonly', width=21)
        self.firmware_combo.pack(side='left')
        self.ur_fields = tk.Frame(parameters, bg=COLORS['panel'])
        self._label(self.ur_fields, '控制柜 IP', 10).pack(side='left', padx=(0, 6))
        self.ur_host_entry = ttk.Entry(self.ur_fields, textvariable=self.ur_host, width=17)
        self.ur_host_entry.pack(side='left', padx=(0, 10))
        self._label(self.ur_fields, '型号（选填）', 10).pack(side='left', padx=(0, 6))
        self.ur_model_entry = ttk.Entry(self.ur_fields, textvariable=self.ur_model, width=10)
        self.ur_model_entry.pack(side='left')
        self.robot_status = self._label(controls, 'PiPER X · can1 未连接 · 选择匹配的固件后连接', 10, 'muted', wraplength=625, justify='left', anchor='w')
        self.robot_status.pack(fill='x', pady=(0, 7))
        self.live_pose = self._label(controls, 'XYZ（mm） —\nRPY（°） —', 11, anchor='w', justify='left')
        self.live_pose.pack(fill='x', pady=(0, 6))
        row = tk.Frame(controls, bg=COLORS['panel'])
        row.pack(fill='x')
        self.start_button = self._button(row, '开始采集', self.start_capture, primary=True)
        self.start_button.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.confirm_button = self._button(row, '确认触点并采集', self.confirm_contact, primary=True)
        self.confirm_button.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.stop_button = self._button(row, '停止并审查', self.stop_capture, danger=True)
        self.stop_button.pack(side='left', fill='x', expand=True)
        self.capture_state = self._label(controls, '未开始 · 每次采点都需确认触点', 10, 'muted')
        self.capture_state.pack(anchor='w', pady=(5, 4))
        self.stability_status = self._label(controls, '确认后停稳 1 秒采一条；未确认时不会记录。', 10, 'muted', wraplength=625, justify='left')
        self.stability_status.pack(anchor='w')
        box = tk.Frame(controls, bg=COLORS['selected'], padx=12, pady=8)
        box.pack(fill='both', expand=True, pady=(6, 0))
        self.guidance = self._label(box, '开始采集后，每次让针尖轻触同一固定点，再点“确认触点并采集”。停稳只用于检查抖动，不能判断是否接触。',
                                    10, wraplength=600, justify='left', anchor='nw')
        self.guidance.pack(fill='both', expand=True)

        bottom = tk.Frame(calibration, bg=COLORS['bg'])
        bottom.grid(row=2, column=0, sticky='nsew', padx=24)
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, minsize=340)
        bottom.rowconfigure(0, weight=1)
        outer, data = self._card(bottom, 14)
        outer.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        toolbar = tk.Frame(data, bg=COLORS['panel'])
        toolbar.pack(fill='x')
        self._label(toolbar, '采集记录与筛选', 14, bold=True).pack(side='left')
        self.filter_combo = ttk.Combobox(toolbar, textvariable=self.filter_value,
            values=('全部数据', '合格', '不合格', '待审查', '待复核'), width=9, state='readonly')
        self.filter_combo.pack(side='right')
        self.filter_combo.bind('<<ComboboxSelected>>', lambda _event: self._render_records())
        self.count_label = self._label(data, '', 10, 'muted')
        self.count_label.pack(anchor='w', pady=(6, 8))
        table = tk.Frame(data, bg=COLORS['panel'])
        table.pack(fill='both', expand=True)
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        columns = ('id', 'time', 'rotation', 'angle', 'residual', 'status', 'reason')
        self.table = ttk.Treeview(table, columns=columns, show='headings', height=2, selectmode='extended')
        for key, label, width in zip(columns,
            ('序号', '采集时间', 'RPY（°）', '姿态差（°）', '残差（mm）', '结论', '原因'),
            (48, 90, 155, 95, 95, 80, 265)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=width if key != 'reason' else 180, stretch=key == 'reason', anchor='w')
        self.table.grid(row=0, column=0, sticky='nsew')
        self.table.tag_configure('合格', foreground=COLORS['green'])
        self.table.tag_configure('不合格', foreground=COLORS['error'])
        self.table.tag_configure('待复核', foreground=COLORS['warning'])
        self.table.tag_configure('待审查', foreground=COLORS['muted'])
        vbar = ttk.Scrollbar(table, orient='vertical', command=self.table.yview)
        vbar.grid(row=0, column=1, sticky='ns')
        hbar = ttk.Scrollbar(table, orient='horizontal', command=self.table.xview)
        hbar.grid(row=1, column=0, sticky='ew')
        self.table.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.table.bind('<<TreeviewSelect>>', self._selection_changed)
        self.detail_label = self._label(data, '选择一条记录，查看完整位姿、稳定范围及审查原因。', 10, 'muted',
                                         wraplength=920, justify='left', anchor='w', height=2)
        self.detail_label.pack(side='bottom', fill='x', pady=(6, 4), before=table)
        row = tk.Frame(data, bg=COLORS['panel'])
        row.pack(side='bottom', fill='x', before=self.detail_label)
        self.delete_bad_button = self._button(row, '删除不合格', self.delete_rejected, compact=True)
        self.delete_bad_button.pack(side='left', padx=(0, 8))
        self.delete_selected_button = self._button(row, '删除选中', self.delete_selected, compact=True)
        self.delete_selected_button.pack(side='left', padx=(0, 8))
        self.review_button = self._button(row, '重新审查 / 计算 TCP', self.review_records, compact=True)
        self.review_button.pack(side='left')
        self.export_button = self._button(row, '导出采集记录', self.export_records, compact=True)
        self.export_button.pack(side='right')

        outer, summary = self._card(bottom, 16)
        outer.grid(row=0, column=1, sticky='nsew')
        self.summary_tabs = ttk.Notebook(summary)
        self.summary_tabs.pack(fill='both', expand=True)
        criteria_page = tk.Frame(self.summary_tabs, bg=COLORS['panel'], padx=7, pady=6)
        self.result_page = tk.Frame(self.summary_tabs, bg=COLORS['panel'], padx=7, pady=12)
        self.summary_tabs.add(criteria_page, text='审查标准')
        self.summary_tabs.add(self.result_page, text='TCP 结果')
        self._label(criteria_page, '审查标准 · 至少 4 条', 12, bold=True).pack(anchor='w', pady=(0, 8))
        for name, label, unit in (
            ('min_angle_deg', '最小姿态差 ≥', '°'), ('max_condition_number', '条件数 ≤', ''),
            ('max_rms_mm', '拟合 RMS ≤', 'mm'), ('max_error_mm', '单点残差 ≤', 'mm')):
            line = tk.Frame(criteria_page, bg=COLORS['panel'])
            line.pack(fill='x', pady=2)
            self._label(line, label, 10, 'muted', width=13, anchor='w').pack(side='left')
            ttk.Entry(line, textvariable=self.criteria_vars[name], width=7).pack(side='left')
            self._label(line, unit, 10, 'muted').pack(side='left', padx=5)
        self._label(self.result_page, 'TCP 偏移 · 法兰坐标系', 12, bold=True).pack(anchor='w', pady=(4, 8))
        self.offset_label = self._label(self.result_page, 'X  —    Y  —    Z  —  mm', 11, 'primary', bold=True, justify='left', anchor='w')
        self.offset_label.pack(fill='x', pady=(0, 5))
        self.result_label = self._label(self.result_page, '等待采集与审查', 10, 'muted', justify='left', anchor='w', wraplength=285)
        self.result_label.pack(fill='x')
        self.save_button = self._button(self.result_page, '保存 TCP 结果', self.save_result, primary=True)
        self.save_button.pack(side='bottom', fill='x', pady=(9, 0))
        self._label(self.result_page, '合格表示数据符合当前门槛，\n不代表已验证整机 0.1 mm 精度。', 9, 'muted', justify='left').pack(side='bottom', anchor='w', pady=(7, 0))

        footer = tk.Frame(self, bg=COLORS['bg'], padx=20, pady=10)
        footer.grid(row=2, column=1, sticky='ew')
        recovery = tk.Frame(footer, bg=COLORS['bg'])
        recovery.pack(fill='x', pady=(0, 7))
        self.teaching_recovery_button = self._button(
            recovery, '解除急停并恢复示教', self.recover_teaching, compact=True)
        self.teaching_recovery_button.pack(side='left')
        self.teaching_note = self._label(recovery, '先可靠支撑机械臂，再确认恢复手动拖动；恢复示教不会自动执行路径。', 10, 'muted', anchor='w', justify='left')
        self.teaching_note.pack(side='left', fill='x', expand=True, padx=14)
        logbox = tk.Frame(footer, bg=COLORS['selected'], padx=12, pady=7)
        logbox.pack(fill='x')
        self._label(logbox, '运行状态', 10, bold=True).pack(side='left', padx=(0, 16))
        self._button(logbox, '打开日志', self.open_log, compact=True).pack(side='right', padx=(12, 0))
        self.status_label = self._label(logbox, '先打开相机、选择并连接机械臂，再点击开始采集。', 10, 'muted', anchor='w', justify='left', height=2)
        self.status_label.pack(fill='x')
        for label in (self.guidance, self.robot_status, self.detail_label, self.teaching_note, self.status_label):
            label.bind('<Configure>', lambda event, widget=label: widget.configure(wraplength=max(200, event.width - 4)))
        self._update_counts()
        self.hole_panel = HolePanel(self.pages, self)
        self.pages.add(self.hole_panel, text='选孔与运动')
        self.board_panel = BoardPanel(self.pages, self)
        self.pages.add(self.board_panel, text='标定板验证')
        self.pages.bind('<<NotebookTabChanged>>', self._refresh_navigation)

    def _build_shell(self):
        sidebar = tk.Frame(self, bg=COLORS['panel'], width=174, padx=12, pady=24,
                           highlightthickness=1, highlightbackground=COLORS['border'], highlightcolor=COLORS['border'])
        sidebar.grid(row=0, column=0, rowspan=3, sticky='nsew', padx=(0, 4))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(5, weight=1)
        self._label(sidebar, 'TCP 标定', 15, bold=True).grid(row=0, column=0, sticky='w', padx=10)
        self._label(sidebar, '视觉与运动工作台', 10, 'muted').grid(row=1, column=0, sticky='w', padx=10, pady=(5, 26))
        self.nav_buttons, self.nav_markers = [], []
        for index, title in enumerate(('TCP 标定', '选孔与运动', '标定板验证')):
            row = tk.Frame(sidebar, bg=COLORS['panel'])
            row.grid(row=index+2, column=0, sticky='ew', pady=5)
            marker = tk.Frame(row, bg=COLORS['panel'], width=3)
            marker.pack(side='left', fill='y')
            button = ttk.Button(row, text=f'{index+1:02d}  {title}', command=lambda i=index: self.pages.select(i),
                                style='Nav.TButton', cursor='hand2', takefocus=True)
            button.pack(side='left', fill='both', expand=True)
            self.nav_buttons.append(button)
            self.nav_markers.append(marker)
        self._label(sidebar, 'PiPER X / UR\n共用实时相机', 10, 'muted', justify='left').grid(row=6, column=0, sticky='w', padx=10)
        header = tk.Frame(self, bg=COLORS['bg'], padx=24, pady=16)
        header.grid(row=0, column=1, sticky='ew')
        self._label(header, 'TCP 标定与视觉验证', 18, bold=True).pack(side='left')
        self.device_badges = []
        for text in ('机械臂 · 未连接', '相机 · 未打开'):
            badge = self._label(header, text, 10, 'muted', padx=12, pady=6)
            badge.configure(bg=COLORS['selected'])
            badge.pack(side='right', padx=(10, 0))
            self.device_badges.append(badge)

    def _refresh_navigation(self, _event=None):
        if not self.pages.select():
            return
        selected = self.pages.index(self.pages.select())
        for index, (button, marker) in enumerate(zip(self.nav_buttons, self.nav_markers)):
            current = selected == index
            disabled = self.pages.tab(index, 'state') == 'disabled'
            button.configure(state='disabled' if disabled else 'normal', cursor='arrow' if disabled else 'hand2')
            button.state(['selected' if current else '!selected'])
            marker.configure(bg=COLORS['primary'] if current else COLORS['panel'])

    def _refresh_shell(self):
        if self._closing:
            return
        robot_live = self.robot_connected and self.live_snapshot is not None
        robot_text = '反馈正常' if robot_live else '等待反馈' if self.robot_connected else '连接中' if self.robot_connecting else '未连接'
        for badge, text, live in (
                (self.device_badges[0], '机械臂 · '+robot_text, robot_live),
                (self.device_badges[1], '相机 · '+('实时画面' if self.camera.is_live() else '未打开或已暂停'), self.camera.is_live())):
            badge.configure(text=text, bg=COLORS['green'] if live else COLORS['selected'],
                            fg=COLORS['panel'] if live else COLORS['muted'])
        self.after(500, self._refresh_shell)

    def _motion_panels(self):
        return tuple(panel for name in ('hole_panel', 'board_panel')
                     if (panel := getattr(self, name, None)) is not None)

    def task_blocked(self, owner):
        """所有入口在 Tk 线程中检查；任务置 busy 后才启动后台线程。"""
        if owner not in ('capture', 'hole', 'board'):
            raise ValueError('未知任务入口。')
        if self.teaching_recovery_busy:
            return '正在恢复手动示教，请等待状态确认；停止按钮仍可使用。'
        if owner != 'capture':
            if self.collecting:
                return 'TCP 标定正在采集，请先停止并完成审查。'
            if self.audit_busy:
                return 'TCP 数据正在审查，请等待审查完成。'
        for name, label in (('hole', '选孔与运动'), ('board', '标定板验证')):
            panel = getattr(self, name + '_panel', None)
            if name != owner and panel is not None and panel.busy:
                return f'“{label}”任务尚未结束，请先停止或等待完成。'
            if name != owner and name == 'hole' and panel is not None and panel.diagnostic_unsaved:
                return '自动跟随过程记录正在归档，请稍候。'
        return None

    def tasks_busy(self):
        return (self.teaching_recovery_busy or self.collecting or self.audit_busy
                or any(panel.busy for panel in self._motion_panels()))

    def recover_teaching(self):
        if self.robot_kind.get() == ROBOT_KINDS[1]:
            self._status('UR 当前仅读取位姿；请使用 UR 示教器恢复手动示教，本按钮不会发送指令。', error=True)
            return
        if self.tasks_busy():
            self._status('请先结束采集、审查或运动任务，再恢复示教。', error=True)
            return
        reader, generation = self.robot_reader, self.robot_generation
        if not self.robot_connected or reader is None:
            self._status('请先连接右臂 can1，再点击“解除急停并恢复示教”。', error=True)
            return
        self.teaching_recovery_busy = True
        self.teaching_recovery_cancelled = False
        self._controls()
        confirmed = messagebox.askokcancel(
            '解除急停并恢复示教',
            '复位可能改变电机保持力。请先确认：\n\n'
            '• 机械臂已有可靠支撑；\n'
            '• 针尖不顶住物体，人员避开夹点。\n\n'
            '确认后解除右臂急停并开启手动拖动示教。\n'
            '不会自动执行路径；后续运动须重新预览并确认。', parent=self, default='cancel')
        if not confirmed or self._closing:
            self.teaching_recovery_busy = False
            if not self._closing:
                self._controls()
            return
        if (not self.robot_connected or self.robot_reader is not reader
                or self.robot_generation != generation):
            self.teaching_recovery_busy = False
            self._status('右臂连接已变化，未发送恢复指令；请重新连接后操作。', error=True)
            self._controls()
            return
        # 手动拖动后所有旧路径和锁定目标都不再适用，不能继续旧任务。
        self.hole_panel.task_generation += 1
        self.hole_panel.plan = self.hole_panel.cycle = None
        self.hole_panel.route_checked.set(False)
        self.hole_panel.history.clear()
        self.board_panel.control_ready = False
        self.board_panel.invalidate('恢复示教后需重新选点并规划，再点击执行。')
        self._status('正在检查停稳并恢复示教，请保持支撑；停止按钮仍可使用。')
        self.logger.info('用户确认可靠支撑，手动请求解除急停并恢复示教')

        def worker():
            try:
                state = reader.recover_teaching(operator_confirmed=True)
                self.logger.info('示教恢复反馈确认：%s', state)
                outcome = (True, '')
            except Exception as exc:
                self.logger.exception('手动恢复示教未完成')
                outcome = (False, str(exc))
            self.teaching_recovery_events.put((generation, reader, outcome))

        threading.Thread(target=worker, name='右臂恢复示教', daemon=True).start()

    def _poll_teaching_recovery(self):
        try:
            generation, reader, (success, error) = self.teaching_recovery_events.get_nowait()
        except queue.Empty:
            return
        self.teaching_recovery_busy = False
        if (generation != self.robot_generation or reader is not self.robot_reader
                or not self.robot_connected):
            self._status('恢复期间连接已中断，无法确认示教状态；请重新连接检查。', error=True)
        elif self.teaching_recovery_cancelled:
            self._status('示教恢复已取消，请以最新停止提示为准。', error=True)
        elif success:
            self._status('控制器已确认解除急停、开启示教。保持支撑轻试拖动；旧路径不会继续执行。')
        else:
            self._status(f'示教恢复未完成：{error} 详情见“打开日志”。', error=True)

    def _save_current(self, _event=None):
        if self.pages.select() == str(self.board_panel):
            self._status('标定板过程记录与结果会自动保存，无需手动导出。')
        elif self.tasks_busy():
            self._status('请先结束当前采集、审查或运动任务，再保存。', error=True)
        else:
            self.save_result()
        return 'break'

    def _status(self, message, error=False):
        if getattr(self, '_last_status_message', None) == (message, error):
            return
        self._last_status_message = (message, error)
        self.status_label.configure(text=datetime.now().strftime('%H:%M:%S')+'  '+str(message).replace('\n', ' '),
                                    fg=COLORS['error'] if error else COLORS['muted'])

    def _criteria(self):
        return validate_criteria({key: variable.get().strip() for key, variable in self.criteria_vars.items()})

    def _criteria_changed(self, *_args):
        if self.suppress_changes:
            return
        if self.criteria_after is not None:
            self.after_cancel(self.criteria_after)
        if self.collecting:
            self.stop_capture('标准已修改，采集已停止；按新标准重新审查。', review=False)
        self._invalidate_review()
        self.criteria_after = self.after(350, self._review_after_edit)

    def _review_after_edit(self):
        self.criteria_after = None
        try:
            self._criteria()
        except ValueError as exc:
            self._status(str(exc), error=True)
            return
        if self.records:
            self.review_records()

    def _controls(self):
        operating = self.tasks_busy()
        busy = self.robot_connected or self.robot_connecting
        is_ur = self.robot_kind.get() == ROBOT_KINDS[1]
        self.connect_button.configure(text='取消连接' if self.robot_connecting else '断开连接' if self.robot_connected else '连接机械臂')
        self._enable(self.connect_button, not self.teaching_recovery_busy)
        for panel in self._motion_panels():
            self._enable(panel.robot_button, not self.teaching_recovery_busy)
        self.teaching_recovery_button.configure(text='UR 示教请使用示教器' if is_ur else
                                               '正在恢复示教…' if self.teaching_recovery_busy else '解除急停并恢复示教')
        self._enable(self.teaching_recovery_button,
                     not is_ur and self.robot_connected and not self.robot_connecting and not operating)
        identity_locked = busy or bool(self.records) or operating
        self.robot_combo.configure(state='disabled' if identity_locked else 'readonly')
        self.firmware_combo.configure(state='disabled' if identity_locked else 'readonly')
        for field in (self.ur_host_entry, self.ur_model_entry):
            field.configure(state='disabled' if identity_locked else 'normal')
        self._enable(self.start_button, not operating)
        self._enable(self.confirm_button, self.collecting and not self.gate.armed and self.task_blocked('capture') is None, primary=True)
        self._enable(self.stop_button, self.collecting)
        self._enable(self.reset_button, not operating and bool(self.records))
        self._enable(self.review_button, not operating and bool(self.records))
        self._enable(self.delete_bad_button, not operating and any(r['status'] == '不合格' for r in self.records))
        self._enable(self.delete_selected_button, not operating and bool(self.table.selection()))
        self._enable(self.export_button, not operating and bool(self.records))
        self._enable(self.save_button, not operating and self._can_save(), primary=True)

    def _can_save(self):
        return (not self.collecting and not self.audit_busy and self.audit is not None and self.audit['passed']
                and self.result is not None and len(self.records) >= 4 and all(r['status'] == '合格' for r in self.records))

    def _row_values(self, record):
        angle, residual = record['nearest_angle_deg'], record['residual_mm']
        return (record['id'], record['captured_at'][11:19], ' / '.join(f'{v:.1f}' for v in record['pose'][3:]),
                '—' if angle is None else f'{angle:.2f}', '—' if residual is None else f'{residual:.4f}',
                record['status'], record['reason'])

    def _insert_record(self, record):
        if self.filter_value.get() not in ('全部数据', record['status']):
            return
        iid = str(record['id'])
        self.table.insert('', 'end', iid=iid, values=self._row_values(record), tags=(record['status'],))
        if self.collecting:
            self.table.see(iid)

    def _render_records(self):
        selected = self.table.selection()
        self.table.delete(*self.table.get_children())
        for record in self.records:
            self._insert_record(record)
        self.table.selection_set([iid for iid in selected if self.table.exists(iid)])
        self._selection_changed()
        self._update_counts()

    def _update_counts(self):
        count = Counter(r['status'] for r in self.records)
        self.count_label.configure(text=f"共 {len(self.records)} 条   ·   合格 {count['合格']}   ·   不合格 {count['不合格']}   ·   待审查 / 复核 {count['待审查'] + count['待复核']}")

    def _selection_changed(self, _event=None):
        selected = set(self.table.selection())
        record = next((r for r in self.records if str(r['id']) in selected), None)
        if record:
            stable = record['stability']
            self.detail_label.configure(text=f"第 {record['id']} 条  XYZ（mm）：" + ' / '.join(f'{v:.4f}' for v in record['pose'][:3])
                + f"  ·  停稳 {stable['duration_s']:.2f}s，位移 {stable['translation_range_mm']:.4f}mm，转角 {stable['rotation_range_deg']:.4f}°\n"
                + record['reason'])
        else:
            self.detail_label.configure(text='选择一条记录，查看完整位姿、稳定范围及审查原因。')
        if hasattr(self, 'save_button'):
            self._controls()

    def _invalidate_review(self):
        self.audit_generation += 1
        self.audit_busy = False
        self.audit = self.result = None
        self.offset_label.configure(text='X  —    Y  —    Z  —  mm')
        self.result_label.configure(text='数据或标准已变化，等待重新审查。', fg=COLORS['muted'])
        for record in self.records:
            record.update(status='待审查', reason='等待按当前数据与标准审查。', residual_mm=None, nearest_angle_deg=None)
        self._render_records()
        self._controls()

    def review_records(self):
        blocked = self.task_blocked('capture')
        if blocked:
            self._status(blocked, error=True)
            return
        if self.collecting:
            self._status('请先停止采集，再审查这批数据。', error=True)
            return
        if not self.records:
            return
        try:
            self._records_robot()
            criteria = self._criteria()
        except ValueError as exc:
            self._invalidate_review()
            self._status(str(exc), error=True)
            return
        self._invalidate_review()
        self.audit_busy = True
        generation = self.audit_generation
        samples = [list(record['pose']) for record in self.records]
        self.result_label.configure(text='正在检查姿态分布与触点一致性…')
        self._controls()

        def worker():
            try:
                self.review_events.put((generation, review_dataset(samples, 'rpy_deg', criteria), None))
            except Exception as exc:
                self.logger.exception('批量 TCP 审查失败')
                self.review_events.put((generation, None, str(exc)))

        threading.Thread(target=worker, name='TCP批量审查', daemon=True).start()

    def _poll_review(self):
        while True:
            try:
                generation, audit, error = self.review_events.get_nowait()
            except queue.Empty:
                break
            if generation != self.audit_generation or self.collecting:
                continue
            self.audit_busy = False
            if error:
                self._status('审查失败：' + error, error=True)
                self.result_label.configure(text='审查失败，无法保存结果。', fg=COLORS['error'])
                continue
            self._apply_review(audit)

    def _apply_review(self, audit):
        if len(audit['rows']) != len(self.records):
            raise ValueError('审查结果与当前记录数量不一致，请重新审查。')
        audit = copy.deepcopy(audit)
        audit['record_ids'] = [record['id'] for record in self.records]
        for row in audit['rows']:
            row['reason'] = re.sub(r'第 (\d+) 条',
                lambda match: f"第 {audit['record_ids'][int(match[1]) - 1]} 条"
                if 1 <= int(match[1]) <= len(self.records) else match[0], row['reason'])
        self.audit = audit
        self.result = audit['result']
        if not isinstance(self.focus_get(), (tk.Entry, ttk.Entry)):
            self.summary_tabs.select(self.result_page)
        for record, checked in zip(self.records, audit['rows']):
            record.update(checked)
        self._render_records()
        self.guidance.configure(text=audit['guidance'])
        if self.result:
            x, y, z = self.result['tcp_offset_mm']
            self.offset_label.configure(text=f'X {x:+.4f} mm\nY {y:+.4f} mm\nZ {z:+.4f} mm')
            all_good = all(r['status'] == '合格' for r in self.records)
            note = '全部合格，可以保存。' if all_good else '请删除不合格记录后再保存。'
            self.result_label.configure(text=f"使用 {len(audit['eligible_indices'])} 条合格数据\nRMS {self.result['rms_mm']:.4f} mm · 最大 {self.result['max_error_mm']:.4f} mm\n{note}",
                                        fg=COLORS['green'] if all_good else COLORS['warning'])
        else:
            self.result_label.configure(text=audit['summary'], fg=COLORS['warning'])
        self._status(audit['summary'])
        self._controls()
        self.logger.info('批量审查完成，合格 %d / %d 条', len(audit['eligible_indices']), len(self.records))

    def delete_rejected(self):
        if self.collecting or self.audit_busy:
            return
        ids = {record['id'] for record in self.records if record['status'] == '不合格'}
        self._delete_ids(ids)

    def delete_selected(self):
        if self.collecting or self.audit_busy:
            return
        self._delete_ids({int(iid) for iid in self.table.selection()})

    def _delete_ids(self, ids):
        if not ids:
            return
        self.records = [record for record in self.records if record['id'] not in ids]
        self._invalidate_review()
        self.logger.info('用户删除 %d 条采集记录', len(ids))
        if self.records:
            self.review_records()
        else:
            self.guidance.configure(text='记录已清空。可以点击开始采集，继续采集新的姿态。')
        self._status(f'已删除 {len(ids)} 条；保留 {len(self.records)} 条，自动重新审查。')

    def _base_payload(self):
        return {
            'format_version': 5, 'saved_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'method': 'multi_pose_pivot', 'robot': self._records_robot(),
            'pose_frame': 'base_from_flange', 'position_unit': 'mm', 'rotation_mode': 'rpy_deg',
            'contact_basis': '操作者逐点点击确认触点，程序随后检查停稳；普通刚性针无自动接触检测。',
            'stability_criteria': DEFAULT_STABILITY,
            'criteria': self._criteria(), 'records': copy.deepcopy(self.records),
            'audit': copy.deepcopy(self.audit),
        }

    def _save_json(self, payload, filename, title):
        path = filedialog.asksaveasfilename(parent=self, title=title, initialdir=self.project_dir / '结果',
                    initialfile=filename, defaultextension='.json', filetypes=[('JSON 数据', '*.json')])
        if not path:
            return False
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        except (OSError, ValueError) as exc:
            self.logger.exception('保存 TCP 数据失败')
            messagebox.showerror('保存失败', str(exc), parent=self)
            return False
        self._status('已保存：' + Path(path).name)
        self.logger.info('用户显式导出 %s', title)
        return True

    def export_records(self):
        if self.collecting or self.audit_busy or not self.records:
            return
        try:
            payload = self._base_payload()
        except ValueError as exc:
            self._status(str(exc), error=True)
            return
        payload['kind'] = 'tcp_capture_review'
        filename = 'UR_TCP采集记录.json' if self.robot_kind.get() == ROBOT_KINDS[1] else 'TCP采集记录.json'
        self._save_json(payload, filename, '导出采集记录与审查状态')

    def save_result(self):
        if not self._can_save():
            self._status('请先停止采集，完成审查，并删除不合格记录；全部保留数据合格后才能保存 TCP。', error=True)
            return
        try:
            payload = self._base_payload()
        except ValueError as exc:
            self._status(str(exc), error=True)
            return
        payload.update(kind='accepted_tcp_calibration', result=self.result,
                       transform_direction='flange_from_tcp',
                       tool_orientation_definition='工具坐标轴与法兰坐标轴平行，仅求针尖位置偏移。',
                       scope='采点和拟合质量符合当前门槛，不代表整机绝对定位精度已验证。')
        filename = 'UR_TCP标定结果.json' if self.robot_kind.get() == ROBOT_KINDS[1] else 'TCP标定结果.json'
        self._save_json(payload, filename, '保存合格 TCP 标定结果')

    def reset(self):
        if self.collecting or self.audit_busy:
            return
        if self.records and not messagebox.askyesno('清空本轮', '清空当前窗口的全部采集记录和结果？已保存的文件不受影响。', parent=self):
            return
        self.records = []
        self.next_record_id = 1
        self._invalidate_review()
        self.capture_state.configure(text='未开始 · 每次采点都需确认触点', fg=COLORS['muted'])
        self.guidance.configure(text='当前记录已清空。开始采集后，每次对好固定点，再点“确认触点并采集”。')
        self._status('已清空本轮数据，连接与相机预览保持可用。')

    def open_log(self):
        path = self.project_dir / '日志' / '运行.log'
        try:
            subprocess.Popen(['xdg-open', str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            messagebox.showerror('无法打开日志', f'{exc}\n{path}', parent=self)

    def report_callback_exception(self, exception, value, traceback):
        self.logger.error('界面操作异常', exc_info=(exception, value, traceback))
        messagebox.showerror('操作未完成', f'{value}\n详细信息见“打开日志”。', parent=self)

    def _close(self):
        for panel in self._motion_panels():
            if panel.busy:
                panel.stop_motion('关闭窗口前，已请求停止运动任务。')
        pending_panels = [panel for panel in self._motion_panels() if not panel.prepare_close()]
        if pending_panels:
            save_errors = [getattr(panel, 'follow_save_error', getattr(panel, 'save_error', None))
                           for panel in pending_panels]
            if any(save_errors):
                self._status('过程记录保存失败，窗口保留：' + '；'.join(filter(None, save_errors)), error=True)
            else:
                self._status('正在自动保存过程记录，完成后关闭。')
            self.after(500, self._close)
            return
        pending = []
        if self.records:
            pending.append('TCP 采集记录')
        if pending and not messagebox.askyesno('关闭 TCP 标定与视觉验证',
                '关闭后，本次窗口中的' + '、'.join(pending) + '将被清空。请确认已导出需要保留的数据。', parent=self):
            return
        self._closing = True
        for panel in self._motion_panels():
            panel.close()
        self.collecting = False
        self.gate.stop()
        self.audit_generation += 1
        self.disconnect_robot()
        self.camera.close()
        self.logger.info('关闭 TCP 连续采集界面')
        self.destroy()


def main(project_dir, logger):
    app = CalibrationApp(project_dir, logger)
    controller = getattr(logger, 'instance_controller', None)
    if controller is not None:
        controller.attach(app)
    app.mainloop()
