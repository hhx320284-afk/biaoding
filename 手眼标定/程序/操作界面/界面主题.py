"""界面主题：十九套配色、对比度核算、样式生成与常用控件。

配色只取 AGENTS.md 里那十九套方案，一行一套，各项颜色都从同一行取，不跨行混搭。
其他项目要用这套界面，直接把这个文件复制到自己的 程序/操作界面/ 即可，
控件颜色一律走 方案["..."]，不要在界面代码里另写色值。

用法（四行）::

    import 界面主题 as 主题
    方案 = 主题.取方案("雾青灰绿")   # 名字见 主题.方案名列表
    窗口.setStyleSheet(主题.生成样式(方案))
    卡, 布局 = 主题.卡片("标题"); 布局.addWidget(主题.按钮("开始", "主按钮"))

现成的对象名（QSS 里已经写好，不要自己再写颜色）：
    页面      整窗背景，中央控件用它
    卡片      白/深色面板 + 圆角边框；分隔线 为 1px 细线
    页标题    24px 粗体；页说明、弱提示、指标名、单位 都是次要文字
    卡片标题  16px 粗体；正文 14px
    指标值    26px 粗体（配合 指标名 / 单位 用 指标() 拼）
    主按钮    深色主操作；危险按钮 红色；链接按钮 纯文字
    徽章      主色徽章；徽章成功 / 徽章警告 / 徽章错误 三态
    侧栏      左侧导航底板；导航项 为可勾选按钮，选中项带主色左边线
    表格      只读表格；表头用选中底色，隔行浅底，无竖表头

描边：方案["描边"] 由本行 muted 混入 card 算出，对卡片和页面背景都刚好 ≥3:1，
用于输入框、下拉框、按钮和列表项，保证控件边界在深浅方案下都看得见。

控件工厂：卡片()、分隔线()、按钮()、徽章()、指标()、导航栏()、表格()、
日志面板()、页面骨架()。整页起步直接用 页面骨架(标题, 导航项, 表头, 数据, 日志行)，
其余按需删改。

工具函数：对比度()、检查对比度(方案)、检查描边(方案)、刷新样式(控件)、
设徽章状态(徽章, 状态)。
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# 中文字体：本机 Noto Sans CJK SC 齐全，缺失时 Qt 会回退到其余无衬线字体。
字体名 = "Noto Sans CJK SC"
字体族 = f'"{字体名}"'

# 状态色所有方案共用：一律做成实心徽章，白字压在上面，不随方案变化。
状态色 = {"成功": "#1F7A4D", "警告": "#8A5A12", "错误": "#A3343F"}

# 十九套方案，字段与 AGENTS.md 的表格一一对应。
# 取舍标准：任意两套的主按钮色在 LAB 空间里 ΔE ≥ 22，避免出现“看着几乎一样”的两套。
方案表: tuple[dict[str, str], ...] = (
    {"名": "雾青灰绿", "用途": "设备操作、状态面板，冷静克制",
     "bg": "#F7FAF9", "card": "#FFFFFF", "text": "#161C1A", "muted": "#586360",
     "primary": "#0F6B5A", "primary_fg": "#FFFFFF", "selected": "#E3F2EE"},
    {"名": "学术靛蓝", "用途": "仪器控制、参数配置，期刊气质",
     "bg": "#F7F8FB", "card": "#FFFFFF", "text": "#171A22", "muted": "#5A6273",
     "primary": "#3C5488", "primary_fg": "#FFFFFF", "selected": "#E8EDF6"},
    {"名": "暖砂陶土", "用途": "资料管理、报告浏览，温润质朴",
     "bg": "#FAF7F3", "card": "#FFFFFF", "text": "#1F1A16", "muted": "#63594F",
     "primary": "#8A4B2A", "primary_fg": "#FFFFFF", "selected": "#F6ECE2"},
    {"名": "深空青", "用途": "暗色监控、实时波形，冷冽专注",
     "bg": "#12141A", "card": "#1A1D25", "text": "#ECEEF2", "muted": "#A7AFBE",
     "primary": "#7CD7EE", "primary_fg": "#0F1418", "selected": "#1E2B36"},
    {"名": "珊瑚绯红", "用途": "告警复核、关键结果，醒目克制",
     "bg": "#FDF7F6", "card": "#FFFFFF", "text": "#221715", "muted": "#6A5753",
     "primary": "#A33F4C", "primary_fg": "#FFFFFF", "selected": "#FAEAEC"},
    {"名": "石墨靛黑", "用途": "暗色仪表盘、长时间值守，低刺激",
     "bg": "#111318", "card": "#181B22", "text": "#E9ECF2", "muted": "#A5ADBC",
     "primary": "#E9B873", "primary_fg": "#1A1408", "selected": "#26303D"},
    # 以下五套主按钮色取自公开发表的科研配色，其余角色由其派生：
    # NPG（Nature 系）朱红与青绿、Okabe-Ito 色盲友好橙、Viridis 深紫、Lancet 紫。
    {"名": "期刊朱红", "用途": "结果标注、重点数据（NPG）",
     "bg": "#FEF9F8", "card": "#FFFFFF", "text": "#160D0C", "muted": "#6B6565",
     "primary": "#E64B35", "primary_fg": "#12100E", "selected": "#FCEBE9"},
    {"名": "色盲友好橙", "用途": "分类对比、配图（Okabe-Ito）",
     "bg": "#FEFCF6", "card": "#FFFFFF", "text": "#161209", "muted": "#6C6963",
     "primary": "#E69F00", "primary_fg": "#12100E", "selected": "#FCF3DE"},
    {"名": "热图紫", "用途": "热力图、聚类结果（Viridis）",
     "bg": "#F8F6F9", "card": "#FFFFFF", "text": "#0D090E", "muted": "#626063",
     "primary": "#440154", "primary_fg": "#FFFFFF", "selected": "#EAE3EC"},
    {"名": "孔雀青", "用途": "趋势展示、多样本（NPG 青绿）",
     "bg": "#F6FCFB", "card": "#FFFFFF", "text": "#091211", "muted": "#636969",
     "primary": "#00A087", "primary_fg": "#12100E", "selected": "#E3F5F2"},
    {"名": "柳叶刀紫", "用途": "临床数据、多组比较（Lancet）",
     "bg": "#FBF9FC", "card": "#FFFFFF", "text": "#120E12", "muted": "#686568",
     "primary": "#925E9F", "primary_fg": "#FFFFFF", "selected": "#F3EDF4"},
    {"名": "实验亮绿", "用途": "数据曲线、分组标记（Lancet）",
     "bg": "#F8FCF8", "card": "#FFFFFF", "text": "#0D130D", "muted": "#656865",
     "primary": "#42B540", "primary_fg": "#12100E", "selected": "#E6F5E6"},
    {"名": "科研黄", "用途": "高亮标注、色盲友好（Okabe-Ito）",
     "bg": "#FEFEF8", "card": "#FFFFFF", "text": "#17160D", "muted": "#6E6D68",
     "primary": "#F0E442", "primary_fg": "#12100E", "selected": "#FDFBE6"},
    {"名": "论文灰蓝", "用途": "附录图表、旁注（NPG 灰紫）",
     "bg": "#FBFBFC", "card": "#FFFFFF", "text": "#111213", "muted": "#676768",
     "primary": "#8491B4", "primary_fg": "#12100E", "selected": "#EFF1F5"},
    {"名": "单色石墨", "用途": "灰度印刷、黑白对照（中性）",
     "bg": "#F8F8F8", "card": "#FFFFFF", "text": "#0D0D0D", "muted": "#616161",
     "primary": "#3F3F3F", "primary_fg": "#FFFFFF", "selected": "#E6E6E6"},
    {"名": "等离子品红", "用途": "重点高亮、分类曲线（Plasma）",
     "bg": "#FAF6FC", "card": "#FFFFFF", "text": "#110913", "muted": "#635D64",
     "primary": "#7D03A8", "primary_fg": "#FFFFFF", "selected": "#EEDEF4"},
    {"名": "数据蓝绿", "用途": "连续数据、等值线（Viridis）",
     "bg": "#F8FAFB", "card": "#FFFFFF", "text": "#0C1011", "muted": "#626465",
     "primary": "#2A788E", "primary_fg": "#FFFFFF", "selected": "#E3EDF0"},
    {"名": "灰褐档案", "用途": "历史记录、归档（NPG）",
     "bg": "#FCFCFB", "card": "#FFFFFF", "text": "#131211", "muted": "#696868",
     "primary": "#B09C85", "primary_fg": "#12100E", "selected": "#F5F2EF"},
    {"名": "珊瑚橘", "用途": "提示强调、待处理（NPG 橘）",
     "bg": "#FFFCFB", "card": "#FFFFFF", "text": "#171211", "muted": "#6C6968",
     "primary": "#F39B7F", "primary_fg": "#12100E", "selected": "#FDF2EE"},
)

方案名列表 = tuple(方案["名"] for 方案 in 方案表)
最小对比度 = 4.5


def 取方案(名: str) -> dict[str, str]:
    名字 = {方案["名"]: 方案 for 方案 in 方案表}
    if 名 not in 名字:
        raise KeyError(f"没有名为「{名}」的配色方案")
    return 名字[名]


def _相对亮度(颜色: str) -> float:
    """WCAG 相对亮度，输入 #RRGGBB。"""
    通道 = []
    for 起点 in (1, 3, 5):
        值 = int(颜色[起点:起点 + 2], 16) / 255
        通道.append(值 / 12.92 if 值 <= 0.03928 else ((值 + 0.055) / 1.055) ** 2.4)
    红, 绿, 蓝 = 通道
    return 0.2126 * 红 + 0.7152 * 绿 + 0.0722 * 蓝


def 对比度(前景: str, 背景: str) -> float:
    """两色的对比度倍数，1.0 表示完全相同。"""
    亮者, 暗者 = sorted((_相对亮度(前景), _相对亮度(背景)), reverse=True)
    return (亮者 + 0.05) / (暗者 + 0.05)


# 界面上真实出现的文字组合，换方案或调色后逐条核算。
对比度检查项 = (
    ("正文 / 页面背景", "text", "bg"),
    ("正文 / 卡片", "text", "card"),
    ("正文 / 选中底色", "text", "selected"),
    ("辅助文字 / 页面背景", "muted", "bg"),
    ("辅助文字 / 卡片", "muted", "card"),
    ("辅助文字 / 选中底色", "muted", "selected"),
    ("主按钮文字 / 主按钮", "primary_fg", "primary"),
    ("白字 / 成功色", "#FFFFFF", "成功"),
    ("白字 / 警告色", "#FFFFFF", "警告"),
    ("白字 / 错误色", "#FFFFFF", "错误"),
)


def 检查对比度(方案: dict[str, str]) -> list[tuple[str, float, bool]]:
    """返回每条文字组合的对比度与是否达标。"""
    查表 = dict(方案) | 状态色
    结果 = []
    for 说明, 前, 背 in 对比度检查项:
        比值 = 对比度(查表.get(前, 前), 查表.get(背, 背))
        结果.append((说明, 比值, 比值 >= 最小对比度))
    return 结果


# 控件描边不属于文字，按下限 3:1 核算（WCAG 非文本对比度）。
最小描边对比度 = 3.0


def _混色(前: str, 后: str, 比例: float) -> str:
    """把「前」按比例混进「后」，比例 0 只有后色，1 只有前色。"""
    值 = []
    for 起点 in (1, 3, 5):
        甲 = int(前[起点:起点 + 2], 16)
        乙 = int(后[起点:起点 + 2], 16)
        值.append(round(甲 * 比例 + 乙 * (1 - 比例)))
    return "#" + "".join(f"{v:02X}" for v in 值)


def 描边色(方案: dict[str, str]) -> str:
    """从本方案的辅助文字色与卡片色混出刚好达到 3:1 的描边。

    配色表里没有单独的边框色，硬写一个就会跨出方案；做法是只拿本行已有的
    muted 往 card 里混，色相不变，二分到对卡片和页面背景都刚好 3:1 就停，
    避免描边过重。
    """
    低, 高 = 0.0, 0.75
    for _ in range(20):
        中 = (低 + 高) / 2
        试 = _混色(方案["muted"], 方案["card"], 中)
        if min(对比度(试, 方案["card"]), 对比度(试, 方案["bg"])) < 最小描边对比度:
            低 = 中
        else:
            高 = 中
    return _混色(方案["muted"], 方案["card"], 高)


# 描边可辨识度：交互控件必须有可见边界，否则深色方案下输入框会“消失”。
描边检查项 = (
    ("描边 / 卡片", "card"),
    ("描边 / 页面背景", "bg"),
)


def 检查描边(方案: dict[str, str]) -> list[tuple[str, float, bool]]:
    """返回描边与相邻底色的对比度，按 3:1 判定。"""
    结果 = []
    for 说明, 底色 in 描边检查项:
        比值 = 对比度(方案["描边"], 方案[底色])
        结果.append((说明, 比值, 比值 >= 最小描边对比度))
    return 结果


for _方案 in 方案表:
    _方案["描边"] = 描边色(_方案)


def 生成样式(方案: dict[str, str]) -> str:
    """按方案生成整套 QSS。悬停、按下只用方案里已有的颜色：
    浅色底换选中底色，深色底（主按钮、危险按钮）加同底色的文字色描边。"""
    c = dict(方案) | 状态色
    深底主 = c["primary_fg"]
    return f"""
QWidget {{
    font-family: {字体族};
    font-size: 14px;
    color: {c["text"]};
}}
QWidget#页面 {{ background: {c["bg"]}; }}
QFrame#卡片 {{
    background: {c["card"]};
    border: 1px solid {c["selected"]};
    border-radius: 14px;
}}
QFrame#分隔线 {{ background: {c["selected"]}; border: none; max-height: 1px; }}

QLabel#页标题 {{ font-size: 24px; font-weight: 700; padding: 0; }}
QLabel#页说明 {{ color: {c["muted"]}; font-size: 13px; }}
QLabel#卡片标题 {{ font-size: 16px; font-weight: 700; }}
QLabel#分组名 {{ color: {c["muted"]}; font-size: 12px; font-weight: 700; }}
QLabel#正文 {{ font-size: 14px; }}
QLabel#弱提示 {{ color: {c["muted"]}; font-size: 12px; }}
QLabel#指标名 {{ color: {c["muted"]}; font-size: 12px; }}
QLabel#指标值 {{ font-size: 26px; font-weight: 700; }}
QLabel#单位 {{ color: {c["muted"]}; font-size: 12px; }}
QLabel#徽章, QLabel#徽章成功, QLabel#徽章警告, QLabel#徽章错误 {{
    border-radius: 6px; padding: 3px 10px; font-size: 12px; font-weight: 700;
    color: #FFFFFF;
}}
QLabel#徽章 {{ background: {c["primary"]}; color: {c["primary_fg"]}; }}
QLabel#徽章成功 {{ background: {c["成功"]}; }}
QLabel#徽章警告 {{ background: {c["警告"]}; }}
QLabel#徽章错误 {{ background: {c["错误"]}; }}

QPushButton {{
    background: {c["card"]};
    color: {c["text"]};
    border: 1px solid {c["描边"]};
    border-radius: 9px;
    padding: 9px 16px;
    min-height: 20px;
}}
QPushButton:hover {{ background: {c["selected"]}; }}
QPushButton:pressed {{ background: {c["selected"]}; border-color: {c["primary"]}; }}
QPushButton:focus {{ border: 2px solid {c["primary"]}; padding: 8px 15px; }}
QPushButton:disabled {{ color: {c["muted"]}; border-color: {c["selected"]}; }}
QPushButton#主按钮 {{
    background: {c["primary"]}; color: {c["primary_fg"]}; font-weight: 700;
    border: 1px solid {c["primary"]};
}}
QPushButton#主按钮:hover {{ border: 2px solid {深底主}; padding: 8px 15px; }}
QPushButton#主按钮:pressed {{ background: {c["selected"]}; color: {c["text"]}; }}
QPushButton#危险按钮 {{
    background: {c["错误"]}; color: #FFFFFF; font-weight: 700;
    border: 1px solid {c["错误"]};
}}
QPushButton#危险按钮:hover {{ border: 2px solid #FFFFFF; padding: 8px 15px; }}
QPushButton#危险按钮:pressed {{ background: {c["selected"]}; color: {c["text"]}; }}
QPushButton#链接按钮 {{
    background: transparent; border: none; color: {c["primary"]};
    padding: 6px 2px; text-align: left; font-weight: 700;
}}
QPushButton#链接按钮:hover {{ color: {c["text"]}; }}

QPushButton#导航项 {{
    background: transparent; border: none; border-left: 3px solid transparent;
    border-radius: 8px; color: {c["muted"]}; text-align: left;
    padding: 9px 12px; font-weight: 700;
}}
QPushButton#导航项:hover {{ background: {c["selected"]}; color: {c["text"]}; }}
QPushButton#导航项:checked {{
    background: {c["selected"]}; color: {c["text"]}; border-left: 3px solid {c["primary"]};
}}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {c["muted"]}; border: 1px solid transparent;
    border-radius: 8px; padding: 8px 16px; margin-right: 6px; font-weight: 700;
}}
QTabBar::tab:hover {{ background: {c["selected"]}; color: {c["text"]}; }}
QTabBar::tab:selected {{
    background: {c["selected"]}; color: {c["text"]}; border: 1px solid {c["描边"]};
}}

QFrame#侧栏 {{
    background: {c["bg"]}; border: 1px solid {c["selected"]}; border-radius: 12px;
}}

QTableWidget {{
    background: {c["card"]}; alternate-background-color: {c["bg"]};
    border: 1px solid {c["描边"]}; border-radius: 10px;
    gridline-color: {c["selected"]}; selection-background-color: {c["selected"]};
    selection-color: {c["text"]};
}}
QHeaderView::section {{
    background: {c["selected"]}; color: {c["text"]}; border: none;
    border-bottom: 1px solid {c["描边"]}; padding: 7px 10px; font-weight: 700;
}}
QTableCornerButton::section {{ background: {c["selected"]}; border: none; }}

QLineEdit, QComboBox {{
    background: {c["card"]};
    border: 1px solid {c["描边"]};
    border-radius: 9px;
    padding: 9px 12px;
    selection-background-color: {c["selected"]};
    selection-color: {c["text"]};
}}
QLineEdit:focus, QComboBox:focus {{ border: 2px solid {c["primary"]}; padding: 8px 11px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {c["card"]}; color: {c["text"]}; border: 1px solid {c["selected"]};
    selection-background-color: {c["selected"]}; selection-color: {c["text"]};
    outline: none;
}}

QListWidget {{
    background: transparent; border: none; outline: none;
}}
QListWidget::item {{
    background: {c["card"]}; border: 1px solid {c["描边"]}; border-radius: 10px;
    padding: 8px 10px; margin: 0 0 6px 0;
}}
QListWidget::item:hover {{ background: {c["selected"]}; }}
QListWidget::item:selected {{
    background: {c["selected"]}; border: 2px solid {c["primary"]}; color: {c["text"]};
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c["selected"]}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QPlainTextEdit, QTextEdit {{
    background: {c["bg"]}; color: {c["text"]}; border: 1px solid {c["selected"]};
    border-radius: 10px; font-family: "Noto Sans Mono CJK SC", {字体族};
    font-size: 13px; padding: 8px;
}}

QStatusBar {{ background: {c["card"]}; color: {c["muted"]}; font-size: 12px;
    border-top: 1px solid {c["selected"]}; }}
QStatusBar::item {{ border: none; }}
"""


# ---------------------------------------------------------------- 截图标注
色值角色 = (("页面", "bg"), ("卡片", "card"), ("正文", "text"), ("辅助", "muted"),
            ("主按钮", "primary"), ("选中", "selected"), ("描边", "描边"))


def 色值清单(方案: dict[str, str]) -> str:
    """纯文本色值清单：截图旁边存一份，也可以直接粘给 AI。"""
    行 = (f"配色方案：{方案['名']}（{方案['用途']}）",
          f"页面背景 {方案['bg']}",
          f"卡片/面板 {方案['card']}",
          f"正文 {方案['text']}",
          f"辅助文字 {方案['muted']}",
          f"主按钮背景 {方案['primary']}",
          f"主按钮文字 {方案['primary_fg']}",
          f"选中底色 {方案['selected']}",
          f"控件描边 {方案['描边']}",
          f"成功 {状态色['成功']}；警告 {状态色['警告']}；错误 {状态色['错误']}")
    return "\n".join(行) + "\n"


def 加色值条(图: QPixmap, 方案: dict[str, str]) -> QPixmap:
    """在截图底部加一条方案名与色值，让图片本身带着准确颜色。

    色块画得比常规控件大，文字也放大：图被转发、被二次压缩后仍然读得清。
    看图的 AI 拿不到代码时，这条就是它唯一的准确依据。
    """
    高 = 68
    结果 = QPixmap(图.width(), 图.height() + 高)
    结果.fill(QColor(方案["card"]))

    画笔 = QPainter(结果)
    画笔.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    画笔.drawPixmap(0, 0, 图)
    顶 = 图.height()
    画笔.setPen(QPen(QColor(方案["描边"])))
    画笔.drawLine(0, 顶, 图.width(), 顶)

    基线 = 顶 + 高 - 16
    画笔.setFont(QFont(字体名, 13, QFont.Weight.Bold))
    画笔.setPen(QColor(方案["text"]))
    标题 = f"{方案['名']}"
    画笔.drawText(16, 基线, 标题)
    横 = 16 + QFontMetrics(画笔.font()).horizontalAdvance(标题) + 12

    画笔.setFont(QFont(字体名, 11))
    度量 = QFontMetrics(画笔.font())
    for 名称, 键 in 色值角色:
        文字 = f"{名称} {方案[键]}"
        宽 = 度量.horizontalAdvance(文字)
        if 横 + 28 + 宽 > 图.width() - 12:
            break
        画笔.setPen(QPen(QColor(方案["muted"]), 1))
        画笔.setBrush(QColor(方案[键]))
        画笔.drawRoundedRect(QRect(横, 基线 - 19, 22, 22), 5, 5)
        横 += 28
        画笔.setPen(QColor(方案["text"]))
        画笔.drawText(横, 基线, 文字)
        横 += 宽 + 16
    画笔.end()
    return 结果


def 需求包(方案: dict[str, str], 界面名: str = "", 用途: str = "", 布局: str = "") -> str:
    """一键复制给 AI 的完整文字：业务占位 + 约束 + 色值 + 控件 + 骨架 + 整套 QSS。

    目标是让"复制 → 粘贴"成为完整交接：能读文本的 AI 拿到这一段就够，
    不用再看截图，也不用回头问配色。尖括号三处由使用者补业务信息。
    """
    名 = 界面名 or "<填界面名称，例如「相机标定」>"
    用 = 用途 or "<一句话：给谁用、做什么>"
    布 = 布局 or "<几块区域、各放什么；不写就按下方「整页结构」>"
    return f"""# 界面需求（整段交给 AI，先把尖括号里的内容填掉）

## 要做什么
- 界面名称：{名}
- 用途：{用}
- 区域布局：{布}
- 技术栈：Python + PySide6。若改用 Tk / Web，只取下面的色值与布局，QSS 不要照搬。

## 必须遵守
1. 只用下面这一套配色，整行取用，不跨方案混搭，不自造新色；状态色只用给出的三种。
2. 正文与辅助文字对背景对比度 ≥ 4.5:1；控件描边对相邻底色 ≥ 3:1。
3. 不要编造数据：示例里的地址、数值、表头都是占位，真实字段以「要做什么」为准。
4. 如果你能读到本工作区文件：把 ui界面/程序/操作界面/界面主题.py 复制到目标项目的
   程序/操作界面/ 再用，只读复用，不要修改原件；整页用 页面骨架() 起步。
   读不到文件就直接套用文末的完整 QSS。

## 配色与色值
{色值清单(方案)}
## 控件对象名（样式已写好，不要再写内联色值）
卡片 / 卡片标题 / 页标题 / 页说明 / 弱提示 / 指标名 / 指标值 / 单位
主按钮 / 危险按钮 / 链接按钮 / 徽章 / 徽章成功 / 徽章警告 / 徽章错误
侧栏 / 导航项 / 表格（QTableWidget 自动带样式）/ 日志面板（QPlainTextEdit）

## 控件工厂（用主题文件时直接调）
卡片(标题)  按钮(文本, "主按钮")  徽章(文本, "成功")  指标(名称, 值, 单位)
导航栏(项列表, 当前)  表格(表头, 数据)  日志面板(行, 高)
页面骨架(标题, 导航项, 表头, 数据, 日志行)

## 整页结构
左：侧栏 + 导航项（可勾选，选中项带主色左边线）
右：标题行（标题 + 状态徽章）→ 指标行 → 主工作区（表格或表单）→ 底部日志面板

## 完整 QSS（当前方案「{方案['名']}」）
```qss
{生成样式(方案)}
```
"""


# ---------------------------------------------------------------- 控件工厂
def 刷新样式(控件: QWidget) -> None:
    """改了 objectName 之后要让 Qt 重新套一遍样式表。"""
    样式 = 控件.style()
    样式.unpolish(控件)
    样式.polish(控件)


def 卡片(标题: str = "", 父: QWidget | None = None) -> tuple[QFrame, QVBoxLayout]:
    """带标题的卡片，返回（卡片, 内容布局）。标题为空时不占位。"""
    卡 = QFrame(父)
    卡.setObjectName("卡片")
    布局 = QVBoxLayout(卡)
    布局.setContentsMargins(18, 16, 18, 16)
    布局.setSpacing(10)
    if 标题:
        名字 = QLabel(标题)
        名字.setObjectName("卡片标题")
        布局.addWidget(名字)
    return 卡, 布局


def 分隔线(父: QWidget | None = None) -> QFrame:
    线 = QFrame(父)
    线.setObjectName("分隔线")
    线.setFixedHeight(1)
    return 线


def 按钮(文本: str, 样式: str | None = None, 父: QWidget | None = None) -> QPushButton:
    钮 = QPushButton(文本, 父)
    if 样式:
        钮.setObjectName(样式)
    钮.setCursor(Qt.CursorShape.PointingHandCursor)
    return 钮


def 设徽章状态(牌: QLabel, 状态: str = "") -> None:
    """切换徽章状态，走 QSS 的对象名，不写内联色值。状态为空时用主按钮配色。"""
    牌.setObjectName(f"徽章{状态}" if 状态 else "徽章")
    刷新样式(牌)


def 徽章(文本: str, 状态: str = "") -> QLabel:
    """状态徽章：底色取状态色，白字；不传状态时用主按钮配色。

    尺寸策略固定成「按内容」，否则放进有剩余空间的布局里会被拉成竖长条。
    """
    牌 = QLabel(文本)
    牌.setAlignment(Qt.AlignmentFlag.AlignCenter)
    牌.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    设徽章状态(牌, 状态)
    return 牌


def 指标(名称: str, 值: str, 单位: str = "") -> QWidget:
    块 = QWidget()
    块.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    列 = QVBoxLayout(块)
    列.setContentsMargins(0, 0, 0, 0)
    列.setSpacing(2)
    名 = QLabel(名称)
    名.setObjectName("指标名")
    数 = QLabel(值)
    数.setObjectName("指标值")
    列.addWidget(名)
    行 = QHBoxLayout()
    行.setContentsMargins(0, 0, 0, 0)
    行.setSpacing(4)
    行.addWidget(数)
    if 单位:
        单 = QLabel(单位)
        单.setObjectName("单位")
        行.addWidget(单, 0, Qt.AlignmentFlag.AlignBottom)
    行.addStretch(1)
    列.addLayout(行)
    return 块


def 导航栏(项: list[str], 当前: int = 0) -> QWidget:
    """左侧导航：可勾选按钮，选中项带主色左边线。"""
    栏 = QWidget()
    列 = QVBoxLayout(栏)
    列.setContentsMargins(0, 0, 0, 0)
    列.setSpacing(4)
    for 序号, 文本 in enumerate(项):
        钮 = QPushButton(文本)
        钮.setObjectName("导航项")
        钮.setCheckable(True)
        钮.setChecked(序号 == 当前)
        钮.setCursor(Qt.CursorShape.PointingHandCursor)
        列.addWidget(钮)
    列.addStretch(1)
    return 栏


def 表格(表头: list[str], 数据: list[list[str]]) -> QTableWidget:
    """只读表格：整行选中、隔行浅底、列宽铺满；表头样式跟着方案走。"""
    表 = QTableWidget(len(数据), len(表头))
    表.setHorizontalHeaderLabels(表头)
    表.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    表.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    表.setAlternatingRowColors(True)
    表.setShowGrid(False)
    表.verticalHeader().setVisible(False)
    表.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    for 行号, 各行 in enumerate(数据):
        for 列号, 值 in enumerate(各行):
            项 = QTableWidgetItem(str(值))
            项.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            表.setItem(行号, 列号, 项)
    return 表


def 日志面板(行: tuple[str, ...] = (), 高: int = 96) -> QPlainTextEdit:
    """只读日志区：等宽字体、自动滚到底，最多保留 2000 行。"""
    框 = QPlainTextEdit()
    框.setReadOnly(True)
    框.setFixedHeight(高)
    框.setMaximumBlockCount(2000)
    if 行:
        框.setPlainText("\n".join(行))
        滚动 = 框.verticalScrollBar()
        滚动.setValue(滚动.maximum())
    return 框


def 页面骨架(标题: str, 导航项: list[str], 表头: list[str], 数据: list[list[str]],
             日志行: tuple[str, ...] = (), 当前: int = 0) -> QWidget:
    """一整页的骨架：左导航 + 标题/指标 + 表格 + 底部日志。

    新界面直接从这里起步，再往右列加自己要的控件；不要的块删掉即可。
    """
    骨架 = QWidget()
    外框 = QHBoxLayout(骨架)
    外框.setContentsMargins(0, 0, 0, 0)
    外框.setSpacing(12)

    侧栏 = QFrame()
    侧栏.setObjectName("侧栏")
    侧栏.setFixedWidth(152)
    侧列 = QVBoxLayout(侧栏)
    侧列.setContentsMargins(10, 12, 10, 12)
    侧列.setSpacing(6)
    品牌 = QLabel("控制台")
    品牌.setObjectName("卡片标题")
    侧列.addWidget(品牌)
    侧列.addWidget(导航栏(导航项, 当前))
    外框.addWidget(侧栏)

    右列 = QVBoxLayout()
    右列.setSpacing(10)
    标题行 = QHBoxLayout()
    名 = QLabel(标题)
    名.setObjectName("卡片标题")
    标题行.addWidget(名)
    标题行.addStretch(1)
    标题行.addWidget(徽章("在线", "成功"), 0, Qt.AlignmentFlag.AlignVCenter)
    右列.addLayout(标题行)

    指标行 = QHBoxLayout()
    指标行.setSpacing(26)
    for 名称, 值, 单位 in (("帧率", "30.0", "Hz"), ("丢帧", "0", "帧"),
                          ("延迟", "18", "ms")):
        指标行.addWidget(指标(名称, 值, 单位))
    指标行.addStretch(1)
    右列.addLayout(指标行)

    右列.addWidget(表格(表头, 数据), 1)
    if 日志行:
        右列.addWidget(日志面板(日志行))
    外框.addLayout(右列, 1)
    return 骨架
