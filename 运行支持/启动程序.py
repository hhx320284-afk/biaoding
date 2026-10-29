"""TCP 标定入口：导入路径、轮转日志与单实例窗口唤起。"""

from __future__ import annotations

import errno
import ctypes
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import sys


class 单实例:
    """使用本机抽象 Unix socket；不创建锁文件或保存用户设置。"""

    def __init__(self, project_dir: Path):
        identity = hashlib.sha256(str(project_dir).encode("utf-8")).hexdigest()[:24]
        self.address = f"\0tcp-calibration-{os.getuid()}-{identity}"
        self.server = None
        self.root = None
        self.logger = None

    def acquire(self) -> bool:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.address)
        except OSError as error:
            server.close()
            if error.errno != errno.EADDRINUSE:
                raise
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                client.connect(self.address)
            return False
        server.listen(8)
        server.setblocking(False)
        self.server = server
        return True

    def attach(self, root) -> None:
        self.root = root
        root.report_callback_exception = self._callback_error
        root.after(150, self._poll)

    def _callback_error(self, kind, value, traceback) -> None:
        self.logger.error("界面操作异常", exc_info=(kind, value, traceback))
        from tkinter import messagebox
        messagebox.showerror(
            "操作未完成", f"{value}\n\n详情请查看“打开日志”。", parent=self.root
        )

    def _poll(self) -> None:
        if self.server is None:
            return
        activate = False
        while True:
            try:
                connection, _ = self.server.accept()
                connection.close()
                activate = True
            except BlockingIOError:
                break
        if activate:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.logger.info("重复启动请求：唤起已有窗口")
        self.root.after(150, self._poll)

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
            self.server = None


def 报告启动错误(error: Exception, log_path: Path) -> None:
    message = f"TCP 标定启动失败：{error}\n\n日志位置：{log_path}"
    print(message, file=sys.stderr)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk._default_root
        created = root is None
        if created:
            root = tk.Tk()
            root.withdraw()
        messagebox.showerror("TCP 标定启动失败", message, parent=root)
        if created:
            root.destroy()
    except Exception:
        pass


def main() -> int:
    project_dir = Path(__file__).resolve().parents[2]
    log_path = project_dir / "日志" / "运行.log"
    instance = 单实例(project_dir)
    logger = None
    try:
        if not instance.acquire():
            return 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("TCP标定")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = RotatingFileHandler(
            log_path, maxBytes=1024 * 1024, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.instance_controller = instance
        instance.logger = logger
        logger.info("启动 PiPER / UR TCP 连续采集标定")
        for category in ("核心算法", "操作界面"):
            sys.path.insert(0, str(project_dir / "程序" / category))
        # 仅本进程加载支持抗锯齿字体的 Tk；不改系统或 Python 环境。
        font_support = Path(__file__).resolve().parent / "Tk字体支持"
        ctypes.CDLL(str(font_support / "lib" / "libtk8.6.so"), mode=ctypes.RTLD_GLOBAL)
        os.environ["TK_LIBRARY"] = str(font_support / "tk8.6")
        from 标定界面 import main as run_interface
        run_interface(project_dir, logger)
        logger.info("TCP 标定正常退出")
        return 0
    except Exception as error:
        if logger is not None:
            logger.exception("TCP 标定启动或运行失败")
        报告启动错误(error, log_path)
        return 1
    finally:
        instance.close()
        if logger is not None:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
