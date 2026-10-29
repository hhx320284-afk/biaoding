"""启动相机标定桌面界面；复用工作区的 Qt 启动方式。"""
import argparse
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / '程序'), str(PROJECT / '程序/操作界面')]


def main():
    parser = argparse.ArgumentParser(description='相机标定桌面界面')
    parser.add_argument('--日志目录', type=Path, default=PROJECT / '日志')
    args = parser.parse_args()
    log_path = args.日志目录 / '运行.log'
    logger = logging.getLogger('相机标定')
    app = None
    server = None
    instance_lock = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(log_path, maxBytes=1024 * 1024, backupCount=2, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        from PySide6.QtCore import QLibraryInfo, QLocale, QLockFile, QStandardPaths, QTranslator
        from PySide6.QtNetwork import QLocalServer, QLocalSocket
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication(['相机标定'])
        app.setApplicationName('相机标定')
        app.setOrganizationName('VLA')
        QLocale.setDefault(QLocale('zh_CN'))
        translator = QTranslator(app)
        if translator.load('qtbase_zh_CN', QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
            app.installTranslator(translator)
        name = 'camera-calibration-' + hashlib.sha256(str(PROJECT).encode()).hexdigest()[:16]
        runtime = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.RuntimeLocation)
        instance_lock = QLockFile(str(Path(runtime) / (name + '.lock')))
        instance_lock.setStaleLockTime(0)
        if not instance_lock.tryLock(150):
            client = QLocalSocket()
            client.connectToServer(name)
            if not client.waitForConnected(1500):
                raise RuntimeError('已有标定进程尚未就绪，请稍后重新点击启动')
            client.write(b'activate')
            client.waitForBytesWritten(500)
            client.disconnectFromServer()
            logger.info('请求激活已有标定窗口')
            return 0
        server = QLocalServer(app)
        QLocalServer.removeServer(name)
        if not server.listen(name):
            raise RuntimeError('无法创建本地窗口通信通道：' + server.errorString())
        from 标定界面 import MainWindow
        window = MainWindow(PROJECT, logger, log_path)

        def activate_window():
            while server.hasPendingConnections():
                connection = server.nextPendingConnection()
                connection.disconnectFromServer()
                connection.deleteLater()
            if window.isMinimized():
                window.showNormal()
            else:
                window.show()
            window.raise_()
            window.activateWindow()

        def handle_error(kind, value, traceback):
            logger.error('界面操作异常', exc_info=(kind, value, traceback))
            QMessageBox.warning(window, '操作失败', f'{value}\n\n日志位置：{log_path}')

        sys.excepthook = handle_error
        server.newConnection.connect(activate_window)
        window.show()
        logger.info('相机标定界面启动')
        result = app.exec()
        logger.info('相机标定界面停止')
        return result
    except Exception as error:
        logger.exception('启动失败')
        message = f'相机标定启动失败：{error}\n日志位置：{log_path}'
        print(message, file=sys.stderr)
        if app is not None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, '启动失败', message)
        return 1
    finally:
        if server is not None:
            server.close()
        if instance_lock is not None and instance_lock.isLocked():
            instance_lock.unlock()
        logging.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
