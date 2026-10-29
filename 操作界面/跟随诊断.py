"""选孔跟随的自动过程记录；复用已有只读 CAN 诊断与失败重试导出。"""

import copy
import queue
import threading
import time

from 控制诊断 import AutomaticControlDiagnostics


class FollowDiagnosticsMixin:
    """面板在主线程轮询；发送、视觉及执行事件可从工作线程写入。"""

    def _init_follow_diagnostics(self):
        self.follow_automatic = AutomaticControlDiagnostics(
            self.app.project_dir / '日志' / '运动诊断')
        self.follow_diagnostic_path = None
        self.follow_save_error = None
        self._follow_exporting = False
        self._follow_export_job = None
        self._follow_finish_reason = None
        self._follow_retry_after = 0.0
        self._follow_export_events = queue.Queue()
        self._follow_recording_context = {}

    def _begin_follow_recording(self, context):
        """在首次准备/运动前调用；context 只含位姿、参数和质量摘要，不含图像。"""
        if self._follow_exporting or self._follow_finish_reason:
            raise ValueError('上一轮跟随过程记录正在保存，请稍候。')
        if not self.follow_automatic.active:
            self._follow_recording_context = copy.deepcopy(context)
        try:
            self.follow_automatic.begin(self.app.robot_reader, context)
        except Exception:
            # begin 可能已经绑定 reader 或留下启动失败事件，仍须归档。
            self._end_follow_recording('跟随诊断启动失败')
            raise
        self.follow_save_error = None
        self.follow_diagnostic_path = None

    def _follow_diagnostic_event(self, label, data=None, trigger=False):
        return self.follow_automatic.event(label, data, trigger=trigger)

    def _end_follow_recording(self, reason):
        """只请求结束；待运动线程退出并处理最终事件后才停止 CAN 记录。"""
        if self.follow_automatic.active and self._follow_finish_reason is None:
            self._follow_finish_reason = str(reason)
            self._follow_diagnostic_event('跟随流程结束请求', {'reason': str(reason)})

    def _follow_end_context(self):
        reader = self.app.robot_reader
        try:
            context = (reader.diagnostic_context()
                       if callable(getattr(reader, 'diagnostic_context', None)) else None)
        except Exception as exc:
            context = {'read_error': str(exc)}
        return {'task': 'selected_hole_follow',
                'settings': self._follow_recording_context,
                'feedback': copy.deepcopy(self.app.live_snapshot),
                'reader_context': context}

    def _poll_follow_diagnostics(self):
        while True:
            try:
                job, path, error = self._follow_export_events.get_nowait()
            except queue.Empty:
                break
            self._follow_exporting = False
            if error:
                self.follow_save_error = error
                self._follow_retry_after = time.monotonic() + 5
                self.app.logger.error('跟随过程自动保存失败：%s', error)
            else:
                self.follow_diagnostic_path = path
                self.follow_save_error = None
                self._follow_retry_after = 0.0
                self.app.logger.info('跟随过程已自动保存：%s', path)
        state = self.follow_automatic.status()
        if self.follow_automatic.active and not state['running'] and not self._follow_finish_reason:
            reason = state['error'] or state.get('automatic_error') or '跟随诊断接收已中断。'
            self._end_follow_recording(reason)
            if self.busy:
                self.stop_motion(reason)
        # busy 必须包含尚未退出的跟随/规划线程；事件处理在此方法之前完成。
        # 因此最后一次发令、停止反馈及线程退出原因仍会进入本轮记录。
        if (self._follow_exporting or self.busy or not self.events.empty()
                or time.monotonic() < self._follow_retry_after):
            return
        if self._follow_finish_reason:
            try:
                self._follow_export_job = self.follow_automatic.finish(
                    self._follow_finish_reason, self._follow_end_context())
            except Exception as exc:
                self.follow_save_error = str(exc)
                self._follow_retry_after = time.monotonic() + 5
                self.app.logger.exception('结束跟随过程诊断失败')
                return
            self._follow_finish_reason = None
        job = self._follow_export_job
        if job is None or job.saved:
            return
        self._follow_exporting = True

        def export():
            try:
                path = job.export()
                self._follow_export_events.put((job, path, None))
            except Exception as exc:
                self._follow_export_events.put((job, None, str(exc)))

        threading.Thread(target=export, name='跟随过程自动保存', daemon=True).start()

    def _follow_diagnostic_text(self):
        if self.follow_save_error:
            return '过程记录保存失败：' + self.follow_save_error + '；内存记录保留，正在自动重试。'
        if self._follow_exporting or self._follow_finish_reason:
            return '正在自动保存本轮跟随过程记录…'
        if self.follow_automatic.active:
            state = self.follow_automatic.status()
            return (f"自动记录中 · {state['retained_frames']:,} 帧 · "
                    f"{state['retained_events']:,} 个事件")
        if self.follow_diagnostic_path:
            issues = self._follow_export_job.context.get('capture_issues', [])
            return '过程记录已自动保存到 日志/运动诊断。' + ('记录存在缺口，详见日志。' if issues else '')
        return '选孔后自动记录，停止或异常时自动保存。'

    @property
    def diagnostic_unsaved(self):
        return self.follow_automatic.active or bool(
            self._follow_export_job and not self._follow_export_job.saved)

    def prepare_close(self):
        self._end_follow_recording('关闭窗口')
        self._poll_follow_diagnostics()
        return not (self.busy or self._follow_finish_reason or self._follow_exporting
                    or self.diagnostic_unsaved)
