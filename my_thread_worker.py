import typing
from queue import Queue

from PyQt5.QtCore import QRunnable, QObject, pyqtSignal


class WorkerSignals(QObject):

    finished = pyqtSignal()

    def __init__(self, parent: typing.Optional[QObject] = None) -> None:
        QObject.__init__(self, parent)


class MyThreadWorker(QRunnable):
    def __init__(self, task, args, kwargs) -> None:
        QRunnable.__init__(self)
        self.signals = WorkerSignals()
        self.task = task
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        value = self.task(*self.args)
        if 'return_queue' in self.kwargs:
            q: Queue = self.kwargs['return_queue']
            q.put_nowait(value)
        self.signals.finished.emit()
