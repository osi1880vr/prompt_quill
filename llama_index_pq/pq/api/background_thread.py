import logging
import queue
import threading
import time
from queue import Queue
from abc import abstractmethod, ABC
from typing import Dict

TASKS_QUEUE = Queue()


class BackgroundThread(threading.Thread, ABC):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    @abstractmethod
    def startup(self) -> None:
        """
        Method that is called before the thread starts.
        Initialize all necessary resources here.
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self) -> None:
        """
        Method that is called shortly after stop() method was called.
        Use it to clean up all resources before thread stops.
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def handle(self) -> None:
        """
        Method that should contain business logic of the thread.
        Will be executed in the loop until stop() method is called.
        Must not block for a long time.
        :return: None
        """
        raise NotImplementedError()

    def run(self) -> None:
        """
        This method will be executed in a separate thread
        when start() method is called.
        :return: None
        """
        self.startup()
        while not self._stopped():
            self.handle()
        self.shutdown()


class NotificationThread(BackgroundThread):
    def startup(self) -> None:
        logging.info('NotificationThread started')

    def shutdown(self) -> None:
        logging.info('NotificationThread stopped')

    def handle(self) -> None:
        try:
            task = TASKS_QUEUE.get(block=False)
            # send_notification(task)
            logging.info(f'Notification for {task} was sent.')
        except queue.Empty:
            time.sleep(1)


class BackgroundThreadFactory:
    @staticmethod
    def create(thread_type: str) -> BackgroundThread:
        if thread_type == 'notification':
            return NotificationThread()

        # if thread_type == 'some_other_type':
        #     return SomeOtherThread()

        raise NotImplementedError('Specified thread type is not implemented.')