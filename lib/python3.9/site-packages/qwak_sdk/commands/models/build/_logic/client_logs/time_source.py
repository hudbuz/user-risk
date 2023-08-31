from abc import ABC, abstractmethod
from time import time
from typing import List


class TimeSource(ABC):
    @abstractmethod
    def current_epoch_time(self) -> int:
        pass


class Stopwatch:
    def __init__(self, time_source: TimeSource):
        self.time_source = time_source
        self.start_time = self.time_source.current_epoch_time()

    def elapsed_time_in_seconds(self) -> int:
        end_time = self.time_source.current_epoch_time()
        return end_time - self.start_time


class SystemClockTimeSource(TimeSource):
    def current_epoch_time(self) -> int:
        return int(time())


class MockClockTimeSource(TimeSource):
    def __init__(self, time_values=None):
        self.time_values = []
        if time_values:
            self.time_values = list(time_values)

    def current_epoch_time(self) -> int:
        return self.time_values.pop(0)

    def set_epoch_time_values(self, values: List[int]) -> None:
        self.time_values = values
