from abc import ABC, abstractmethod
from typing import Any


class BaseOutputAdapter(ABC):
    @staticmethod
    def http_content_type():
        return {"Content-Type": "text/plain; charset=utf-8"}

    @staticmethod
    @abstractmethod
    def pack_user_func_return_value(return_result: Any) -> Any:
        pass
