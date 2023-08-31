from abc import ABC, abstractmethod
from typing import Any, Sequence


class BaseInputAdapter(ABC):
    @abstractmethod
    def extract_user_func_arg(self, data: str) -> Sequence[Any]:
        pass
