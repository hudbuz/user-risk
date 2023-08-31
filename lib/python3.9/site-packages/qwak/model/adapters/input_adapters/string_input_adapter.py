from typing import Any, Sequence

from .base_input_adapter import BaseInputAdapter


class StringInputAdapter(BaseInputAdapter):
    def extract_user_func_arg(self, data: str) -> Sequence[Any]:
        pass
