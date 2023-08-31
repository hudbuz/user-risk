from typing import Any

from .base_input import BaseInputAdapter


class FileInput(BaseInputAdapter):
    def extract_user_func_args(self, data: Any) -> Any:
        return data
