from typing import Any, Sequence

from .base_input_adapter import BaseInputAdapter


class ImageInputAdapter(BaseInputAdapter):
    def extract_user_func_arg(self, data: Any) -> Sequence[Any]:
        pass
