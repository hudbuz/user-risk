from .base_input import BaseInputAdapter


class StringInput(BaseInputAdapter):
    def extract_user_func_args(self, data) -> str:
        return data
