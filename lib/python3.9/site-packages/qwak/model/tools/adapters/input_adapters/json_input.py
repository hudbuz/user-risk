import json
import traceback
from typing import Dict, List, Union

from qwak.exceptions import QwakHTTPException

from .string_input import StringInput


class JsonInput(StringInput):
    def extract_user_func_args(
        self, data: str
    ) -> Union[bool, None, Dict, List, int, float, str]:
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            raise QwakHTTPException(400, message="Not a valid JSON format")
        except Exception:
            err = traceback.format_exc()
            raise QwakHTTPException(500, message=f"Internal Server Error: {err}")
