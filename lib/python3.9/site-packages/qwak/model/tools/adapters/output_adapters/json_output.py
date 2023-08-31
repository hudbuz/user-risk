import json
from typing import Any

from qwak.exceptions import QwakHTTPException
from qwak.model.tools.adapters.encoders import NumpyJsonEncoder

from .base_output import BaseOutputAdapter


class JsonOutput(BaseOutputAdapter):
    def pack_user_func_return_value(self, return_result: Any) -> str:
        try:
            return json.dumps(return_result, cls=NumpyJsonEncoder, ensure_ascii=False)
        except AssertionError as e:
            QwakHTTPException(400, message=str(e))
        except Exception as e:
            QwakHTTPException(500, message=str(e))
