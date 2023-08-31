import base64
import json
import traceback

from qwak.exceptions import QwakHTTPException
from qwak.model.tools.adapters.encoders import TF_B64_KEY

from .string_input import StringInput


def b64_hook(o):
    if isinstance(o, dict) and TF_B64_KEY in o:
        return base64.b64decode(o[TF_B64_KEY])
    return o


class TfTensorInput(StringInput):
    def extract_user_func_args(self, data: str) -> "tensorflow.Tensor":  # noqa F401
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "tensorflow package is required to use TfTensorInput adapter"
            )

        try:
            parsed_json = json.loads(data, object_hook=b64_hook)
            if parsed_json.get("instances") is None:
                raise QwakHTTPException(
                    status_code=400,
                    message="input format is not implemented",
                )
            else:
                instances = parsed_json.get("instances")
                return tf.constant(instances)
        except json.JSONDecodeError:
            raise QwakHTTPException(status_code=400, message="Not a valid JSON format")
        except Exception:
            err = traceback.format_exc()
            raise QwakHTTPException(
                status_code=500, message=f"Internal Server Error: {err}"
            )
