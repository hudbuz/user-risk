import json

from qwak.exceptions import QwakHTTPException
from qwak.model.tools.adapters.encoders import TfTensorJsonEncoder

from .json_output import JsonOutput


def tf_to_numpy(tensor):
    """
    Tensor -> ndarray
    List[Tensor] -> tuple[ndarray]
    """
    try:
        import numpy as np  # noqa F401
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "Tensorflow and numpy packages are required to use TfTensorOutput"
        )

    if isinstance(tensor, (list, tuple)):
        return tuple(tf_to_numpy(t) for t in tensor)

    if tf.__version__.startswith("1."):
        with tf.compat.v1.Session():
            return tensor.numpy()
    else:
        return tensor.numpy()


class TfTensorOutput(JsonOutput):
    def pack_user_func_return_value(
        self,
        return_result,
    ) -> str:
        result = tf_to_numpy(return_result)
        try:
            return json.dumps(result, cls=TfTensorJsonEncoder)
        except Exception as e:
            raise QwakHTTPException(message=str(e), status_code=500)
