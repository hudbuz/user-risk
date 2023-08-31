import json

TF_B64_KEY = "b64"


class B64JsonEncoder(json.JSONEncoder):
    def default(self, o):
        import base64

        if isinstance(o, bytes):
            try:
                return o.decode("utf-8")
            except UnicodeDecodeError:
                return {TF_B64_KEY: base64.b64encode(o).decode("utf-8")}

        try:
            return super(B64JsonEncoder, self).default(o)
        except (TypeError, OverflowError):
            return {"unknown_obj": str(o)}


class NumpyJsonEncoder(B64JsonEncoder):
    def default(self, o):
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()

        if isinstance(o, np.ndarray):
            return o.tolist()

        return super(NumpyJsonEncoder, self).default(o)


class TfTensorJsonEncoder(NumpyJsonEncoder):
    def default(self, o):
        import tensorflow as tf

        # Tensor -> ndarray or object
        if isinstance(o, tf.Tensor):
            if tf.__version__.startswith("1."):
                with tf.compat.v1.Session():
                    return o.numpy()
            else:
                return o.numpy()
        return super(TfTensorJsonEncoder, self).default(o)
