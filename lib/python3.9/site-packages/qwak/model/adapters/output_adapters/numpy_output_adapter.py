import io

from qwak.exceptions import QwakException

try:
    import numpy as np
except ImportError:
    pass
    # Numpy is supported either by supplying it as a dependency (or sub-dependency)
    # in your Python project, or during the runtime

from .base_output_adapter import BaseOutputAdapter


class NumpyOutputAdapter(BaseOutputAdapter):
    @staticmethod
    def http_content_type():
        return {"Content-Type": "application/x-npy"}

    @staticmethod
    def pack_user_func_return_value(return_result) -> bytes:
        if not isinstance(return_result, np.ndarray):
            raise QwakException(
                "XNpyOutputAdapter can only work with numpy.ndarray type"
            )
        try:
            results_buffer = io.BytesIO()
            # Disallow pickle: Changed in version 1.16.3: in response to CVE-2019-6446.
            np.save(results_buffer, return_result, allow_pickle=False)
            return results_buffer.getvalue()
        except Exception as e:
            raise ValueError(f"Failed to dump numpy array. Error is {e}")
