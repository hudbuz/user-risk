from typing import Any, Type

from .base_output import BaseOutputAdapter


def detect_suitable_adapter(result) -> Type[BaseOutputAdapter]:
    try:
        import pandas as pd

        if isinstance(result, (pd.DataFrame, pd.Series)):
            from .dataframe_output import DataFrameOutput

            return DataFrameOutput
    except ImportError:
        pass

    try:
        import tensorflow as tf

        if isinstance(result, tf.Tensor):
            from .tensorflow_tensor_output import TfTensorOutput

            return TfTensorOutput
    except ImportError:
        pass

    from .json_output import JsonOutput

    return JsonOutput


class DefaultOutput(BaseOutputAdapter):
    """
    Detect suitable output adapter automatically
    OutputAdapter converts returns of user defined API function into specific output,
    such as HTTP response, command line stdout or AWS Lambda event object.

    Args:
        cors (str): DEPRECATED. Moved to the configuration file.
            The value of the Access-Control-Allow-Origin header set in the
            HTTP/AWS Lambda response object. If set to None, the header will not be set.
            Default is None.
    """

    def __init__(self, **kwargs):
        super(DefaultOutput, self).__init__(**kwargs)
        self.actual_adapter = None
        from .json_output import JsonOutput

        self.backup_adapter = JsonOutput()

    def pack_user_func_return_value(self, return_result: Any) -> Any:
        if self.actual_adapter is None:
            self.actual_adapter = detect_suitable_adapter(return_result)()
        if self.actual_adapter:
            return self.actual_adapter.pack_user_func_return_value(return_result)

        raise NotImplementedError()
