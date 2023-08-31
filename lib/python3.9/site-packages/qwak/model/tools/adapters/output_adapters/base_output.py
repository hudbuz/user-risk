from typing import Any


class BaseOutputAdapter:
    @property
    def config(self):
        return dict()

    def pack_user_func_return_value(
        self,
        return_result: Any,
    ) -> Any:
        """
        Pack the return value of user defined API function into InferenceResults
        """
        raise NotImplementedError()
