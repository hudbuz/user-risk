from typing import Dict, Optional, Union

try:
    from bentoml.exceptions import InferenceException
except ImportError:
    from .qwak_mock_http_exception import MockHttpException as InferenceException


class QwakHTTPException(InferenceException):
    def __init__(
        self,
        status_code: int,
        message: Union[str, Dict],
        exception_class_name: Optional[str] = None,
    ):
        super().__init__(status_code, message)
        self.exception_class_name = (
            exception_class_name if exception_class_name else type(self).__name__
        )
