from typing import Dict, Optional, Union


class QwakLoginException(Exception):
    """
    Raise when there is a login error
    """


class InferenceException(Exception):
    """Raised to return a custom HTTP response from the predict function

    Args:
        status_code (int): HTTP status code - supported statuses: 4xx, 5xx
            If status_code is not in the supported range, it will be set to 500,
            and an error message will be added to the response body.
        message (str or Dict): Error message to be returned as the response body
            If a string, it will be converted to format: {"message": message}
            If a dictionary, it will be returned without any changes

    """

    def __init__(self, status_code: int, message: Union[str, Dict]):
        super().__init__(str(message))
        self.status_code = status_code

        if isinstance(message, str):
            message = {"message": message}
        self.message = message

        if not (400 <= self.status_code < 600):
            self.status_code = 500
            self.message[
                "qwak_backend_message"
            ] = f"Invalid status code. Given value: {status_code}. Supported: 4xx, 5xx"


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
