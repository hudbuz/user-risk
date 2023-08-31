from .quiet_error import QuietError


class QwakLoginException(QuietError):
    def __init__(
        self, message="Failed to login to Qwak. Please check your credentials"
    ):
        self.message = message
