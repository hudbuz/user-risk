from .quiet_error import QuietError


class QwakException(QuietError):
    def __init__(self, message):
        self.message = message
