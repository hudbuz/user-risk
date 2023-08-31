from .qwak_exception import QwakException


class QwakNotFoundException(QwakException):
    def __init__(self, message):
        self.message = message
