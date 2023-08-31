class QwakGeneralBuildException(Exception):
    def __init__(self, message: str, src_exception: Exception = None):
        self._message = message
        self._exception_msg = str(src_exception) if src_exception else ""

    @property
    def message(self) -> str:
        msg = f"""Message: {self._message}
Exception message: {self._exception_msg}"""
        return msg

    def __str__(self):
        return self.message
