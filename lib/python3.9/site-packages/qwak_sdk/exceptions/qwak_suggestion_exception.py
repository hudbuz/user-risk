from typing import Optional


class QwakSuggestionException(Exception):
    def __init__(
        self,
        message: str,
        src_exception: Optional[Exception] = None,
        suggestion: Optional[str] = "",
    ):
        self._message = message
        self._src_exception = str(src_exception) if src_exception else ""
        self._suggestion = suggestion

    @property
    def message(self) -> str:
        if self._src_exception:
            msg = f"""Message: {self._message}.
Exception message: {self._src_exception}.
Recommendation: {self._suggestion}."""
        else:
            msg = f"""Message: {self._message}.
Recommendation: {self._suggestion}."""
        return msg

    def __str__(self):
        return self.message
