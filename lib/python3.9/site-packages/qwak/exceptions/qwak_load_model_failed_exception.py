from .quiet_error import QuietError


class QwakLoadModelFailedException(QuietError):
    """
    Raises when `load_model()` raises an exception
    """

    def __init__(self, exception):
        self.message = f"Failed to load model from load_model(), Error: {exception}"
