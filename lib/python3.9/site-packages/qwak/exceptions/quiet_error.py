import sys


class QuietError(Exception):
    # All who inherit me shall not traceback, but be spoken of cleanly
    def __init__(self, message):
        self.message = None

    def __str__(self):
        return f"\033[91m{self.message}\033[0m"


def quiet_hook(kind, message, traceback):
    if issubclass(kind, QuietError):
        print("{0}".format(message))  # Only print Error Type and Message
    else:
        sys.__excepthook__(
            kind, message, traceback
        )  # Print Error Type, Message and Traceback


sys.excepthook = quiet_hook
