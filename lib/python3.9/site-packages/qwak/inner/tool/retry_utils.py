import time


def retry(
    f,
    args=None,
    kwargs=None,
    exceptions=Exception,
    attempts=-1,
    delay=0,
):
    _attempts = attempts
    args = args if args else list()
    kwargs = kwargs if kwargs else dict()
    while _attempts:
        try:
            return f(*args, **kwargs)
        except exceptions:
            _attempts -= 1
            if not _attempts:
                raise

            time.sleep(delay)
