import time


def create_qwak_timer(name):
    try:
        start_time = time.time()
        yield
    finally:
        elapsed_secs = time.time() - start_time
        print(f"elapsed time: {elapsed_secs} seconds")
