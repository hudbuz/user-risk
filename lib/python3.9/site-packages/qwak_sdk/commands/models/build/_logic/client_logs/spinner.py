from contextlib import contextmanager
from typing import Generator, Optional

from yaspin import yaspin
from yaspin.core import Yaspin


@contextmanager
def spinner(text: Optional[str], show: bool) -> Generator[Yaspin, None, None]:
    if show:
        with yaspin(text=text, color="blue", timer=True).bold as sp:
            yield sp
    else:
        yield
