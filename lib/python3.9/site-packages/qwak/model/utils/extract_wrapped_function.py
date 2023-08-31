from types import FunctionType
from typing import Any, Callable


def extract_wrapped(decorated: Callable[..., Any]):
    if decorated.__closure__:
        closure = (c.cell_contents for c in decorated.__closure__)
        return next((c for c in closure if isinstance(c, FunctionType)), decorated)
    return decorated
