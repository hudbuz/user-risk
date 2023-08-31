import contextlib

from dependency_injector.wiring import Provide, inject
from qwak.inner.runtime_di.containers import QwakRuntimeContainer


@inject
@contextlib.contextmanager
def qwak_timer(
    name, create_timer_function=Provide[QwakRuntimeContainer.timer_function_creator]
):
    """
    The Qwak timer is a utility for measuring the runtime of ML models, specifically for visibility and latency evaluation.
    It provides a convenient way to measure the time it takes for models to perform predictions
    and to provide a breakdown of the different parts in the production process
    Args:
        name (str): The Qwak timer name. Users may define multiple impl with different names
    """
    return create_timer_function(name)
