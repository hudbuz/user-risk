from qwak.inner.runtime_di import QwakRuntimeContainer

from .run_model_locally import run_local


def wire_runtime():
    container = QwakRuntimeContainer()
    from qwak.model import decorators

    container.wire(
        packages=[
            decorators,
        ]
    )
    return container


wire_runtime()
__all__ = ["run_local"]
