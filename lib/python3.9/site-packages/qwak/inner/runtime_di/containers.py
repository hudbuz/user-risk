from dependency_injector import containers, providers
from qwak.model.decorators.impl.api_implementation import create_decorator_function
from qwak.model.decorators.impl.timer_implementation import create_qwak_timer


class QwakRuntimeContainer(containers.DeclarativeContainer):
    """
    Qwak Core's Runtime Dependency Injection Container
    """

    api_decorator_function_creator = providers.Object(create_decorator_function)
    timer_function_creator = providers.Object(create_qwak_timer)
