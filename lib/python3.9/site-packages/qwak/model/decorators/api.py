from typing import Callable

from dependency_injector.wiring import Provide, inject
from qwak.inner.runtime_di.containers import QwakRuntimeContainer
from qwak.model.adapters.input_adapters.base_input_adapter import BaseInputAdapter
from qwak.model.adapters.input_adapters.dataframe_input_adapter import (
    DataFrameInputAdapter,
)
from qwak.model.adapters.output_adapters.base_output_adapter import BaseOutputAdapter
from qwak.model.adapters.output_adapters.dataframe_output_adapter import (
    DataFrameOutputAdapter,
)


@inject
def api_decorator(
    analytics: bool = True,
    analytics_sample_ratio: float = 1.0,
    feature_extraction: bool = False,
    input_adapter: BaseInputAdapter = DataFrameInputAdapter(),
    output_adapter: BaseOutputAdapter = DataFrameOutputAdapter(),
    api_decorator_function_creator=Provide[
        QwakRuntimeContainer.api_decorator_function_creator
    ],
) -> Callable:
    if callable(analytics):
        raise TypeError(
            """
        You forgot to call the `@api` decorator.

        Correct way -
        @api()
        def function():
            pass

        Wrong way -
        @api
        def function():
            pass
        """
        )
    return api_decorator_function_creator(
        analytics,
        feature_extraction,
        input_adapter,
        output_adapter,
        analytics_sample_ratio,
    )
