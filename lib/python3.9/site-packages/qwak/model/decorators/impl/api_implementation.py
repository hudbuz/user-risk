from functools import wraps
from typing import Any, Callable

from qwak.model.adapters import BaseInputAdapter, BaseOutputAdapter
from qwak.model.utils.extract_wrapped_function import extract_wrapped


def create_decorator_function(
    analytics: bool,
    feature_extraction: bool,
    input_adapter: BaseInputAdapter,
    output_adapter: BaseOutputAdapter,
    analytics_sample_ratio: float,
) -> Callable:
    def api_decorator_function_load_logic(func: Callable[..., Any]):
        func = extract_wrapped(func)
        setattr(func, "_input_adapter", input_adapter)
        setattr(func, "_output_adapter", output_adapter)
        setattr(func, "_fs_extraction", feature_extraction)

        @wraps(func)
        def api_decorator_function_execution_logic(*args, **kwargs) -> Any:
            output = func(*args, **kwargs)
            return output

        return api_decorator_function_execution_logic

    return api_decorator_function_load_logic
