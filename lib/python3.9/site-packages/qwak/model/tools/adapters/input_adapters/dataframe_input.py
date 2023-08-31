import json
from typing import Union, List

import pandas
from qwak.exceptions import QwakHTTPException

from .string_input import StringInput


try:
    from bentoml.adapters import DataframeInput as BentoMLDataframeInput
except ImportError:
    pass


class DataframeInput(StringInput):
    def __init__(
        self,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        # Verify pandas imported properly and retry import if it has failed initially
        try:
            import pandas  # noqa: F401
        except ImportError:
            raise ImportError("Pandas package is required to use DataframeInput.")

    def extract_user_func_args(
        self,
        data: Union[
            str, List
        ],  # it's a list of InferenceTasks, but we don't want to fail when bentoml is not installed
    ) -> "pandas.DataFrame":
        try:
            if isinstance(data, list):
                adapter = BentoMLDataframeInput()
                # because parsing is deferred to bentoml, it will also set the batch property of a task
                (df,) = adapter.extract_user_func_args(data)
                return df
            else:
                return pandas.DataFrame.from_dict(json.loads(data), orient="columns")
        except Exception as e:
            raise QwakHTTPException(
                status_code=400,
                message=f"Error loading DataFrame input: {e}.",
            )
