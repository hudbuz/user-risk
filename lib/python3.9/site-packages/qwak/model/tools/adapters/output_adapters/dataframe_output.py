import json

from qwak.exceptions import QwakHTTPException

from .json_output import JsonOutput


def df_to_json(result):
    import pandas as pd

    if isinstance(result, pd.DataFrame):
        return result.to_json(orient="records")

    if isinstance(result, pd.Series):
        return pd.DataFrame(result).to_json(orient="records")
    return json.dumps(result)


class DataFrameOutput(JsonOutput):
    def pack_user_func_return_value(
        self,
        return_result,
    ) -> str:
        try:
            return df_to_json(return_result)
        except Exception as e:
            raise QwakHTTPException(message=str(e), status_code=500)
