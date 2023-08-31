import inspect
from typing import Union

from qwak.feature_store.online.client import OnlineClient
from qwak.model.base import QwakModel

from .adapters.input import get_input_adapter
from .adapters.output import get_output_adapter

_FS_EXTRACTION = "_fs_extraction"
_DF = "df"
_EXTRACTED_DF = "extracted_df"


def run_local(model: QwakModel, payload: Union[str, bytes]):
    """
    Invokes the .build, .initialize_model, and in order and then invokes the input adapter, calls the predict method
    and invokes the output adapter.
    :param model: the model to run locally
    :param payload: a string or bytes with the input data
    :return: a string or bytes with the model output
    """
    if not (hasattr(model, "initialized") and model.initialized):
        model.build()
        model.initialize_model()
        setattr(model, "initialized", True)
    input_adapter = get_input_adapter(model)
    output_adapter = get_output_adapter(model)
    input_data = input_adapter.extract_user_func_args(payload)

    if getattr(model.predict, _FS_EXTRACTION, False):
        args_list = list(inspect.signature(model.predict).parameters.keys())
        if (_DF or _EXTRACTED_DF) not in args_list:
            raise ValueError(
                """
                Missing 'extracted_df' or 'df' arguments in function invocation, even though 'feature_extraction' flag is true.
                manner:
                >>> @api(feature_extraction=True)
                >>> def predict(df: pd.DataFrame, extracted_df: pd.DataFrame):
                >>>    ...
                >>>    ...
                """
            )

        extracted_df = _extract_online_features(model, input_data)
        output_data = model.predict(input_data, extracted_df)
    else:
        output_data = model.predict(input_data)

    return output_adapter.pack_user_func_return_value(output_data)


def _extract_online_features(model, df):
    return OnlineClient().get_feature_values(
        schema=model.schema(), df=df, model_name="Test Model"
    )
