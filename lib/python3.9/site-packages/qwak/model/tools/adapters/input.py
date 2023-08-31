from typing import Any, Union

from qwak.exceptions import QwakHTTPException
from qwak.model.adapters import (
    DataFrameInputAdapter,
    FileInputAdapter,
    ImageInputAdapter,
    JsonInputAdapter,
    MultiInputAdapter,
    NumpyInputAdapter,
    ProtoInputAdapter,
    StringInputAdapter,
    TfTensorInputAdapter,
)
from qwak.model.base import QwakModel

from .input_adapters.dataframe_input import DataframeInput
from .input_adapters.file_input import FileInput
from .input_adapters.image_input import ImageInput
from .input_adapters.json_input import JsonInput
from .input_adapters.string_input import StringInput
from .input_adapters.tf_tensor_input import TfTensorInput


def get_input_adapter(
    model: QwakModel,
) -> Union[JsonInput, DataframeInput, StringInput]:
    adapter = getattr(model.predict, "_input_adapter", "")

    class QwakInput(StringInput):
        def extract_user_func_args(self, data: str) -> Any:
            try:
                return adapter.extract_user_func_arg(data)
            except Exception as e:
                raise QwakHTTPException(
                    status_code=500,
                    message=f"Failed to deserialize input message. Error is: {e}. For more information "
                    f"please check model logs.",
                )

    class MultiFormatInput(QwakInput):
        pass

    mapping = {
        JsonInputAdapter: JsonInput,
        DataFrameInputAdapter: DataframeInput,
        StringInputAdapter: StringInput,
        ProtoInputAdapter: QwakInput,
        NumpyInputAdapter: QwakInput,
        MultiInputAdapter: MultiFormatInput,
        TfTensorInputAdapter: TfTensorInput,
        ImageInputAdapter: ImageInput,
        FileInputAdapter: FileInput,
    }

    if adapter:
        adapter.mappings = mapping
        for qwak_impl, runtime_impl in mapping.items():
            if isinstance(adapter, qwak_impl):
                return runtime_impl()

    return DataframeInput()
