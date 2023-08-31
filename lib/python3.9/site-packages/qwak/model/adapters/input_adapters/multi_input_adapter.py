from typing import Any, List, Sequence

from .base_input_adapter import BaseInputAdapter
from .proto_input_adapter import ProtoInputAdapter

try:
    from bentoml.adapters import JsonInput
    from bentoml.types import InferenceTask
except ImportError:
    pass
    # bentoml is supported either by supplying it as a dependency (or sub-dependency)
    # in your Python project, or during the runtime


class MultiInputAdapter(BaseInputAdapter):
    def __init__(self, adapters: List[BaseInputAdapter]):
        self.adapters = adapters

    def extract_user_func_arg(self, data: bytes) -> Sequence[Any]:
        task = InferenceTask(data=data)
        for inner_adapter in self.adapters:
            try:
                if isinstance(inner_adapter, ProtoInputAdapter):
                    adapter_impl = inner_adapter
                    extracted_data = inner_adapter.extract_user_func_arg(data)
                else:
                    if isinstance(inner_adapter, type):
                        adapter_impl = self.mappings.get(inner_adapter)
                    else:
                        adapter_impl = self.mappings.get(inner_adapter.__class__)

                    extracted_data = adapter_impl().extract_user_func_args([task])

                if isinstance(extracted_data, tuple):
                    extracted_data = extracted_data[0]
                if extracted_data is None:
                    raise ValueError("Parser returned None")

                if adapter_impl == JsonInput and extracted_data == [] and len(data) > 3:
                    # JsonParser return an empty array when it cannot parse the input.
                    # This if raises an exception when the returned [] is caused by an error
                    # (the input isn't an empty array)
                    raise ValueError(
                        "JsonParser returned an empty array when the data is too long for an empty array"
                    )
                return extracted_data
            except Exception as e:
                print(f"{inner_adapter} skipped an adapter because of: {e}")

        raise ValueError(f"Cannot parse with given adapters: {self.adapters}")
