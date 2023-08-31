import dataclasses
from typing import Any, Dict, List, Type, Union


def create_dataclass_instance(dataclass_type: Type, input_dict: Dict[str, Any]) -> Any:
    fields = {f.name: f.type for f in dataclasses.fields(dataclass_type)}

    def construct_dataclass(
        dc_type: Union[Type, List], value: Any
    ) -> Union[Type, List]:
        if dataclasses.is_dataclass(dc_type):
            return create_dataclass_instance(dc_type, value)
        elif getattr(dc_type, "__origin__", None) is list:  # This is a List
            element_type = dc_type.__args__[0]
            return [construct_dataclass(element_type, v) for v in value]
        else:
            return value

    return dataclass_type(
        **{k: construct_dataclass(fields[k], v) for k, v in input_dict.items()}
    )
