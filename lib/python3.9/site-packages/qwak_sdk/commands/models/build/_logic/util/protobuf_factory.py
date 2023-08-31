from qwak.exceptions import QwakException


def protobuf_factory(
    protobuf_class, exclude_fields=None, include_only=None, mapping=None
):
    def get_final_field_names(cls):
        fields = dict(cls.__dataclass_fields__)
        all_field_names = set(fields.keys())
        excluded_fields = set(exclude_fields) if exclude_fields else set()
        included_fields = set(include_only) if include_only else all_field_names
        unrecognized_fields = included_fields.difference(all_field_names)
        if unrecognized_fields:
            raise QwakException(
                f"Unknown fields were included: {list(unrecognized_fields)}"
            )
        return included_fields.difference(excluded_fields)

    def protofy_decorator(cls):
        if not hasattr(cls, "__dataclass_fields__"):
            raise QwakException("Protofy must receive a dataclass")

        fields_mapping = mapping if mapping else {}
        final_field_names = get_final_field_names(cls)

        def to_proto(self) -> protobuf_class:
            final_fields = {}
            for field_name in final_field_names:
                field_val = self.__getattribute__(field_name)
                field_name = (
                    fields_mapping[field_name]
                    if field_name in fields_mapping
                    else field_name
                )
                final_fields[field_name] = (
                    field_val.to_proto()
                    if hasattr(field_val, "to_proto")
                    else field_val
                )
            return protobuf_class(**final_fields)

        setattr(cls, "to_proto", to_proto)
        return cls

    return protofy_decorator
