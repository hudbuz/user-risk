from enum import Enum
from typing import TypeVar

from .base_input_adapter import BaseInputAdapter

T = TypeVar("T")


class ProtobufFlavor(Enum):
    GOOGLE_PROTO = 1
    BETTERPROTO = 2


class ProtoInputAdapter(BaseInputAdapter):
    def __init__(
        self, message: T, flavor: ProtobufFlavor = ProtobufFlavor.GOOGLE_PROTO
    ):
        self.message: T = message
        self.flavor = flavor

    def extract_user_func_arg(self, data: bytes) -> T:
        try:
            model_input = self.message()
            if self.flavor is ProtobufFlavor.GOOGLE_PROTO:
                model_input.ParseFromString(data)

            if self.flavor is ProtobufFlavor.BETTERPROTO:
                model_input.parse(data)

            return model_input
        except Exception as e:
            raise ValueError(f"Failed to serialize Proto message. Error is {e}")
