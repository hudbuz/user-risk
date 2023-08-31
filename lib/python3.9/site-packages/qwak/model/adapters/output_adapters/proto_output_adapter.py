from google.protobuf.message import Message

from .base_output_adapter import BaseOutputAdapter


class ProtoOutputAdapter(BaseOutputAdapter):
    @staticmethod
    def http_content_type():
        return {"Content-Type": "application/octet-stream"}

    @staticmethod
    def pack_user_func_return_value(return_result: Message) -> bytes:
        try:
            return return_result.SerializeToString()
        except Exception as e:
            raise ValueError(f"Failed to serialize proto message. Error is {e}")
