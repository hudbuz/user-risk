import grpc
from _qwak_proto.qwak.audience.v1.audience_pb2 import Audience
from qwak.clients.audience import AudienceClient

from qwak_sdk.exceptions import QwakCommandException, QwakResourceNotFound


def get_audience(audience_id: str) -> Audience:
    try:
        return AudienceClient().get_audience(audience_id=audience_id)
    except grpc.RpcError as e:
        if e.args[0].code == grpc.StatusCode.NOT_FOUND:
            raise QwakResourceNotFound(e.args[0].details)
        raise QwakCommandException(e.args[0].details)
