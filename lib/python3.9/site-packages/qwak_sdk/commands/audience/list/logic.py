from typing import List

import grpc
from _qwak_proto.qwak.audience.v1.audience_pb2 import AudienceEntry
from qwak.clients.audience import AudienceClient

from qwak_sdk.exceptions import QwakCommandException, QwakResourceNotFound


def list_audience() -> List[AudienceEntry]:
    try:
        return AudienceClient().list_audience()
    except grpc.RpcError as e:
        if e.args[0].code == grpc.StatusCode.NOT_FOUND:
            raise QwakResourceNotFound(e.args[0].details)
        raise QwakCommandException(e.args[0].details)
