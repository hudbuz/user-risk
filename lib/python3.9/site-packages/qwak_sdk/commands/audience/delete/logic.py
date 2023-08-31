import grpc
from qwak.clients.audience import AudienceClient

from qwak_sdk.exceptions import QwakCommandException, QwakResourceNotFound


def delete_audience(audience_id: str) -> None:
    try:
        AudienceClient().delete_audience(audience_id=audience_id)
    except grpc.RpcError as e:
        if e.args[0].code == grpc.StatusCode.NOT_FOUND:
            raise QwakResourceNotFound(e.args[0].details)
        raise QwakCommandException(e.args[0].details)
