from typing import Optional

import grpc
from _qwak_proto.qwak.data_versioning.data_versioning_pb2 import DataTagSpec
from _qwak_proto.qwak.data_versioning.data_versioning_service_pb2 import (
    RegisterDataTagRequest,
)
from _qwak_proto.qwak.data_versioning.data_versioning_service_pb2_grpc import (
    DataVersioningManagementServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class DataVersioningManagementClient:
    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._data_management_service = DataVersioningManagementServiceStub(
            grpc_channel
        )

    def register_data_tag(
        self,
        model_id: str,
        tag: str,
        build_id: Optional[str] = "",
        extension: Optional[str] = "",
    ) -> None:
        """
        Register data tag to service
        Args:
            build_id: build id to save tag by.
            model_id: model id to save tag by.
            tag: tag to save.
            extension: The file extension

        Returns:

        """
        try:
            self._data_management_service.RegisterDataTag(
                RegisterDataTagRequest(
                    data_tag_spec=(
                        DataTagSpec(
                            build_id=build_id,
                            model_id=model_id,
                            tag=tag,
                            extension_type=extension,
                        )
                    )
                )
            )
        except grpc.RpcError as e:
            if e.args[0].code != grpc.StatusCode.ALREADY_EXISTS:
                raise QwakException(
                    f"Failed to register data tag, error is {e.details()}"
                )
