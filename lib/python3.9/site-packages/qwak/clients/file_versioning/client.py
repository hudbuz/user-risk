from pathlib import Path
from typing import Optional

import grpc
from _qwak_proto.qwak.file_versioning.file_versioning_pb2 import FileTagSpec
from _qwak_proto.qwak.file_versioning.file_versioning_service_pb2 import (
    RegisterFileTagRequest,
)
from _qwak_proto.qwak.file_versioning.file_versioning_service_pb2_grpc import (
    FileVersioningManagementServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class FileVersioningManagementClient:
    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._file_management_service = FileVersioningManagementServiceStub(
            grpc_channel
        )

    def register_file_tag(
        self, model_id: str, tag: str, file_path: str, build_id: Optional[str] = ""
    ) -> None:
        """
        Register file tag to service
        Args:
            model_id: model id to save tag by.
            file_path: The file path
            tag: tag to save the file under.
            build_id: build id to save tag by.

        Returns:

        """
        try:
            file_extension = Path(file_path).suffix.replace(".", "")

            self._file_management_service.RegisterFileTag(
                RegisterFileTagRequest(
                    file_tag_spec=(
                        FileTagSpec(
                            build_id=build_id,
                            model_id=model_id,
                            tag=tag,
                            extension_type=file_extension,
                        )
                    )
                )
            )
        except grpc.RpcError as e:
            if e.args[0].code != grpc.StatusCode.ALREADY_EXISTS:
                raise QwakException(
                    f"Failed to register file tag, error is {e.details()}"
                )
            else:
                raise e
