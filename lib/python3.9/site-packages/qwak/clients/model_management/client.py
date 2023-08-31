from typing import Optional

import grpc
from _qwak_proto.qwak.models.models_pb2 import (
    CreateModelRequest,
    DeleteModelRequest,
    GetModelMetadataRequest,
    GetModelRequest,
    Model,
    ModelSpec,
)
from _qwak_proto.qwak.models.models_pb2_grpc import ModelsManagementServiceStub
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class ModelsManagementClient:
    """
    Used for interacting with Feature Registry endpoints
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._models_management_service = ModelsManagementServiceStub(grpc_channel)

    def get_model(self, model_id, exception_on_missing: bool = True) -> Optional[Model]:
        try:
            return self._models_management_service.GetModel(
                GetModelRequest(model_id=model_id)
            ).model

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND and not exception_on_missing:
                return None
            raise QwakException(f"Failed to get model, error is {e.details()}")

    def get_model_by_uuid(
        self, model_uuid, exception_on_missing: bool = True
    ) -> Optional[Model]:
        try:
            return self._models_management_service.GetModel(
                GetModelRequest(model_uuid=model_uuid)
            ).model

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND and not exception_on_missing:
                return None
            raise QwakException(f"Failed to get model, error is {e.details()}")

    def get_model_uuid(self, model_id):
        model = self.get_model(model_id)
        return model.uuid

    def create_model(self, project_id, model_name, model_description):
        try:
            return self._models_management_service.CreateModel(
                CreateModelRequest(
                    model_spec=ModelSpec(
                        model_named_id=model_name.lower().replace(" ", "-"),
                        display_name=model_name,
                        project_id=project_id,
                        model_description=model_description,
                    )
                )
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to create model, error is {e}")

    def delete_model(self, project_id, model_id):
        try:
            return self._models_management_service.DeleteModel(
                DeleteModelRequest(model_id=model_id, project_id=project_id)
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to delete model, error is {e.details()}")

    def is_model_exists(self, model_id: str) -> bool:
        """Check if model exists in environment

        Args:
            model_id: the model id to check if exists.

        Returns: if model exists.
        """
        try:
            self._models_management_service.GetModel(
                GetModelRequest(model_id=model_id)
            ).model
            return True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return False
            raise QwakException(
                f"Failed to check if model {model_id} is exists, error is {e.details()}"
            )

    def get_model_metadata(self, model_id: str):
        try:
            return self._models_management_service.GetModelMetadata(
                GetModelMetadataRequest(model_id=model_id)
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to get model metadata, error is {e.details()}")
