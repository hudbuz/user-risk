import uuid
from typing import Dict

import grpc
from _qwak_proto.qwak.models.models_pb2 import (
    CreateModelRequest,
    CreateModelResponse,
    DeleteModelRequest,
    DeleteModelResponse,
    GetModelMetadataResponse,
    GetModelRequest,
    GetModelResponse,
    Model,
    ModelMetadata,
    ModelSpec,
    ModelStatus,
)
from _qwak_proto.qwak.models.models_pb2_grpc import ModelsManagementServiceServicer
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class ModelsManagementServiceMock(ModelsManagementServiceServicer):
    def __init__(self):
        super(ModelsManagementServiceMock, self).__init__()
        self.models: Dict[str, Model] = dict()
        self.models_by_uuid: Dict[str, Model] = dict()

    def CreateModel(self, request: CreateModelRequest, context) -> CreateModelResponse:
        try:
            formatted_model_id = request.model_spec.model_named_id.lower().replace(
                "-", "_"
            )
            created_model = self.model_spec_to_model(
                request.model_spec, formatted_model_id
            )

            self.models[formatted_model_id] = created_model
            self.models_by_uuid[created_model.uuid] = created_model

            return CreateModelResponse(model_id=formatted_model_id)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetModel(self, request: GetModelRequest, context) -> GetModelResponse:
        try:
            if request.model_id:
                if request.model_id not in self.models:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    details = f"Model id {request.model_id} not found"
                    context.set_details(details)
                    raise grpc.RpcError(details)
                else:
                    model: Model = self.models[request.model_id]
            else:
                if request.model_uuid not in self.models_by_uuid:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    details = f"Model uuid {request.model_uuid} not found"
                    context.set_details(details)
                    raise grpc.RpcError(details)
                else:
                    model: Model = self.models_by_uuid[request.model_uuid]

            return GetModelResponse(model=model)
        except grpc.RpcError as e:
            raise e
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeleteModel(self, request: DeleteModelRequest, context) -> DeleteModelResponse:
        try:
            del self.models[request.model_id]
            return DeleteModelResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetModelMetadata(self, request, context):
        try:
            if request.model_id == "throw_exception":
                raise Exception("This is a test exception")
            model = self.models[request.model_id]
            return GetModelMetadataResponse(model_metadata=ModelMetadata(model=model))
        except Exception as e:
            raise_internal_grpc_error(context, e)

    @staticmethod
    def model_spec_to_model(model_spec: ModelSpec, formatted_model_id: str):
        return Model(
            model_id=formatted_model_id,
            display_name=model_spec.display_name,
            model_status=ModelStatus.ACTIVE,
            model_description=model_spec.model_description,
            project_id=model_spec.project_id,
            uuid=str(uuid.uuid4()),
        )
