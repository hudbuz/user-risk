import uuid

import grpc
from _qwak_proto.qwak.feature_store.features.feature_set_pb2 import (
    DeployedModelInUseLink,
    FeatureSet,
    FeatureSetDefinition,
    FeatureSetMetadata,
    FeaturesetSchedulingState,
    FeatureStatus,
)
from _qwak_proto.qwak.feature_store.features.feature_set_service_pb2 import (
    DeleteFeatureSetResponse,
    GetFeatureSetByNameResponse,
    GetFeaturesetSchedulingStateResponse,
    ListFeatureSetsResponse,
    RegisterFeatureSetResponse,
)
from _qwak_proto.qwak.feature_store.features.feature_set_service_pb2_grpc import (
    FeatureSetServiceServicer,
)


class FeatureSetServiceMock(FeatureSetServiceServicer):
    def __init__(self):
        self._features_spec = {}

    def reset_features(self):
        self._features_spec.clear()

    def RegisterFeatureSet(self, request, context):
        fs_id = str(uuid.uuid4())
        self._features_spec[fs_id] = request.feature_set_spec
        return RegisterFeatureSetResponse(
            feature_set=FeatureSet(
                feature_set_definition=FeatureSetDefinition(
                    feature_set_id=fs_id, feature_set_spec=request.feature_set_spec
                )
            )
        )

    def DeleteFeatureSet(self, request, context):
        if request.feature_set_id in self._features_spec:
            self._features_spec.pop(request.feature_set_id)
            return DeleteFeatureSetResponse()

        context.set_details(f"Feature set ID {request.feature_set_id} doesn't exist'")
        context.set_code(grpc.StatusCode.NOT_FOUND)

    def GetFeatureSetByName(self, request, context):
        feature_sets = [
            (fs_id, fs_spec)
            for (fs_id, fs_spec) in self._features_spec.items()
            if fs_spec.name == request.feature_set_name
        ]
        if feature_sets:
            fs_id, fs_spec = feature_sets[0]
            return GetFeatureSetByNameResponse(
                feature_set=FeatureSet(
                    feature_set_definition=FeatureSetDefinition(
                        feature_set_id=fs_id,
                        feature_set_spec=fs_spec,
                        status=FeatureStatus.VALID,
                    ),
                    metadata=FeatureSetMetadata(),
                    deployed_models_in_use_link=[DeployedModelInUseLink()],
                )
            )

        context.set_details(
            f"Feature set named {request.feature_set_name} doesn't exist'"
        )
        context.set_code(grpc.StatusCode.NOT_FOUND)

    def ListFeatureSets(self, request, context):
        return ListFeatureSetsResponse(
            feature_families=[
                FeatureSet(
                    feature_set_definition=FeatureSetDefinition(
                        feature_set_spec=feature_spec, status=FeatureStatus.VALID
                    ),
                    metadata=FeatureSetMetadata(),
                    deployed_models_in_use_link=[DeployedModelInUseLink()],
                )
                for feature_spec in self._features_spec.values()
            ]
        )

    def GetFeaturesetSchedulingState(self, request, context):
        return GetFeaturesetSchedulingStateResponse(
            state=FeaturesetSchedulingState.SCHEDULING_STATE_ENABLED
        )
