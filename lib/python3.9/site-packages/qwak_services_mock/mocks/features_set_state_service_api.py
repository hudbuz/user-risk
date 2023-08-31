import grpc
from _qwak_proto.qwak.feature_store.features.feature_set_state_pb2 import (
    FeatureSetState,
)
from _qwak_proto.qwak.feature_store.features.feature_set_state_service_pb2 import (
    GetFeatureSetStateByIdV1Response,
    GetFeatureSetStateByNameV1Response,
)
from _qwak_proto.qwak.feature_store.features.feature_set_state_service_pb2_grpc import (
    FeatureSetStateServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class FeaturesSetStateServiceApiMock(FeatureSetStateServiceServicer):
    def __init__(self):

        self._feature_state_states: dict[(str, str):FeatureSetState] = {}
        super(FeaturesSetStateServiceApiMock, self).__init__()

    def given_feature_set_state(self, feature_set_state: FeatureSetState):
        self._feature_state_states[
            (feature_set_state.feature_set_id, feature_set_state.run_id)
        ] = feature_set_state

    def GetFeatureSetStateById(self, request, context):
        """get feature set state by id"""
        try:
            return GetFeatureSetStateByIdV1Response(
                feature_set_state=self._feature_state_states[
                    (request.feature_set_id, request.run_id)
                ]
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetFeatureSetStateByName(self, request, context):
        """get feature set state by name"""
        try:
            feature_sets_by_name = [
                self._feature_state_states[(feature_set_id, run_id)]
                for feature_set_id, run_id in self._feature_state_states.keys()
                if self._feature_state_states[(feature_set_id, run_id)].feature_set_name
                == request.feature_set_name
            ]
            if len(feature_sets_by_name) > 0:
                return GetFeatureSetStateByNameV1Response(
                    feature_set_state=feature_sets_by_name.pop()
                )
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Received a non existing feature set name")
                return GetFeatureSetStateByNameV1Response()
        except Exception as e:
            raise_internal_grpc_error(context, e)
