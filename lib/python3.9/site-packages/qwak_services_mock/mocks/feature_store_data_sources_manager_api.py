import grpc
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSource,
    DataSourceDefinition,
)
from _qwak_proto.qwak.feature_store.sources.data_source_service_pb2 import (
    CreateDataSourceResponse,
    GetDataSourceByNameResponse,
)
from _qwak_proto.qwak.feature_store.sources.data_source_service_pb2_grpc import (
    DataSourceServiceServicer,
)


class DataSourceServiceMock(DataSourceServiceServicer):
    def __init__(self):
        self._data_sources_spec = {}

    def CreateDataSource(self, request, context):
        data_source_type = request.data_source_spec.WhichOneof("type")
        ds_name = (
            request.data_source_spec.batch_source.name
            if data_source_type == "batch_source"
            else request.data_source_spec.stream_source.name
        )

        self._data_sources_spec[ds_name] = request.data_source_spec
        return CreateDataSourceResponse()

    def GetDataSourceByName(self, request, context):
        if request.data_source_name not in self._data_sources_spec:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return GetDataSourceByNameResponse()

        return GetDataSourceByNameResponse(
            data_source=DataSource(
                data_source_definition=DataSourceDefinition(
                    data_source_id="123",
                    data_source_spec=self._data_sources_spec[request.data_source_name],
                ),
                metadata=None,
                feature_sets=[],
            )
        )
