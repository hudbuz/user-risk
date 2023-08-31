from _qwak_proto.qwak.feature_store.entities.entity_pb2 import EntitySpec
from _qwak_proto.qwak.feature_store.entities.entity_service_pb2 import (
    CreateEntityRequest,
    CreateEntityResponse,
    DeleteEntityRequest,
    DeleteEntityResponse,
    GetEntityByNameRequest,
    GetEntityByNameResponse,
    ListEntitiesRequest,
    ListEntitiesResponse,
    UpdateEntityRequest,
    UpdateEntityResponse,
)
from _qwak_proto.qwak.feature_store.entities.entity_service_pb2_grpc import (
    EntityServiceStub,
)
from _qwak_proto.qwak.feature_store.features.feature_set_pb2 import FeatureSetSpec
from _qwak_proto.qwak.feature_store.features.feature_set_service_pb2 import (
    CreateUploadURLRequest,
    CreateUploadURLRequestV1,
    DeleteFeatureSetRequest,
    DeleteFeatureSetResponse,
    GetFeatureSetByNameRequest,
    GetFeatureSetByNameResponse,
    GetFeaturesetSchedulingStateRequest,
    GetFeaturesetSchedulingStateResponse,
    ListFeatureSetsRequest,
    ListFeatureSetsResponse,
    PauseFeaturesetRequest,
    RegisterFeatureSetRequest,
    RegisterFeatureSetResponse,
    ResumeFeaturesetRequest,
    RunBatchFeatureSetRequest,
    RunParquetBatchFeatureSetRequest,
    UpdateFeatureSetRequest,
    UpdateFeatureSetResponse,
)
from _qwak_proto.qwak.feature_store.features.feature_set_service_pb2_grpc import (
    FeatureSetServiceStub,
)
from _qwak_proto.qwak.feature_store.sources.data_source_service_pb2 import (
    CreateDataSourceRequest,
    CreateDataSourceResponse,
    CreateDataSourceUploadURLRequest,
    DeleteDataSourceRequest,
    DeleteDataSourceResponse,
    GetDataSourceByNameRequest,
    GetDataSourceByNameResponse,
    ListDataSourcesRequest,
    ListDataSourcesResponse,
    UpdateDataSourceRequest,
    UpdateDataSourceResponse,
)
from _qwak_proto.qwak.feature_store.sources.data_source_service_pb2_grpc import (
    DataSourceServiceStub,
)
from dependency_injector.wiring import Provide
from grpc import RpcError, StatusCode
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class FeatureRegistryClient:
    """
    Used for interacting with Feature Registry endpoints
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._features_service = FeatureSetServiceStub(grpc_channel)
        self._entity_service = EntityServiceStub(grpc_channel)
        self._sources_service = DataSourceServiceStub(grpc_channel)

    def list_feature_sets(self) -> ListFeatureSetsResponse:
        """
        Returns:
            ListFeatureSetsResponse
        """
        try:
            return self._features_service.ListFeatureSets(ListFeatureSetsRequest())
        except RpcError as e:
            raise QwakException(f"Failed to list features, error is {e.details()}")

    def get_entity_by_name(self, entity_name: str) -> GetEntityByNameResponse:
        """
        Args:
            entity_name: entity name to get
        Returns:
            GetEntityByNameResponse
        """
        return self._fetch_by_name(
            lambda name: self._entity_service.GetEntityByName(
                GetEntityByNameRequest(entity_name=name)
            ),
            entity_name,
            "entity",
        )

    def delete_feature_set(self, feature_set_id: str) -> DeleteFeatureSetResponse:
        """
        Args:
            feature_set_id: feature set id to delete
        Returns:
            DeleteFeatureSetResponse
        """
        try:
            return self._features_service.DeleteFeatureSet(
                DeleteFeatureSetRequest(feature_set_id=feature_set_id)
            )
        except RpcError as e:
            raise QwakException(f"Failed to delete feature, error is {e.details()}")

    def delete_entity(self, entity_id: str) -> DeleteEntityResponse:
        """
        Args:
            entity_id: entity id to delete
        Returns:
            DeleteEntityResponse
        """
        try:
            return self._entity_service.DeleteEntity(
                DeleteEntityRequest(entity_id=entity_id)
            )
        except RpcError as e:
            raise QwakException(f"Failed to delete entity, error is {e.details()}")

    def delete_data_source(self, data_source_id: str) -> DeleteDataSourceResponse:
        """
        Args:
            data_source_id: data source id to delete
        Returns:
            DeleteDataSourceResponse
        """
        try:
            return self._sources_service.DeleteDataSource(
                DeleteDataSourceRequest(data_source_id=data_source_id)
            )
        except RpcError as e:
            raise QwakException(f"Failed to delete data source, error is {e.details()}")

    def run_feature_set(self, feature_set_name: str):
        """
        Trigger a single ETL run for that feature set name
        Args:
            feature_set_name: feature set name to run
        Returns:
            None
        """
        try:
            self._features_service.RunBatchFeatureSet(
                RunBatchFeatureSetRequest(feature_set_name=feature_set_name)
            )
        except RpcError as e:
            raise QwakException(f"Failed to run feature, error is {e.details()}")

    def run_parquet_batch_feature_set(
        self, feature_set_name: str, parquet_location: str, timestamp_column: str
    ):
        """
        Trigger a single ETL run on an explicit parquet source for that feature set name
        Args:
            feature_set_name: feature set name to run
            parquet_location: parquet location to read from
            timestamp_column: the name of the timestamp column
        Returns:
            None
        """
        try:
            self._features_service.RunParquetBatchFeatureSet(
                RunParquetBatchFeatureSetRequest(
                    feature_set_name=feature_set_name,
                    parquet_location=parquet_location,
                    timestamp_column=timestamp_column,
                )
            )
        except RpcError as e:
            raise QwakException(f"Failed to run feature, error is {e.details()}")

    def update_entity(
        self, entity_id: str, proto_entity_spec: EntitySpec
    ) -> UpdateEntityResponse:
        """
        Args:
            entity_id: entity id to update
            proto_entity_spec: entity spec proto to update
        Returns:
            UpdateEntityResponse
        """
        try:
            return self._entity_service.UpdateEntity(
                UpdateEntityRequest(entity_id=entity_id, entity_spec=proto_entity_spec)
            )
        except RpcError as e:
            raise QwakException(f"Failed to update entity, error is {e.details()}")

    def create_entity(self, proto_entity_spec: EntitySpec) -> CreateEntityResponse:
        """
        Args:
            proto_entity_spec: entity spec proto to create
        Returns:
            CreateEntityResponse
        """
        try:
            return self._entity_service.CreateEntity(
                CreateEntityRequest(entity_spec=proto_entity_spec)
            )
        except RpcError as e:
            raise QwakException(f"Failed to create entity, error is {e.details()}")

    def list_entities(self) -> ListEntitiesResponse:
        """
        Returns:
             ListEntitiesResponse
        """
        try:
            return self._entity_service.ListEntities(ListEntitiesRequest())
        except RpcError as e:
            raise QwakException(f"Failed to list entities, error is {e.details()}")

    def list_data_sources(self) -> ListDataSourcesResponse:
        """
        Returns:
            ListDataSourcesResponse
        """
        try:
            return self._sources_service.ListDataSources(ListDataSourcesRequest())
        except RpcError as e:
            raise QwakException(f"Failed to list data sources, error is {e.details()}")

    def get_data_source_by_name(
        self, data_source_name: str
    ) -> GetDataSourceByNameResponse:
        """
        Args:
            data_source_name: data source name to get
        Returns:
            GetDataSourceByNameResponse
        """
        return self._fetch_by_name(
            lambda name: self._sources_service.GetDataSourceByName(
                GetDataSourceByNameRequest(data_source_name=name)
            ),
            data_source_name,
            "data source",
        )

    def update_data_source(
        self,
        data_source_id,
        proto_data_source: FeatureSetSpec,
    ) -> UpdateDataSourceResponse:
        """
        Args:
            data_source_id: data source id to update
            proto_data_source: proto data source spec to update
        Returns:
             UpdateDataSourceResponse
        """
        try:
            return self._sources_service.UpdateDataSource(
                UpdateDataSourceRequest(
                    data_source_id=data_source_id,
                    data_source_spec=proto_data_source,
                )
            )
        except RpcError as e:
            raise QwakException(f"Failed to update data source, error is {e.details()}")

    def create_data_source(
        self, proto_data_source: FeatureSetSpec
    ) -> CreateDataSourceResponse:
        """
        Args:
            proto_data_source: proto data source spec to create
        Returns:
            CreateDataSourceResponse
        """
        try:
            return self._sources_service.CreateDataSource(
                CreateDataSourceRequest(data_source_spec=proto_data_source)
            )
        except RpcError as e:
            raise QwakException(f"Failed to create data source, error is {e.details()}")

    def get_feature_set_by_name(self, feature_set_name) -> GetFeatureSetByNameResponse:
        """
        Args:
            feature_set_name: feature set name to get
        Returns:
             GetFeatureSetByNameResponse
        """
        return self._fetch_by_name(
            lambda name: self._features_service.GetFeatureSetByName(
                GetFeatureSetByNameRequest(feature_set_name=name)
            ),
            feature_set_name,
            "feature set",
        )

    def update_feature_set(
        self,
        feature_set_id: str,
        proto_feature_set: FeatureSetSpec,
    ) -> UpdateFeatureSetResponse:
        """
        Args:
            feature_set_id: feature set id to update
            proto_feature_set: features set to proto serialization
        Returns:
            UpdateFeatureSetResponse
        """

        try:
            return self._features_service.UpdateFeatureSet(
                UpdateFeatureSetRequest(
                    feature_set_id=feature_set_id, feature_set_spec=proto_feature_set
                )
            )
        except RpcError as e:
            raise QwakException(f"Failed to update feature set, error is {e.details()}")

    def create_feature_set(
        self, proto_feature_set: FeatureSetSpec
    ) -> RegisterFeatureSetResponse:
        """
        Args:
            proto_feature_set: features set to proto serialization
        Returns:
            RegisterFeatureSetResponse
        """

        try:
            return self._features_service.RegisterFeatureSet(
                RegisterFeatureSetRequest(feature_set_spec=proto_feature_set)
            )
        except RpcError as e:
            raise QwakException(f"Failed to create feature set, error is {e.details()}")

    @staticmethod
    def _fetch_by_name(extract_function, name, fs_object_type):
        """
        Args:
            extract_function: function to run on name param
            name: name of object to fetch
            fs_object_type: fs object type for exception reporting
        Returns:
            extract_function output
        """
        try:
            return extract_function(name)
        except RpcError as e:
            if e.code() == StatusCode.NOT_FOUND:
                return None
            else:
                raise QwakException(
                    f"Failed to fetch {fs_object_type} by name [{name}], error code: [{e.details()}]"
                )

    def get_presigned_url(self, feature_set_name: str):
        """
        get presigned url according feature set name
        Args:
            feature_set_name: name of the feature set
        Returns:
            presigned url
        """
        try:
            return self._features_service.CreateUploadURL(
                CreateUploadURLRequest(feature_set_name=feature_set_name)
            ).upload_url
        except Exception as e:
            raise QwakException(
                f"Failed to get presigned url for feature name: {feature_set_name} , error code: [{e.details()}]"
            )

    def get_presigned_url_v1(self, feature_set_name: str, transformation) -> str:
        """
        Get presigned url according feature set name and transformation type
        @param feature_set_name: featureset name for which we're getting the presigned URL
        @type feature_set_name: str
        @param transformation: The transformation for which we're getting the presigned URL
        @type transformation: BatchTransformation
        @return:
        @rtype:
        """
        try:
            return self._features_service.CreateUploadURLV1(
                CreateUploadURLRequestV1(
                    featureset_name=feature_set_name,
                    transformation=transformation._to_proto(),
                )
            ).upload_url
        except Exception as e:
            raise QwakException(
                f"Failed to get presigned url for featureset name {feature_set_name} and transformation type "
                f"{type(transformation)}, error code: [{e.details()}]"
            )

    def get_data_source_presigned_url(
        self, data_source_name: str, object_name: str
    ) -> str:
        """
        create a pre-sign url for data source according to qwak path convention
        Args:
            data_source_name: str:  data source name
            object_name: str object name
        Returns:
            presigned url: str
        """
        try:
            return self._sources_service.CreateDataSourceUploadURL(
                CreateDataSourceUploadURLRequest(
                    data_source_name=data_source_name, object_name=object_name
                )
            ).upload_url
        except Exception as e:
            raise QwakException(
                f"Failed to get presigned url for data source: {data_source_name}, object name: {object_name} , error code: [{e.details()}]"
            ) from e

    def get_feature_set_scheduling_state(
        self, feature_set_name: str
    ) -> GetFeaturesetSchedulingStateResponse:
        try:
            return self._features_service.GetFeaturesetSchedulingState(
                GetFeaturesetSchedulingStateRequest(featureset_name=feature_set_name)
            )
        except Exception as e:
            raise QwakException(
                f"Failed to query scheduling state for Featureset {feature_set_name}, nested exception: [{e.details()}]"
            ) from e

    def pause_feature_set(self, feature_set_name: str):
        try:
            self._features_service.PauseFeatureset(
                PauseFeaturesetRequest(featureset_name=feature_set_name)
            )
        except Exception as e:
            raise QwakException(
                f"Failed to pause Featureset {feature_set_name}, nested exception: [{e.details()}]"
            ) from e

    def resume_feature_set(self, feature_set_name: str):
        try:
            self._features_service.ResumeFeatureset(
                ResumeFeaturesetRequest(featureset_name=feature_set_name)
            )
        except Exception as e:
            raise QwakException(
                f"Failed to resume Featureset {feature_set_name}, nested exception: [{e.details()}]"
            ) from e
