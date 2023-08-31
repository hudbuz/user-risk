from typing import Dict, Optional

import grpc
from _qwak_proto.qwak.feature_store.entities.entity_pb2 import Entity
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    BatchFeatureSetV1,
    StreamingAggregationFeatureSet,
    StreamingFeatureSetV1,
)
from _qwak_proto.qwak.kube_deployment_captain.feature_set_deployment_pb2 import (
    BatchDetails,
    FeatureSetDeployment,
)
from _qwak_proto.qwak.kube_deployment_captain.kube_deployment_captain_service_pb2 import (
    BatchFeatureSetV1DeploymentRequest,
    DeployFeatureSetResponse,
    DeployStreamingAggregationCompactionRequest,
    DeployStreamingAggregationCompactionResponse,
    StreamingFeatureSetV1DeploymentRequest,
    UndeployBatchFeaturesetRequest,
    UndeployFeatureSetRequest,
    UndeployFeatureSetResponse,
)
from _qwak_proto.qwak.kube_deployment_captain.kube_deployment_captain_service_pb2_grpc import (
    KubeDeploymentCaptainStub,
)
from google.protobuf.timestamp_pb2 import Timestamp
from qwak.exceptions import QwakException
from qwak.inner.tool.grpc.grpc_tools import create_grpc_channel


class KubeDeploymentClient:
    """
    Used for interacting with Kube Deployment Captain
    """

    def __init__(
        self,
        edge_services_url: str,
        enable_ssl: bool = True,
        grpc_channel: Optional[grpc.Channel] = None,
    ):
        self._channel = None

        if grpc_channel:
            self._channel = grpc_channel
        else:
            self._channel = create_grpc_channel(
                url=edge_services_url,
                enable_ssl=enable_ssl,
                status_for_retry=(
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.INTERNAL,
                ),
            )

        self._kube_deployment = KubeDeploymentCaptainStub(self._channel)

    def deploy_streaming_offline_feature_set(
        self,
        feature_set_name: str,
        streaming_feature_set: StreamingFeatureSetV1,
        extra_deployment_configuration: Optional[Dict[str, str]],
        extra_env_configuration: Optional[Dict[str, str]],
        entity: Entity,
        feature_set_id: str,
        environment_id: str,
        secret_service_url: Optional[str],
        run_id: Optional[int] = 0,
    ) -> DeployFeatureSetResponse:
        """
        Deploy streaming offline feature set
        Args: feature_set_name: Name of the feature set
              extra_deployment_configuration: Dictionary of spark configuration
              extra_env_configuration: Environment configuration
              Entity: entity of the feature set
              feature_set_id: Feature set id
              environment_id: Environment id
              secret_service_url: Secret service of the url

        Return: DeployFeatureSetResponse
        """
        try:
            streaming_featureset_request = StreamingFeatureSetV1DeploymentRequest(
                feature_set_deployment=FeatureSetDeployment(
                    featureset_name=feature_set_name,
                    featureset_id=feature_set_id,
                    streaming_feature_set=streaming_feature_set,
                    extra_deployment_configuration=extra_deployment_configuration,
                    extra_env_configuration=extra_env_configuration,
                    entity=entity,
                    secret_service_url=secret_service_url,
                    run_id=run_id,
                ),
                environment_id=environment_id,
            )

            return self._kube_deployment.DeployStreamingOfflineFeatureSetV1(
                streaming_featureset_request
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to deploy offline streaming feature set, error is {repr(e)}"
            )

    def deploy_batch_feature_set(
        self,
        featureset_name: str,
        featureset_id: str,
        batch_featureset: BatchFeatureSetV1,
        extra_deployment_configuration: Optional[Dict[str, str]],
        extra_env_configuration: Optional[Dict[str, str]],
        entity: Entity,
        secret_service_url: str,
        job_run_id: int,
        environment_id: str,
        batch_execution_date: Timestamp,
        try_number: int,
        batch_trigger_date: Timestamp,
    ) -> DeployFeatureSetResponse:
        """
        Deploy batch featureset
        Args: featureset_name: Name of the feature set
              featureset_id: Id of the featureset
              batch_featureset: batch featureset definition
              extra_deployment_configuration: Dictionary of spark configuration
              extra_env_configuration: Environment configuration
              Entity: entity of the feature set
              secret_service_url: Secret service of the url
              job_run_id: the run id of the job
              environment_id: Environment id
              batch_execution_date: the current batch execution time
              prev_batch_execution_date: previous batch exection time
              prev_ingestion_window_date: previous ingestion window time
              batch_trigger_date: system trigger date
        Return: DeployFeatureSetResponse
        """
        try:
            batch_featureset_request = BatchFeatureSetV1DeploymentRequest(
                feature_set_deployment=FeatureSetDeployment(
                    featureset_name=featureset_name,
                    featureset_id=featureset_id,
                    batch_feature_set=batch_featureset,
                    extra_deployment_configuration=extra_deployment_configuration,
                    extra_env_configuration=extra_env_configuration,
                    secret_service_url=secret_service_url,
                    entity=entity,
                    run_id=job_run_id,
                ),
                batch_details=BatchDetails(
                    batch_execution_date=batch_execution_date,
                    try_number=try_number,
                    batch_trigger_date=batch_trigger_date,
                ),
                environment_id=environment_id,
            )
            return self._kube_deployment.DeployBatchFeatureSetV1(
                batch_featureset_request
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to deploy batch feature set, error is {repr(e)}"
            )

    def deploy_streaming_aggregation_compaction(
        self,
        feature_set_id: str,
        feature_set_name: str,
        environment_id: str,
        streaming_aggregation_feature_set: StreamingAggregationFeatureSet,
        entity: Entity,
        extra_deployment_configuration: Optional[Dict[str, str]],
        extra_env_configuration: Optional[Dict[str, str]],
        secret_service_url: str,
        job_run_id: int,
    ) -> DeployStreamingAggregationCompactionResponse:
        try:
            streaming_aggregation_compaction_deploy_request: DeployStreamingAggregationCompactionRequest = DeployStreamingAggregationCompactionRequest(
                feature_set_deployment=FeatureSetDeployment(
                    featureset_name=feature_set_name,
                    featureset_id=feature_set_id,
                    streaming_aggregation_feature_set=streaming_aggregation_feature_set,
                    extra_deployment_configuration=extra_deployment_configuration,
                    extra_env_configuration=extra_env_configuration,
                    entity=entity,
                    secret_service_url=secret_service_url,
                    run_id=job_run_id,
                ),
                environment_id=environment_id,
            )
            return self._kube_deployment.DeployStreamingAggregationCompaction(
                streaming_aggregation_compaction_deploy_request
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to deploy streaming aggregation feature set, error is {repr(e)}"
            )

    def undeploy_featureset_resources(
        self, featureset_name: str
    ) -> UndeployFeatureSetResponse:
        """
        Undeploy featureset by name. Delete all spark resources.
        """
        try:
            undeploy_request: UndeployFeatureSetRequest = UndeployFeatureSetRequest(
                feature_set_name=featureset_name
            )
            return self._kube_deployment.UndeployFeatureSet(undeploy_request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to undeploy featureset: {featureset_name}, error is {repr(e)}"
            )

    def undeploy_batch_featureset(
        self, featureset_name: str
    ) -> UndeployFeatureSetResponse:
        """
        Undeploy featureset by name. Delete all spark resources.
        """
        try:
            undeploy_request: UndeployBatchFeaturesetRequest = (
                UndeployBatchFeaturesetRequest(featureset_name=featureset_name)
            )
            return self._kube_deployment.UndeployBatchFeatureset(undeploy_request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to undeploy batch featureset: {featureset_name}, error is {repr(e)}"
            )
