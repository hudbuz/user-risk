from concurrent import futures
from typing import Any, Generator, List, Tuple

import grpc
import pytest
from _qwak_proto.qwak.administration.v0.authentication.authentication_service_pb2_grpc import (
    add_AuthenticationServiceServicer_to_server,
)
from _qwak_proto.qwak.admiral.user_application_instance.v0.user_application_instance_service_pb2_grpc import (
    add_UserApplicationInstanceServiceServicer_to_server,
)
from _qwak_proto.qwak.analytics.analytics_service_pb2_grpc import (
    add_AnalyticsQueryServiceServicer_to_server,
)
from _qwak_proto.qwak.audience.v1.audience_api_pb2_grpc import (
    add_AudienceAPIServicer_to_server,
)
from _qwak_proto.qwak.auto_scaling.v1.auto_scaling_service_pb2_grpc import (
    add_AutoScalingServiceServicer_to_server,
)
from _qwak_proto.qwak.automation.v1.automation_management_service_pb2_grpc import (
    add_AutomationManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.batch_job.v1.batch_job_service_pb2_grpc import (
    add_BatchJobManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.build.v1.build_api_pb2_grpc import add_BuildAPIServicer_to_server
from _qwak_proto.qwak.builds.builds_orchestrator_service_pb2_grpc import (
    add_BuildsOrchestratorServiceServicer_to_server,
)
from _qwak_proto.qwak.builds.builds_pb2_grpc import (
    add_BuildsManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.builds.internal_builds_orchestrator_service_pb2_grpc import (
    add_InternalBuildsOrchestratorServiceServicer_to_server,
)
from _qwak_proto.qwak.data_versioning.data_versioning_service_pb2_grpc import (
    add_DataVersioningManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.deployment.alert_service_pb2_grpc import (
    add_AlertManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.deployment.deployment_service_pb2_grpc import (
    add_DeploymentManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.ecosystem.v0.ecosystem_runtime_service_pb2_grpc import (
    add_QwakEcosystemRuntimeServicer_to_server,
)
from _qwak_proto.qwak.feature_store.entities.entity_service_pb2_grpc import (
    add_EntityServiceServicer_to_server,
)
from _qwak_proto.qwak.feature_store.features.feature_set_service_pb2_grpc import (
    add_FeatureSetServiceServicer_to_server,
)
from _qwak_proto.qwak.feature_store.jobs.v1.job_service_pb2_grpc import (
    add_JobServiceServicer_to_server,
)
from _qwak_proto.qwak.feature_store.sources.data_source_service_pb2_grpc import (
    add_DataSourceServiceServicer_to_server,
)
from _qwak_proto.qwak.features_operator.v3.features_operator_async_service_pb2_grpc import (
    add_FeaturesOperatorAsyncServiceServicer_to_server,
)
from _qwak_proto.qwak.file_versioning.file_versioning_service_pb2_grpc import (
    add_FileVersioningManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.instance_template.instance_template_service_pb2_grpc import (
    add_InstanceTemplateManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.kube_deployment_captain.kube_deployment_captain_service_pb2_grpc import (
    add_KubeDeploymentCaptainServicer_to_server,
)
from _qwak_proto.qwak.logging.log_reader_service_pb2_grpc import (
    add_LogReaderServiceServicer_to_server,
)
from _qwak_proto.qwak.models.models_pb2_grpc import (
    add_ModelsManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.monitoring.v0.alerting_channel_management_service_pb2_grpc import (
    add_AlertingChannelManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.projects.projects_pb2_grpc import (
    add_ProjectsManagementServiceServicer_to_server,
)
from _qwak_proto.qwak.secret_service.secret_service_pb2_grpc import (
    add_SecretServiceServicer_to_server,
)
from _qwak_proto.qwak.self_service.user.v1.user_service_pb2_grpc import (
    add_UserServiceServicer_to_server,
)
from _qwak_proto.qwak.vectors.v1.collection.collection_service_pb2_grpc import (
    add_VectorCollectionServiceServicer_to_server,
)
from _qwak_proto.qwak.vectors.v1.vector_service_pb2_grpc import (
    add_VectorServiceServicer_to_server,
)
from _qwak_proto.qwak.workspace.workspace_service_pb2_grpc import (
    add_WorkspaceManagementServiceServicer_to_server,
)
from qwak.inner.di_configuration import QwakContainer
from qwak_services_mock.mocks.alert_manager_service_api import (
    AlertManagerServiceApiMock,
)
from qwak_services_mock.mocks.alert_registry_service_api import (
    AlertsRegistryServiceApiMock,
)
from qwak_services_mock.mocks.analytics_api import AnalyticsApiMock
from qwak_services_mock.mocks.audience_service_api import AudienceServiceApiMock
from qwak_services_mock.mocks.authentication_service import AuthenticationServiceMock
from qwak_services_mock.mocks.automation_management_service import (
    AutomationManagementServiceMock,
)
from qwak_services_mock.mocks.autoscaling_service_api import AutoscalingServiceApiMock
from qwak_services_mock.mocks.batch_job_manager_service import BatchJobManagerService
from qwak_services_mock.mocks.build_management import BuildsManagementServiceMock
from qwak_services_mock.mocks.build_orchestrator_build_api import (
    BuildOrchestratorBuildApiMock,
)
from qwak_services_mock.mocks.build_orchestrator_service_api import (
    BuildOrchestratorServiceApiMock,
)
from qwak_services_mock.mocks.data_versioning_service import DataVersioningServiceMock
from qwak_services_mock.mocks.deployment_management_service import (
    DeploymentManagementServiceMock,
)
from qwak_services_mock.mocks.ecosystem_service_api import EcoSystemServiceMock
from qwak_services_mock.mocks.feature_store_data_sources_manager_api import (
    DataSourceServiceMock,
)
from qwak_services_mock.mocks.feature_store_entities_manager_api import (
    EntityServiceMock,
)
from qwak_services_mock.mocks.feature_store_feature_set_manager_api import (
    FeatureSetServiceMock,
)
from qwak_services_mock.mocks.features_operator_v3_service import (
    FeaturesOperatorV3ServiceMock,
)
from qwak_services_mock.mocks.file_versioning_service import FileVersioningServiceMock
from qwak_services_mock.mocks.instance_template_management_service import (
    InstanceTemplateManagementServiceMock,
)
from qwak_services_mock.mocks.internal_build_orchestrator_service import (
    InternalBuildOrchestratorServiceMock,
)
from qwak_services_mock.mocks.job_registry_service_api import JobRegistryServiceApiMock
from qwak_services_mock.mocks.kube_captain_service_api import KubeCaptainServiceApiMock
from qwak_services_mock.mocks.logging_service import LoggingServiceApiMock
from qwak_services_mock.mocks.model_management_service import (
    ModelsManagementServiceMock,
)
from qwak_services_mock.mocks.project_manager_service import ProjectManagerServiceMock
from qwak_services_mock.mocks.qwak_mocks import QwakMocks
from qwak_services_mock.mocks.secret_service import SecretServiceMock
from qwak_services_mock.mocks.self_service_user_service import (
    SelfServiceUserServiceMock,
)
from qwak_services_mock.mocks.user_application_instance_service_api import (
    UserApplicationInstanceServiceApiMock,
)
from qwak_services_mock.mocks.vector_serving_api import VectorServingServiceMock
from qwak_services_mock.mocks.vectors_management_api import (
    VectorCollectionManagementServiceMock,
)
from qwak_services_mock.mocks.workspace_manager_service_mock import (
    WorkspaceManagerServiceMock,
)
from qwak_services_mock.utils.service_utils import find_free_port


def qwak_container():
    free_port = find_free_port()
    container = QwakContainer(
        config={
            "grpc": {
                "core": {
                    "address": f"localhost:{free_port}",
                    "enable_ssl": False,
                },
                "builds": {
                    "internal_address": f"localhost:{free_port}",
                    "enable_ssl": False,
                },
            },
        },
    )
    from qwak.clients import (
        alert_management,
        alerts_registry,
        analytics,
        audience,
        automation_management,
        autoscaling,
        batch_job_management,
        build_management,
        build_orchestrator,
        data_versioning,
        deployment,
        feature_store,
        file_versioning,
        instance_template,
        kube_deployment_captain,
        logging_client,
        model_management,
        project,
        secret_service,
        user_application_instance,
        vector_store,
        workspace_manager,
    )
    from qwak.clients.administration import eco_system, self_service

    container.wire(
        packages=[
            alert_management,
            audience,
            automation_management,
            autoscaling,
            analytics,
            batch_job_management,
            build_management,
            build_orchestrator,
            data_versioning,
            deployment,
            instance_template,
            feature_store,
            file_versioning,
            kube_deployment_captain,
            logging_client,
            model_management,
            project,
            self_service,
            eco_system,
            user_application_instance,
            secret_service,
            alerts_registry,
            workspace_manager,
            vector_store,
        ]
    )

    return free_port


def qwak_service_mock_creator(server, mocks: List[Tuple[Any, Any, Any]]) -> QwakMocks:
    activated_mocks = {
        mock[0]: mock[1]() if callable(mock[1]) else mock[1] for mock in mocks
    }
    qwak_mocks = QwakMocks(**activated_mocks)
    for property_name, value, servicer in mocks:
        if servicer:
            servicer(getattr(qwak_mocks, property_name), server)

    return qwak_mocks


def attach_servicers(free_port, server):
    qwak_mocks = qwak_service_mock_creator(
        server,
        [
            (
                "autoscaling_service_mock",
                AutoscalingServiceApiMock,
                add_AutoScalingServiceServicer_to_server,
            ),
            (
                "build_orchestrator_build_api",
                BuildOrchestratorBuildApiMock,
                add_BuildAPIServicer_to_server,
            ),
            (
                "build_orchestrator_service_api",
                BuildOrchestratorServiceApiMock,
                add_BuildsOrchestratorServiceServicer_to_server,
            ),
            (
                "internal_build_orchestrator_service",
                InternalBuildOrchestratorServiceMock,
                add_InternalBuildsOrchestratorServiceServicer_to_server,
            ),
            (
                "build_management_service_mock",
                BuildsManagementServiceMock,
                add_BuildsManagementServiceServicer_to_server,
            ),
            (
                "alert_manager_service_mock",
                AlertManagerServiceApiMock,
                add_AlertManagementServiceServicer_to_server,
            ),
            (
                "automation_management_service_mock",
                AutomationManagementServiceMock,
                add_AutomationManagementServiceServicer_to_server,
            ),
            (
                "ecosystem_client_mock",
                EcoSystemServiceMock,
                add_QwakEcosystemRuntimeServicer_to_server,
            ),
            (
                "job_registry_service_mock",
                JobRegistryServiceApiMock,
                add_JobServiceServicer_to_server,
            ),
            (
                "project_manager_service_mock",
                ProjectManagerServiceMock,
                add_ProjectsManagementServiceServicer_to_server,
            ),
            (
                "kube_captain_service_mock",
                KubeCaptainServiceApiMock,
                add_KubeDeploymentCaptainServicer_to_server,
            ),
            (
                "file_versioning_service_mock",
                FileVersioningServiceMock,
                add_FileVersioningManagementServiceServicer_to_server,
            ),
            (
                "data_versioning_service_mock",
                DataVersioningServiceMock,
                add_DataVersioningManagementServiceServicer_to_server,
            ),
            (
                "model_management_service_mock",
                ModelsManagementServiceMock,
                add_ModelsManagementServiceServicer_to_server,
            ),
            (
                "logging_service_mock",
                LoggingServiceApiMock,
                add_LogReaderServiceServicer_to_server,
            ),
            (
                "audience_api_mock",
                AudienceServiceApiMock,
                add_AudienceAPIServicer_to_server,
            ),
            (
                "self_service_user_service_mock",
                SelfServiceUserServiceMock,
                add_UserServiceServicer_to_server,
            ),
            (
                "analytics_api_mock",
                AnalyticsApiMock,
                add_AnalyticsQueryServiceServicer_to_server,
            ),
            (
                "deployment_management_service_mock",
                DeploymentManagementServiceMock,
                add_DeploymentManagementServiceServicer_to_server,
            ),
            (
                "batch_job_manager_service",
                BatchJobManagerService,
                add_BatchJobManagementServiceServicer_to_server,
            ),
            (
                "user_application_instance_service_mock",
                UserApplicationInstanceServiceApiMock,
                add_UserApplicationInstanceServiceServicer_to_server,
            ),
            (
                "secret_service_mock",
                SecretServiceMock,
                add_SecretServiceServicer_to_server,
            ),
            (
                "authentication_service_mock",
                AuthenticationServiceMock,
                add_AuthenticationServiceServicer_to_server,
            ),
            (
                "fs_entities_service",
                EntityServiceMock,
                add_EntityServiceServicer_to_server,
            ),
            (
                "fs_feature_sets_service",
                FeatureSetServiceMock,
                add_FeatureSetServiceServicer_to_server,
            ),
            (
                "fs_data_sources_service",
                DataSourceServiceMock,
                add_DataSourceServiceServicer_to_server,
            ),
            (
                "features_operator_service",
                FeaturesOperatorV3ServiceMock,
                add_FeaturesOperatorAsyncServiceServicer_to_server,
            ),
            (
                "instance_templates_service",
                InstanceTemplateManagementServiceMock,
                add_InstanceTemplateManagementServiceServicer_to_server,
            ),
            (
                "alerts_registry_service",
                AlertsRegistryServiceApiMock,
                add_AlertingChannelManagementServiceServicer_to_server,
            ),
            (
                "workspace_manager_service",
                WorkspaceManagerServiceMock,
                add_WorkspaceManagementServiceServicer_to_server,
            ),
            (
                "vector_serving_service",
                VectorServingServiceMock,
                add_VectorServiceServicer_to_server,
            ),
            (
                "vector_collection_service",
                VectorCollectionManagementServiceMock,
                add_VectorCollectionServiceServicer_to_server,
            ),
            ("port", free_port, None),
        ],
    )
    return qwak_mocks


@pytest.fixture
def qwak_services_mock() -> Generator[QwakMocks, None, None]:
    free_port = qwak_container()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    qwak_mocks = attach_servicers(free_port, server)

    server.add_insecure_port(f"[::]:{free_port}")
    server.start()

    yield qwak_mocks

    server.stop(0)
