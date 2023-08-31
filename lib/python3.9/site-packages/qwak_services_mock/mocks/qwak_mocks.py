import dataclasses

from qwak_services_mock.mocks import workspace_manager_service_mock
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


@dataclasses.dataclass
class QwakMocks:
    port: int
    build_orchestrator_build_api: BuildOrchestratorBuildApiMock
    build_orchestrator_service_api: BuildOrchestratorServiceApiMock
    internal_build_orchestrator_service: InternalBuildOrchestratorServiceMock
    build_management_service_mock: BuildsManagementServiceMock
    alert_manager_service_mock: AlertManagerServiceApiMock
    automation_management_service_mock: AutomationManagementServiceMock
    job_registry_service_mock: JobRegistryServiceApiMock
    project_manager_service_mock: ProjectManagerServiceMock
    kube_captain_service_mock: KubeCaptainServiceApiMock
    file_versioning_service_mock: FileVersioningServiceMock
    data_versioning_service_mock: DataVersioningServiceMock
    model_management_service_mock: ModelsManagementServiceMock
    logging_service_mock: LoggingServiceApiMock
    autoscaling_service_mock: AutoscalingServiceApiMock
    audience_api_mock: AudienceServiceApiMock
    self_service_user_service_mock: SelfServiceUserServiceMock
    analytics_api_mock: AnalyticsApiMock
    deployment_management_service_mock: DeploymentManagementServiceMock
    batch_job_manager_service: BatchJobManagerService
    ecosystem_client_mock: EcoSystemServiceMock
    user_application_instance_service_mock: UserApplicationInstanceServiceApiMock
    secret_service_mock: SecretServiceMock
    authentication_service_mock: AuthenticationServiceMock
    features_operator_service: FeaturesOperatorV3ServiceMock
    fs_data_sources_service: DataSourceServiceMock
    fs_entities_service: EntityServiceMock
    fs_feature_sets_service: FeatureSetServiceMock
    instance_templates_service: InstanceTemplateManagementServiceMock
    alerts_registry_service: AlertsRegistryServiceApiMock
    workspace_manager_service: workspace_manager_service_mock
    vector_serving_service: VectorServingServiceMock
    vector_collection_service: VectorCollectionManagementServiceMock
