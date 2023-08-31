from datetime import timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from _qwak_proto.qwak.builds.builds_pb2 import DESCRIPTOR
from qwak.clients.analytics.client import AnalyticsEngineClient
from qwak.clients.automation_management.client import AutomationsManagementClient
from qwak.clients.build_management.client import BuildsManagementClient
from qwak.clients.build_orchestrator.client import BuildFilter, BuildOrchestratorClient
from qwak.clients.deployment.client import DeploymentManagementClient
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.clients.model_management.client import ModelsManagementClient
from qwak.clients.project.client import ProjectsManagementClient
from qwak.exceptions import QwakException
from qwak.qwak_client.builds.build import Build
from qwak.qwak_client.builds.filters.metric_filter import MetricFilter
from qwak.qwak_client.builds.filters.parameter_filter import ParameterFilter
from qwak.qwak_client.deployments.deployment import Deployment, EnvironmentAudienceRoute
from qwak.qwak_client.models.model import Model
from qwak.qwak_client.models.model_metadata import ModelMetadata
from qwak.qwak_client.projects.project import Project

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pass


class QwakClient:
    """
    Comprehensive Qwak client to manage operations performed against the Qwak system.
    Acts as a wrapper and single entry point for the different Qwak clients - BatchClient,
    DeploymentClient, and more.
    """

    @lru_cache(maxsize=1)
    def _get_project_management(self):
        return ProjectsManagementClient()

    @lru_cache(maxsize=1)
    def _get_model_management(self):
        return ModelsManagementClient()

    @lru_cache(maxsize=1)
    def _get_build_management(self):
        return BuildsManagementClient()

    @lru_cache(maxsize=1)
    def _get_build_orchestrator(self):
        return BuildOrchestratorClient()

    @lru_cache(maxsize=1)
    def _get_deployment_management(self):
        return DeploymentManagementClient()

    @lru_cache(maxsize=1)
    def _get_feature_registry_client(self):
        return FeatureRegistryClient()

    @lru_cache(maxsize=1)
    def _get_automations_client(self):
        return AutomationsManagementClient()

    @lru_cache(maxsize=1)
    def _get_analytics_engine(self):
        return AnalyticsEngineClient()

    def get_latest_build(
        self, model_id: str, build_status: str = "SUCCESSFUL"
    ) -> Optional[str]:
        """
        Get the latest build by its model ID.
        Optionally gets a build_status, by default filters on 'SUCCESSFUL'

        Args:
            model_id (str): The model ID
            build_status (str): build statuses to filter on. Valid values are 'SUCCESSFUL', 'IN_PROGRESS', 'FAILED'

        Returns:
             str: The build ID of the latest build according to the build status. None if no builds match the filter.

        """
        if (
            build_status
            not in DESCRIPTOR.enum_types_by_name["BuildStatus"].values_by_name
        ):
            raise QwakException(
                f"Invalid build status {build_status}. Valid options are 'SUCCESSFUL', 'IN_PROGRESS', 'FAILED'"
            )

        model = self._get_model_management().get_model(model_id=model_id)
        model_uuid = model.uuid

        builds = self._get_build_management().list_builds(model_uuid=model_uuid).builds
        if not builds:
            return None

        builds = [
            build
            for build in builds
            if build.build_status
            == DESCRIPTOR.enum_types_by_name["BuildStatus"]
            .values_by_name[build_status]
            .number
        ]
        builds.sort(key=lambda r: r.created_at.seconds)

        return builds[-1].build_spec.build_id if builds else None

    def get_builds_by_tags(
        self,
        model_id: str,
        tags: List[str],
        match_any: bool = True,
        include_extra_tags: bool = True,
    ) -> List[Build]:
        """
        Get builds by its model ID and explicit tags

        Args:
            model_id (str): The model ID
            tags (List[str]): List of tags to filter by
            match_any (bool): Whether matching any tag is enough to return a build (default: True)
            include_extra_tags (bool): Whether builds are allowed to have more tags than the ones specified in the tags list (default: True)

        Returns:
             List[Build]: List of builds that contains the requested tags.

        """
        model = self._get_model_management().get_model(model_id=model_id)
        model_uuid = model.uuid

        builds_proto = (
            self._get_build_orchestrator()
            .list_builds(
                model_uuid=model_uuid,
                build_filter=BuildFilter(
                    tags=tags,
                    require_all_tags=not match_any,
                    include_extra_tags=include_extra_tags,
                ),
            )
            .build
        )

        builds = [Build.from_proto(build) for build in builds_proto]

        return builds if builds else None

    def get_build(self, build_id: str) -> Build:
        """
        Get builds by its build ID

        Args:
            build_id (str): The build ID

        Returns:
             Build: The requested build.

        """
        build_proto = self._get_build_orchestrator().get_build(build_id).build

        build = Build.from_proto(build_proto)

        if build:
            model_proto = self._get_model_management().get_model_by_uuid(
                model_uuid=build_proto.model_uuid
            )
            build.model_id = model_proto.model_id

        return build if build else None

    def list_builds(
        self,
        model_id: str,
        tags: List[str],
        filters: List[Union[MetricFilter, ParameterFilter]],
    ) -> List[Build]:
        """
        List builds by its model ID and explicit filters

        Args:
            model_id (str): The model ID
            tags (List[str]): List of tags to filter by
            filters (List[str]): List of metric and parameter filters

        Returns:
             List[Build]: List of builds that contains the requested filters.

        """
        model = self._get_model_management().get_model(model_id=model_id)
        model_uuid = model.uuid

        metric_filters = list()
        parameter_filters = list()

        for build_filter in filters:
            if isinstance(build_filter, MetricFilter):
                metric_filters.append(MetricFilter.to_proto(build_filter))
            else:
                parameter_filters.append(ParameterFilter.to_proto(build_filter))

        builds_proto = (
            self._get_build_orchestrator()
            .list_builds(
                model_uuid=model_uuid,
                build_filter=BuildFilter(
                    tags=tags,
                    metric_filters=metric_filters,
                    parameter_filters=parameter_filters,
                ),
            )
            .build
        )

        return [Build.from_proto(build) for build in builds_proto]

    def set_tag(self, build_id: str, tag: str) -> None:
        """
        Assign a tag to an existing build

        Args:
            build_id (str): The build ID
            tag (str): The tag to assign

        """
        self._get_build_orchestrator().register_tags(build_id=build_id, tags=[tag])

    def set_tags(self, build_id: str, tags: List[str]) -> None:
        """
        Assign a list of tags to an existing build

        Args:
            build_id (str): The build ID
            tags (List[str]): List of tags to assign
        """
        self._get_build_orchestrator().register_tags(build_id=build_id, tags=tags)

    def create_project(self, project_name: str, project_description: str) -> str:
        """
        Create project

        Args:
            project_name (str): The requested name
            project_description (str): The requested description

        Returns:
             str: The project ID of the newly created project

        """
        project = self._get_project_management().create_project(
            project_name=project_name, project_description=project_description
        )

        return project.project.project_id

    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get model by its project ID

        Args:
            project_id (str): The model ID

        Returns:
             Optional[Project]: Project by ID.

        """
        return Project.from_proto(
            self._get_project_management().get_project(project_id)
        )

    def list_projects(self) -> List[Project]:
        """
        List projects

        Returns:
             List[Project]: List of projects

        """
        projects_proto = self._get_project_management().list_projects().projects

        return [Project.from_proto(project) for project in projects_proto]

    def delete_project(self, project_id: str) -> None:
        """
        Delete project by its project ID

        Args:
            project_id (str): The project ID
        """
        return self._get_project_management().delete_project(project_id=project_id)

    def create_model(
        self, project_id: str, model_name: str, model_description: str
    ) -> str:
        """
        Create model

        Args:
            project_id (str): The project ID to associate the model
            model_name (str): The requested name
            model_description (str): The requested description

        Returns:
             str: The model ID of the newly created project

        """
        model = self._get_model_management().create_model(
            project_id=project_id,
            model_name=model_name,
            model_description=model_description,
        )

        return model.model_id

    def get_model(self, model_id: str) -> Optional[Model]:
        """
        Get model by its model ID

        Args:
            model_id (str): The model ID

        Returns:
             Optional[Model]: Model by ID.

        """
        return Model.from_proto(
            self._get_model_management().get_model(model_id=model_id)
        )

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """
        Get model metadata by its model ID

        Args:
            model_id (str): The model ID

        Returns:
            Model metadata by ID.
        """
        service_response = self._get_model_management().get_model_metadata(model_id)
        model_metadata_from_service = service_response.model_metadata
        model_metadata = ModelMetadata()

        model_metadata.model = Model.from_proto(model_metadata_from_service.model)
        model_metadata.deployed_builds = [
            Build.from_builds_management(build)
            for build in model_metadata_from_service.build
        ]
        model_metadata.deployments = [
            Deployment.from_proto(deployment)
            for deployment in model_metadata_from_service.deployment_details
        ]

        audience_routes = []
        for (
            env_id,
            env,
        ) in model_metadata_from_service.audience_routes_grouped_by_environment.items():
            for route in env.audience_routes:
                audience_routes.append(
                    EnvironmentAudienceRoute.from_proto(env_id, route)
                )
        model_metadata.audience_routes = audience_routes

        return model_metadata

    def delete_model(self, model_id: str) -> None:
        """
        Delete model by its model ID

        Args:
            model_id (str): The model ID
        """
        return self._get_model_management().delete_model(
            project_id=self.get_model(model_id).project_id, model_id=model_id
        )

    def get_deployed_build_id_per_environment(
        self, model_id: str
    ) -> Optional[Dict[str, str]]:
        """
        Get deployed build ID per environment by its model ID

        Args:
            model_id (str): The model ID

        Returns:
             Dict[str, str]: Map environment to deployed build ID. None if the model is not deployed.

        """
        model = self._get_model_management().get_model(model_id=model_id)
        model_uuid = model.uuid

        deployment_details = self._get_deployment_management().get_deployment_details(
            model_id=model_id, model_uuid=model_uuid
        )

        if not deployment_details:
            return None

        environment_to_deployment_details = (
            deployment_details.environment_to_deployment_details
        )
        env_to_deployed_build_id = {}
        for (
            env_id,
            environment_deployment_details_message,
        ) in environment_to_deployment_details.items():
            env_to_deployed_build_id[
                env_id
            ] = environment_deployment_details_message.deployments_details[0].build_id

        return env_to_deployed_build_id

    def run_analytics_query(
        self, query: str, timeout: timedelta = None
    ) -> "pd.DataFrame":
        """
        Runs a Qwak Analytics Query and returns the Pandas DataFrame with the results.

        Args:
            query (str): The query to run
            timeout (timedelta): The timeout for the query - optional.
                If not specified, the function will wait indefinitely.

        Returns:
            pd.DataFrame: The results of the query
        """
        try:
            import pandas as pd
        except ImportError:
            raise QwakException(
                "Missing Pandas dependency required for running an analytics query."
            )
        client = self._get_analytics_engine()
        url = client.get_analytics_data(query=query, timeout=timeout)
        return pd.read_csv(url)

    def delete_feature_set(self, feature_set_name: str):
        """
        Delete a feature set

        Args:
            feature_set_name (str): The feature set name
        """
        client = self._get_feature_registry_client()
        feature_set_def = client.get_feature_set_by_name(
            feature_set_name=feature_set_name
        )
        if feature_set_def:
            client.delete_feature_set(
                feature_set_def.feature_set.feature_set_definition.feature_set_id
            )
        else:
            raise QwakException(f"Feature set named '{feature_set_name}' not found")

    def delete_data_source(self, data_source_name: str):
        """
        Delete a data source

        Args:
            data_source_name (str): The data source name
        """
        client = self._get_feature_registry_client()
        data_source_def = client.get_data_source_by_name(
            data_source_name=data_source_name
        )
        if data_source_def:
            client.delete_data_source(
                data_source_def.data_source.data_source_definition.data_source_id
            )
        else:
            raise QwakException(f"Data source named '{data_source_name}' not found")

    def delete_entity(self, entity_name: str):
        """
        Delete an entity_name

        Args:
            entity_name (str): The entity name
        """
        client = self._get_feature_registry_client()
        entity_def = client.get_entity_by_name(entity_name=entity_name)
        if entity_def:
            client.delete_entity(entity_def.entity.entity_definition.entity_id)
        else:
            raise QwakException(f"Entity named '{entity_name}' not found")

    def trigger_batch_feature_set(self, feature_set_name):
        """
        Triggers a batch feature set ingestion job immediately.

        Args:
            feature_set_name (str): The feature set name
        """
        client = self._get_feature_registry_client()
        client.run_feature_set(feature_set_name=feature_set_name)

    def trigger_automation(self, automation_name):
        """
        Triggers an automation immediately.

        Args:
            automation_name (str): The automation name
        """
        client = self._get_automations_client()
        automation = client.get_automation_by_name(automation_name=automation_name)
        if not automation:
            raise QwakException(
                f"Could not find automation with given name '{automation_name}'"
            )
        client.run_automation(automation_id=automation.id)
