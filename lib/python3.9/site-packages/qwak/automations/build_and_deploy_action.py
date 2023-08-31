from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from _qwak_proto.qwak.automation.v1.action_pb2 import (
    Action as ActionProto,
    AdvancedDeploymentOptions,
    BuildAndDeployAction,
    BuildMetricCondition,
    BuildSpec,
    CpuResources as CpuResourcesProto,
    DeploymentCondition as DeploymentConditionProto,
    DeploymentSize,
    DeploymentSpec,
    GitModelSource,
    GpuResources as GpuResourceProto,
    GpuType,
    Resources as ResourceProto,
)
from _qwak_proto.qwak.automation.v1.auto_scaling_pb2 import (
    AGGREGATION_TYPE_AVERAGE,
    AGGREGATION_TYPE_MAX,
    AGGREGATION_TYPE_MIN,
    AGGREGATION_TYPE_P50,
    AGGREGATION_TYPE_P90,
    AGGREGATION_TYPE_P95,
    AGGREGATION_TYPE_P99,
    AGGREGATION_TYPE_SUM,
    METRIC_TYPE_CPU,
    METRIC_TYPE_LATENCY,
    METRIC_TYPE_MEMORY,
    AutoScalingConfig as AutoScalingConfigProto,
    AutoScalingPrometheusTrigger as AutoScalingPrometheusTriggerProto,
    QuerySpec,
    ScaleTrigger as ScaleTriggerProto,
    Triggers as TriggersProto,
)
from _qwak_proto.qwak.user_application.common.v0.resources_pb2 import (
    ClientPodComputeResources,
    CpuResources as CommonCpuResourcesProto,
    GpuResources as CommonGpuResourcesProto,
    GpuType as CommonGpuTypeProto,
    PodComputeResourceTemplateSpec,
)
from qwak.automations.common import (
    Action,
    ThresholdDirection,
    get_memory_amount,
    map_common_memory_units,
    map_common_memory_units_proto,
    map_memory_units,
    map_memory_units_proto,
    map_proto_threshold_to_direction,
    map_threshold_direction_to_proto,
)


@dataclass
class DeploymentCondition(ABC):
    @abstractmethod
    def to_proto(self):
        # abstract method
        pass

    @staticmethod
    @abstractmethod
    def from_proto(message: DeploymentConditionProto):
        # abstract method
        pass


@dataclass
class AutoScaleTrigger(ABC):
    @abstractmethod
    def to_proto(self):
        # abstract method
        pass

    @staticmethod
    @abstractmethod
    def from_proto(message: ScaleTriggerProto):
        # abstract method
        pass


@dataclass
class BuildMetric(DeploymentCondition):
    metric_name: str = field(default="")
    direction: ThresholdDirection = field(default=ThresholdDirection.ABOVE)
    threshold: str = field(default="")

    def to_proto(self):
        return DeploymentConditionProto(
            build_metric=BuildMetricCondition(
                metric_name=self.metric_name,
                threshold=self.threshold,
                threshold_direction=map_threshold_direction_to_proto(self.direction),
            )
        )

    @staticmethod
    def from_proto(message: DeploymentConditionProto):
        return BuildMetric(
            metric_name=message.build_metric.metric_name,
            threshold=message.build_metric.threshold,
            direction=map_proto_threshold_to_direction(
                message.build_metric.threshold_direction
            ),
        )

    def __str__(self):
        return f"Metric Name: {self.metric_name}\tThreshold: {self.threshold}\tDirection: {self.direction.name}"


@dataclass
class Resources(ABC):
    @abstractmethod
    def to_proto(self):
        # abstract method
        pass

    @staticmethod
    @abstractmethod
    def from_proto(message: ResourceProto):
        # abstract method
        pass


@dataclass
class CpuResources(Resources):
    cpu_fraction: float = field(default=2)
    memory: str = field(default="2Gi")

    def to_proto(self):
        return ResourceProto(
            cpu_resources=CpuResourcesProto(
                cpu=self.cpu_fraction,
                memory_units=map_memory_units(memory=self.memory),
                memory_amount=get_memory_amount(self.memory),
            )
        )

    def to_common_proto(self):
        return CommonCpuResourcesProto(
            cpu=self.cpu_fraction,
            memory_units=map_common_memory_units(memory=self.memory),
            memory_amount=get_memory_amount(self.memory),
        )

    @staticmethod
    def from_proto(message: ResourceProto):
        return CpuResources(
            cpu_fraction=message.cpu_resources.cpu,
            memory=str(message.cpu_resources.memory_amount)
            + map_memory_units_proto(message.cpu_resources.memory_units),
        )

    @staticmethod
    def from_common_cpu_proto(message: CommonCpuResourcesProto):
        return CpuResources(
            cpu_fraction=message.cpu,
            memory=str(message.memory_amount)
            + map_common_memory_units_proto(message.memory_units),
        )

    def __str__(self):
        return f"CPU: {self.cpu_fraction}, Memory: {self.memory}"


@dataclass
class GpuResources(Resources):
    gpu_type: str = field(default=None)
    gpu_amount: int = field(default=None)

    def to_proto(self):
        return ResourceProto(
            gpu_resources=GpuResourceProto(
                gpu_type=self.gpu_type,
                gpu_amount=self.gpu_amount,
            )
        )

    def to_gpu_proto(self):
        return GpuResourceProto(
            gpu_type=self.gpu_type,
            gpu_amount=self.gpu_amount,
        )

    def to_common_gpu_proto(self):
        return CommonGpuResourcesProto(
            gpu_amount=self.gpu_amount, gpu_type=self.gpu_type
        )

    @staticmethod
    def from_proto(message: ResourceProto):
        return GpuResources(
            gpu_type=GpuType.Name(message.gpu_resources.gpu_type),
            gpu_amount=message.gpu_resources.gpu_amount,
        )

    @staticmethod
    def from_gpu_proto(message: GpuResourceProto):
        return GpuResources(
            gpu_type=GpuType.Name(message.gpu_type),
            gpu_amount=message.gpu_amount,
        )

    @staticmethod
    def from_common_gpu_proto(message: CommonGpuResourcesProto):
        return GpuResources(
            gpu_type=CommonGpuTypeProto.Name(message.gpu_type),
            gpu_amount=message.gpu_amount,
        )

    def __str__(self):
        return f"GPU Type: {self.gpu_type}, GPU Amount: {self.gpu_amount}"


@dataclass
class ClientResources(Resources):
    instance: str = field(default="")
    gpu_resources: GpuResources = field(default=None)
    cpu_resources: CpuResources = field(default=None)

    def to_proto(self):
        if self.instance:
            client_pod_compute_resources = ClientPodComputeResources(
                template_spec=PodComputeResourceTemplateSpec(template_id=self.instance)
            )
        elif self.gpu_resources:
            client_pod_compute_resources = ClientPodComputeResources(
                gpu_resources=self.gpu_resources.to_common_gpu_proto()
            )
        elif self.cpu_resources:
            client_pod_compute_resources = ClientPodComputeResources(
                cpu_resources=self.cpu_resources.to_common_proto()
            )
        else:
            client_pod_compute_resources = ClientPodComputeResources()

        return ResourceProto(client_pod_compute_resources=client_pod_compute_resources)

    @staticmethod
    def from_proto(message: ResourceProto):
        resources_type = message.client_pod_compute_resources.WhichOneof("resources")
        if resources_type == "template_spec":
            return ClientResources(
                instance=message.client_pod_compute_resources.template_spec.template_id
            )
        elif resources_type == "gpu_resources":
            return ClientResources(
                gpu_resources=GpuResources.from_common_gpu_proto(
                    message.client_pod_compute_resources.gpu_resources
                )
            )
        else:
            return ClientResources(
                cpu_resources=CpuResources.from_common_cpu_proto(
                    message.client_pod_compute_resources.cpu_resources
                )
            )


@dataclass
class BuildSpecifications:
    parameters: Dict[str, str] = field(default_factory=dict)
    git_uri: str = field(default=None)
    tags: List[str] = field(default_factory=list)
    git_access_token_secret: str = field(default=None)
    git_branch: str = field(default="main")
    main_dir: str = field(default="main")
    base_image: str = field(default="")
    assumed_iam_role: str = field(default="")
    resources: Resources = field(default_factory=CpuResources)
    dependency_file_path: str = field(default=None)
    env_vars: List[str] = field(default_factory=list)

    def to_proto(self):
        return BuildSpec(
            parameters=self.parameters,
            git_model_source=GitModelSource(
                git_uri=self.git_uri,
                git_credentials_secret_name=self.git_access_token_secret,
                git_branch=self.git_branch,
            ),
            main_dir=self.main_dir,
            tags=self.tags,
            resource=self.resources.to_proto(),
            base_image=self.base_image,
            assumed_iam_role=self.assumed_iam_role,
            dependency_file_path=self.dependency_file_path,
            env_vars=self.env_vars,
        )

    @staticmethod
    def from_proto(build_spec: BuildSpec):
        resources = map_resources_name_to_class(
            build_spec.resource.WhichOneof("resource")
        )
        return BuildSpecifications(
            parameters=build_spec.parameters,
            git_uri=build_spec.git_model_source.git_uri,
            git_access_token_secret=build_spec.git_model_source.git_credentials_secret_name,
            git_branch=build_spec.git_model_source.git_branch,
            tags=build_spec.tags,
            main_dir=build_spec.main_dir,
            base_image=build_spec.base_image,
            resources=resources.from_proto(build_spec.resource) if resources else None,
            dependency_file_path=build_spec.dependency_file_path,
            assumed_iam_role=build_spec.assumed_iam_role,
            env_vars=build_spec.env_vars,
        )

    def __str__(self):
        result = f"Git Uri:\t{self.git_uri}\n"
        result += f"Git Branch:\t{self.git_branch}\n"
        result += (
            f"Git Access Token Secret:\t{self.git_access_token_secret}\n"
            if self.git_access_token_secret
            else ""
        )
        result += f"Main Dir:\t{self.main_dir}\n" if self.main_dir != "main" else ""
        result += f"Parameters:\t{self.parameters}\n" if self.parameters else ""
        result += f"Tags:\t{self.tags}\n" if self.tags else ""
        result += (
            f"IAM Role:\t{self.assumed_iam_role}\n" if self.assumed_iam_role else ""
        )
        result += f"Base Image:\t{self.base_image}\n" if self.base_image else ""
        result += f"Resources:\n{self.resources}\n" if self.resources else ""
        result += f"Environment Variables:\n{self.env_vars}\n" if self.env_vars else ""
        return result


class MetricType(Enum):
    cpu = METRIC_TYPE_CPU
    latency = METRIC_TYPE_LATENCY
    memory = METRIC_TYPE_MEMORY


class AggregationType(Enum):
    avg = AGGREGATION_TYPE_AVERAGE
    max = AGGREGATION_TYPE_MAX
    min = AGGREGATION_TYPE_MIN
    sum = AGGREGATION_TYPE_SUM
    p50 = AGGREGATION_TYPE_P50
    p90 = AGGREGATION_TYPE_P90
    p95 = AGGREGATION_TYPE_P95
    p99 = AGGREGATION_TYPE_P99


@dataclass
class AutoScalingConfig:
    min_replica_count: int = field(default=None)
    max_replica_count: int = field(default=None)
    polling_interval: int = field(default=None)
    cool_down_period: int = field(default=None)
    triggers: List[AutoScaleTrigger] = field(default_factory=list)

    def to_proto(self):
        return AutoScalingConfigProto(
            min_replica_count=self.min_replica_count,
            max_replica_count=self.max_replica_count,
            polling_interval=self.polling_interval,
            cool_down_period=self.cool_down_period,
            triggers=TriggersProto(
                triggers=[trigger.to_proto() for trigger in self.triggers]
            ),
        )

    @staticmethod
    def from_proto(message: AutoScalingConfigProto):
        triggers = [
            map_autoscaling_trigger_name_to_class(
                trigger.WhichOneof("trigger_type")
            ).from_proto(trigger)
            for trigger in message.triggers.triggers
        ]
        return AutoScalingConfig(
            min_replica_count=message.min_replica_count,
            max_replica_count=message.max_replica_count,
            polling_interval=message.polling_interval,
            cool_down_period=message.cool_down_period,
            triggers=triggers,
        )

    def __str__(self):
        result = f"min replica count: {self.min_replica_count} \t max replica count: {self.max_replica_count} \t polling interval: {self.polling_interval} \t cool down period: {self.cool_down_period} \n"
        result += "Scale triggers: "
        for trigger in self.triggers:
            result += f" {trigger}"
        return result


@dataclass
class DeploymentSpecifications:
    number_of_http_server_workers: int = field(default=2)
    http_request_timeout_ms: int = field(default=5000)
    max_batch_size: int = field(default=1)
    daemon_mode: bool = field(default=False)
    variation_name: str = field(default="default")
    custom_iam_role_arn: str = field(default="")
    number_of_pods: int = field(default=2)
    cpu_fraction: float = field(default=2)
    memory: str = field(default="2Gi")
    gpu_resources: GpuResources = field(default=None)
    auto_scale_config: AutoScalingConfig = field(default_factory=AutoScalingConfig)
    environments: List[str] = field(default_factory=list)
    env_vars: List[str] = field(default_factory=list)
    deployment_timeout: int = field(default=0)
    instance: str = field(default="")

    def to_proto(self):
        deployment_size = self.__get_deployment_size_proto()
        return DeploymentSpec(
            selected_variation_name=self.variation_name,
            deployment_size=deployment_size,
            advanced_options=AdvancedDeploymentOptions(
                number_of_http_server_workers=self.number_of_http_server_workers,
                http_request_timeout_ms=self.http_request_timeout_ms,
                max_batch_size=self.max_batch_size,
                daemon_mode=self.daemon_mode,
                custom_iam_role_arn=self.custom_iam_role_arn,
                auto_scaling_config=self.auto_scale_config.to_proto(),
                deployment_process_timeout_limit=self.deployment_timeout,
            ),
            environments=self.environments,
            env_vars=self.env_vars,
        )

    @staticmethod
    def from_proto(message: DeploymentSpec):
        if (
            (
                message.deployment_size.client_pod_compute_resources.cpu_resources
                and message.deployment_size.client_pod_compute_resources.cpu_resources.cpu
                > 0
            )
            or (
                message.deployment_size.client_pod_compute_resources.gpu_resources
                and message.deployment_size.client_pod_compute_resources.gpu_resources.gpu_amount
                > 0
            )
            or (
                message.deployment_size.client_pod_compute_resources.template_spec
                and message.deployment_size.client_pod_compute_resources.template_spec.template_id
            )
        ):
            return DeploymentSpecifications.__from_proto_common_resources(message)
        else:
            return DeploymentSpecifications.__from_proto_deprecated_resources(message)

    def __str__(self):
        result = self.__get_deployment_resources_output()
        result += (
            f"Number of HTTP Workers: {self.number_of_http_server_workers}\t"
            f"HTTP Request Timeout: {self.http_request_timeout_ms}\t"
            f"Daemon Mode: {self.daemon_mode}\t"
            f"Max Batch Size: {self.max_batch_size}\n"
        )
        result += f"Environment Variables: {self.env_vars}\n"
        result += (
            f"Custom IAM Role: {self.custom_iam_role_arn}"
            if self.custom_iam_role_arn
            else ""
        )
        result += f"Auto scaling config: {self.auto_scale_config}"
        return result

    def __get_deployment_resources_output(self) -> str:
        result = f"Number of Pods: {self.number_of_pods}\t"
        if self.instance:
            result += f"Instance: {self.instance}\n"
        elif self.gpu_resources and self.gpu_resources.gpu_amount > 0:
            result += (
                f"GPU: {self.gpu_resources.gpu_amount}\t"
                f"Type: {self.gpu_resources.gpu_type}\n"
            )
        else:
            result += f"CPU: {self.cpu_fraction}\tMemory: {self.memory}\n"

        return result

    def __get_deployment_size_proto(self) -> DeploymentSize:
        if self.instance:
            client_pod_compute_resources = ClientPodComputeResources(
                template_spec=PodComputeResourceTemplateSpec(template_id=self.instance)
            )
        elif self.gpu_resources and self.gpu_resources.gpu_amount > 0:
            client_pod_compute_resources = ClientPodComputeResources(
                gpu_resources=CommonGpuResourcesProto(
                    gpu_amount=self.gpu_resources.gpu_amount,
                    gpu_type=self.gpu_resources.gpu_type,
                )
            )
        else:
            client_pod_compute_resources = ClientPodComputeResources(
                cpu_resources=CommonCpuResourcesProto(
                    cpu=self.cpu_fraction,
                    memory_amount=get_memory_amount(self.memory),
                    memory_units=map_common_memory_units(self.memory),
                )
            )
        return DeploymentSize(
            number_of_pods=self.number_of_pods,
            client_pod_compute_resources=client_pod_compute_resources,
            cpu=self.cpu_fraction,
            memory_units=map_memory_units(memory=self.memory),
            memory_amount=get_memory_amount(self.memory),
            gpu_resources=self.gpu_resources.to_gpu_proto()
            if self.gpu_resources
            else None,
        )

    @staticmethod
    def __from_proto_common_resources(message):
        return DeploymentSpecifications(
            variation_name=message.selected_variation_name,
            number_of_http_server_workers=message.advanced_options.number_of_http_server_workers,
            http_request_timeout_ms=message.advanced_options.http_request_timeout_ms,
            max_batch_size=message.advanced_options.max_batch_size,
            daemon_mode=message.advanced_options.daemon_mode,
            custom_iam_role_arn=message.advanced_options.custom_iam_role_arn,
            number_of_pods=message.deployment_size.number_of_pods,
            cpu_fraction=message.deployment_size.client_pod_compute_resources.cpu_resources.cpu,
            memory=str(
                message.deployment_size.client_pod_compute_resources.cpu_resources.memory_amount
            )
            + map_common_memory_units_proto(
                message.deployment_size.client_pod_compute_resources.cpu_resources.memory_units
            ),
            gpu_resources=GpuResources.from_common_gpu_proto(
                message.deployment_size.client_pod_compute_resources.gpu_resources
            )
            if (
                message.deployment_size.client_pod_compute_resources.gpu_resources
                and message.deployment_size.client_pod_compute_resources.gpu_resources.gpu_amount
                > 0
            )
            else None,
            instance=message.deployment_size.client_pod_compute_resources.template_spec.template_id
            if message.deployment_size.client_pod_compute_resources.template_spec.template_id
            else "",
            auto_scale_config=AutoScalingConfig.from_proto(
                message.advanced_options.auto_scaling_config
            ),
            environments=message.environments,
            env_vars=message.env_vars,
            deployment_timeout=message.advanced_options.deployment_process_timeout_limit,
        )

    @staticmethod
    def __from_proto_deprecated_resources(message):
        return DeploymentSpecifications(
            variation_name=message.selected_variation_name,
            number_of_http_server_workers=message.advanced_options.number_of_http_server_workers,
            http_request_timeout_ms=message.advanced_options.http_request_timeout_ms,
            max_batch_size=message.advanced_options.max_batch_size,
            daemon_mode=message.advanced_options.daemon_mode,
            custom_iam_role_arn=message.advanced_options.custom_iam_role_arn,
            number_of_pods=message.deployment_size.number_of_pods,
            cpu_fraction=message.deployment_size.cpu,
            memory=str(message.deployment_size.memory_amount)
            + map_memory_units_proto(message.deployment_size.memory_units),
            gpu_resources=GpuResources.from_gpu_proto(
                message.deployment_size.gpu_resources
            )
            if message.deployment_size.gpu_resources
            else None,
            auto_scale_config=AutoScalingConfig.from_proto(
                message.advanced_options.auto_scaling_config
            ),
            environments=message.environments,
            env_vars=message.env_vars,
            deployment_timeout=message.advanced_options.deployment_process_timeout_limit,
        )


@dataclass
class QwakBuildDeploy(Action):
    build_spec: BuildSpecifications
    deployment_condition: DeploymentCondition
    deployment_spec: DeploymentSpecifications

    def to_proto(self):
        return ActionProto(
            build_deploy=BuildAndDeployAction(
                build_spec=self.build_spec.to_proto(),
                deployment_spec=self.deployment_spec.to_proto(),
                deployment_condition=self.deployment_condition.to_proto(),
            )
        )

    @staticmethod
    def from_proto(message: ActionProto):
        deployment_condition = map_deployment_condition_name_to_class(
            message.build_deploy.deployment_condition.WhichOneof("condition")
        )
        return QwakBuildDeploy(
            build_spec=BuildSpecifications.from_proto(message.build_deploy.build_spec),
            deployment_spec=DeploymentSpecifications.from_proto(
                message.build_deploy.deployment_spec
            ),
            deployment_condition=deployment_condition.from_proto(
                message.build_deploy.deployment_condition
            )
            if deployment_condition
            else None,
        )

    def __str__(self):
        return f"Build Specifications:\n{self.build_spec}\nDeployment Specification:\n{self.deployment_spec}\nDeployment Condition:\n{self.deployment_condition}"


@dataclass
class AutoScaleQuerySpec:
    metric_type: str = field(default=None)
    aggregation_type: str = field(default=None)
    time_period: int = field(default=None)

    def to_proto(self):
        return QuerySpec(
            time_period=self.time_period,
            metric_type=MetricType[self.metric_type.lower()].value,
            aggregation_type=AggregationType[self.aggregation_type.lower()].value,
        )

    @staticmethod
    def from_proto(message: QuerySpec):
        return AutoScaleQuerySpec(
            metric_type=map_auto_scaling_metric_type_proto_to_name(message.metric_type),
            aggregation_type=map_aggregation_type_proto_to_name(
                message.aggregation_type
            ),
            time_period=message.time_period,
        )

    def __str__(self):
        return f"metric type: {self.metric_type}\taggregation type: {self.aggregation_type}\ttime period: {self.time_period}"


@dataclass
class AutoScalingPrometheusTrigger(AutoScaleTrigger):
    query_spec: AutoScaleQuerySpec = field(default=QuerySpec)
    threshold: int = field(default=None)

    def to_proto(self):
        return ScaleTriggerProto(
            prometheus_trigger=AutoScalingPrometheusTriggerProto(
                query_spec=self.query_spec.to_proto(), threshold=self.threshold
            )
        )

    @staticmethod
    def from_proto(message: ScaleTriggerProto):
        return AutoScalingPrometheusTrigger(
            threshold=message.prometheus_trigger.threshold,
            query_spec=AutoScaleQuerySpec.from_proto(
                message.prometheus_trigger.query_spec
            ),
        )

    def __str__(self):
        return f"threshold: {self.threshold} \t {self.query_spec}"


def map_resources_name_to_class(resource_name: str):
    mapping = {
        "cpu_resources": CpuResources,
        "gpu_resources": GpuResources,
        "client_pod_compute_resources": ClientResources,
    }
    return mapping.get(resource_name)


def map_deployment_condition_name_to_class(deployment_condition_name: str):
    mapping = {"build_metric": BuildMetric}
    return mapping.get(deployment_condition_name)


def map_auto_scaling_metric_type_proto_to_name(metric_type):
    mapping = {
        METRIC_TYPE_CPU: "cpu",
        METRIC_TYPE_LATENCY: "latency",
        METRIC_TYPE_MEMORY: "memory",
    }
    return mapping.get(metric_type)


def map_aggregation_type_proto_to_name(aggregation_type):
    mapping = {
        AGGREGATION_TYPE_AVERAGE: "avg",
        AGGREGATION_TYPE_MAX: "max",
        AGGREGATION_TYPE_MIN: "min",
        AGGREGATION_TYPE_SUM: "sum",
        AGGREGATION_TYPE_P50: "p50",
        AGGREGATION_TYPE_P90: "p90",
        AGGREGATION_TYPE_P95: "p95",
        AGGREGATION_TYPE_P99: "p99",
    }
    return mapping.get(aggregation_type)


def map_autoscaling_trigger_name_to_class(auto_scaling_trigger_name: str):
    mapping = {"prometheus_trigger": AutoScalingPrometheusTrigger}
    return mapping.get(auto_scaling_trigger_name)
