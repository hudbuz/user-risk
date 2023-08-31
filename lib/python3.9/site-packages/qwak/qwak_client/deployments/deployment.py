from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List

from _qwak_proto.qwak.audience.v1.audience_pb2 import (
    AudienceRoutesEntry as AudienceProto,
)
from _qwak_proto.qwak.audience.v1.audience_pb2 import Route as RouteProto
from _qwak_proto.qwak.auto_scaling.v1.auto_scaling_pb2 import (
    AutoScalingConfig as AutoScalingProto,
)
from _qwak_proto.qwak.deployment.deployment_pb2 import (
    DeploymentDetails as DeploymentProto,
)
from _qwak_proto.qwak.deployment.deployment_pb2 import (
    RealTimeConfig as RealTimeConfigProto,
)
from _qwak_proto.qwak.deployment.deployment_pb2 import (
    ServingStrategy as ServingStrategyProto,
)
from _qwak_proto.qwak.deployment.deployment_pb2 import StreamConfig as StreamConfigProto


@dataclass
class TrafficSpec:
    percentage: int = field(default=None)
    is_shadow: bool = field(default=None)


@dataclass
class Variation:
    name: str = field(default=None)
    environment_id: str = field(default=None)
    traffic_spec: TrafficSpec = field(default=None)


@dataclass
class Route:
    variation_name: str = field(default=None)
    weight: int = field(default=None)
    shadow: bool = field(default=None)
    model_id: str = field(default=None)
    environment_id: str = field(default=None)
    audience_id: str = field(default=None)


@dataclass
class AudienceRoute:
    audience_id: str = field(default=None)
    audience_name: str = field(default=None)
    routes: List[Route] = field(default_factory=list)
    order: int = field(default=None)


@dataclass
class EnvironmentAudienceRoute(AudienceRoute):
    environment_id: str = field(default=None)

    @staticmethod
    def from_proto(environment_id: str, proto: AudienceProto):
        return EnvironmentAudienceRoute(
            audience_id=proto.audience_id,
            audience_name=proto.audience_name,
            routes=Deployment.parse_routes(proto.routes),
            order=proto.order,
            environment_id=environment_id,
        )


class Metric(Enum):
    NOT_VALID = 0
    CPU = 1
    MEMORY = 2
    LATENCY = 3
    GPU = 4
    ERROR_RATE = 5
    THROUGHPUT = 6


class Aggregation(Enum):
    NOT_VALID = 0
    MIN = 1
    MAX = 2
    AVERAGE = 3
    SUM = 4


@dataclass
class QuerySpec:
    metric: Metric = field(default=None)
    aggregation: Aggregation = field(default=None)
    time_period: int = field(default=None)
    error_code: str = field(default=None)


class Triggers(ABC):
    pass


@dataclass
class PrometheusTrigger(Triggers):
    query_spec: QuerySpec = field(default=None)
    threshold: int = field(default=None)


@dataclass
class AutoScalingConfig:
    min_replica_count: int = field(default=None)
    max_replica_count: int = field(default=None)
    pooling_interval: int = field(default=None)
    cool_down_period: int = field(default=None)
    triggers: List[Triggers] = field(default_factory=list)


@dataclass
class TrafficConfig:
    selected_variation_name: str = field(default=None)
    variations: List[Variation] = field(default_factory=list)
    audiences: List[AudienceRoute] = field(default_factory=list)
    fallback_variation: str = field(default=None)


class ServingConfig(ABC):
    pass


@dataclass
class RealTimeConfig(ServingConfig):
    traffic_config: TrafficConfig = field(default=None)
    autoscaling_config: AutoScalingConfig = field(default=None)


@dataclass
class BatchConfig(ServingConfig):
    pass


class AutoOffSet(Enum):
    UNKNOWN = 0
    EARLIEST = 1
    LATEST = 2


@dataclass
class Consumer:
    bootstrap_server: List[str] = field(default_factory=list)
    topic: str = field(default=None)
    group: str = field(default=None)
    timeout: int = field(default=None)
    auto_offset: AutoOffSet = field(default=None)
    max_batch_size: int = field(default=None)
    max_poll_latency: float = field(default=None)


class Compression(Enum):
    UNKNOWN = 0
    UNCOMPRESSED = 1
    GZIP = 2
    SNAPPY = 3
    LZ4 = 4
    ZSTD = 5


@dataclass
class Producer:
    bootstrap_server: List[str] = field(default_factory=list)
    topic: str = field(default=None)
    compression: Compression = field(default=None)


@dataclass
class KafkaConfig:
    consumer: Consumer = field(default=None)
    producer: Producer = field(default=None)
    workers: int = field(default=None)


@dataclass
class StreamConfig(ServingConfig):
    kafka: KafkaConfig = field(default=None)


class BuildStatus(Enum):
    INVALID = 0
    IN_PROGRESS = 1
    SUCCESSFUL = 2
    FAILED = 3
    REMOTE_BUILD_INITIALIZING = 4
    REMOTE_BUILD_CANCELLED = 5
    REMOTE_BUILD_TIMED_OUT = 6
    REMOTE_BUILD_UNKNOWN = 7
    SYNCING_ENVIRONMENTS = 8
    FINISHED_SYNCING = 9


class MemoryUnit(Enum):
    UNKNOWN = 0
    MIB = 1
    GIB = 2


class KubeDeploymentType(Enum):
    UNDEFINED = 0
    ONLINE = 1
    STREAM = 2
    BATCH = 3


class GpuType(Enum):
    INVALID_GPU = 0
    ONLINE = 1
    STREAM = 2
    BATCH = 3


@dataclass
class Deployment:
    build_id: str = field(default=None)
    number_of_pods: int = field(default=None)
    cpu_fraction: float = field(default=None)
    memory_amount: int = field(default=None)
    memory_units: MemoryUnit = field(default=MemoryUnit.UNKNOWN)
    number_of_workers: int = field(default=None)
    http_request_timeout_ms: int = field(default=None)
    kube_deployment_type: KubeDeploymentType = field(
        default=KubeDeploymentType.UNDEFINED
    )
    serving_config: ServingConfig = field(default=None)
    variation: Variation = field(default=None)
    custom_iam_role_arn: str = field(default=None)
    max_batch_size: int = field(default=None)
    gpu_type: GpuType = field(default=GpuType.INVALID_GPU)
    gpu_amount: int = field(default=None)
    environment_id: str = field(default=None)
    available_replicas: int = field(default=None)
    deployment_process_timeout_limit: int = field(default=None)
    daemon_mode: bool = field(default=None)
    purchase_option: str = field(default=None)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    last_deployed: datetime = field(default=datetime.now())

    @staticmethod
    def parse_routes(proto: List[RouteProto]) -> List[Route]:
        routes = []
        for proto_route in proto:
            route = Route(
                variation_name=proto_route.variation_name,
                weight=proto_route.weight,
                shadow=proto_route.shadow,
                model_id=proto_route.model_id,
                environment_id=proto_route.environment_id,
                audience_id=proto_route.audience_id,
            )
            routes.append(route)
        return routes

    @staticmethod
    def parse_audience_routes(proto: List[RouteProto]) -> List[AudienceRoute]:
        routes = []
        for proto_route in proto:
            route = AudienceRoute(
                audience_id=proto_route.audience_id,
                audience_name=proto_route.audience_name,
                routes=Deployment.parse_routes(proto_route.routes),
                order=proto_route.order,
            )
            routes.append(route)
        return routes

    @staticmethod
    def parse_autoscaling(proto: AutoScalingProto) -> AutoScalingConfig:
        triggers = []
        for proto_trigger in proto.triggers.triggers:
            trigger = PrometheusTrigger(
                query_spec=QuerySpec(
                    metric=Metric(
                        proto_trigger.prometheus_trigger.query_spec.metric_type
                    ),
                    aggregation=Aggregation(
                        proto_trigger.prometheus_trigger.query_spec.aggregation_type
                    ),
                    time_period=proto_trigger.prometheus_trigger.query_spec.time_period,
                    error_code=proto_trigger.prometheus_trigger.query_spec.error_code,
                ),
                threshold=proto_trigger.prometheus_trigger.threshold,
            )
            triggers.append(trigger)
        return AutoScalingConfig(
            min_replica_count=proto.min_replica_count,
            max_replica_count=proto.max_replica_count,
            pooling_interval=proto.polling_interval,
            cool_down_period=proto.cool_down_period,
            triggers=triggers,
        )

    @staticmethod
    def parse_realtime_config(proto: RealTimeConfigProto) -> RealTimeConfig:
        variations = []
        for proto_variation in proto.traffic_config.variations:
            variation = Variation(
                name=proto_variation.name,
                environment_id=proto_variation.environment_id,
                traffic_spec=TrafficSpec(
                    percentage=proto_variation.traffic.percentage,
                    is_shadow=proto_variation.traffic.is_shadow,
                ),
            )
            variations.append(variation)
        audiences = []
        for proto_audience in proto.traffic_config.audience_routes_entries:
            audience = AudienceRoute(
                audience_id=proto_audience.audience_id,
                audience_name=proto_audience.audience_name,
                routes=Deployment.parse_routes(proto_audience.routes),
                order=proto_audience.order,
            )
            audiences.append(audience)

        autoscaling = Deployment.parse_autoscaling(proto.auto_scaling_config)
        return RealTimeConfig(
            traffic_config=TrafficConfig(
                variations=variations,
                audiences=audiences,
                fallback_variation=proto.traffic_config.fallback_variation,
            ),
            autoscaling_config=autoscaling,
        )

    @staticmethod
    def parse_streaming_config(proto_config: StreamConfigProto) -> StreamConfig:
        return StreamConfig(
            kafka=KafkaConfig(
                consumer=Consumer(
                    bootstrap_server=list(proto_config.kafka.consumer.bootstrap_server),
                    topic=proto_config.kafka.consumer.topic,
                    group=proto_config.kafka.consumer.group,
                    timeout=proto_config.kafka.consumer.timeout,
                    auto_offset=AutoOffSet(proto_config.kafka.consumer.auto_offset_type)
                    if proto_config.kafka.consumer.auto_offset_type
                    else None,
                    max_batch_size=proto_config.kafka.consumer.max_batch_size,
                    max_poll_latency=proto_config.kafka.consumer.max_poll_latency,
                ),
                producer=Producer(
                    bootstrap_server=list(proto_config.kafka.producer.bootstrap_server),
                    topic=proto_config.kafka.producer.topic,
                    compression=Compression(
                        proto_config.kafka.producer.compression_type
                    )
                    if proto_config.kafka.producer.compression_type
                    else None,
                ),
                workers=proto_config.kafka.workers,
            )
        )

    @staticmethod
    def parse_serving_strategy(proto: ServingStrategyProto) -> ServingConfig:
        which = proto.WhichOneof("Strategy")
        if which == "realtime_config":
            return Deployment.parse_realtime_config(proto.realtime_config)
        elif which == "batch_config":
            return BatchConfig()
        elif which == "stream_config":
            return Deployment.parse_streaming_config(proto.stream_config)
        else:
            raise ValueError(f"Unknown serving strategy: {which}")

    @staticmethod
    def from_proto(proto: DeploymentProto):
        variation = Variation(
            name=proto.variation.name,
            environment_id=proto.variation.environment_id,
            traffic_spec=TrafficSpec(
                percentage=proto.variation.traffic.percentage,
                is_shadow=proto.variation.traffic.is_shadow,
            ),
        )
        return Deployment(
            build_id=proto.build_id,
            number_of_pods=proto.number_of_pods,
            cpu_fraction=proto.cpu_fraction,
            memory_amount=proto.memory_amount,
            memory_units=MemoryUnit(proto.memory_units),
            number_of_workers=proto.number_of_workers,
            http_request_timeout_ms=proto.http_request_timeout_ms,
            kube_deployment_type=KubeDeploymentType(proto.kube_deployment_type),
            serving_config=Deployment.parse_serving_strategy(proto.serving_strategy),
            variation=variation,
            custom_iam_role_arn=proto.custom_iam_role_arn,
            max_batch_size=proto.max_batch_size,
            gpu_type=GpuType(proto.gpu_type),
            gpu_amount=proto.gpu_amount,
            environment_id=proto.environment_id,
            available_replicas=proto.available_replicas,
            daemon_mode=bool(proto.daemon_mode.value),
            deployment_process_timeout_limit=proto.deployment_process_timeout_limit,
            purchase_option=proto.purchase_option,
            environment_variables=dict(proto.environment_variables),
            last_deployed=datetime.fromtimestamp(
                proto.last_deployed.seconds + proto.last_deployed.nanos / 1e9
            ),
        )
