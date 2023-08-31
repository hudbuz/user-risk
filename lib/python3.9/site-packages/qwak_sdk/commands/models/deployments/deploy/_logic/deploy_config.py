from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from qwak_sdk.commands.audience._logic.config.v1.audience_config import AudienceConfig
from qwak_sdk.commands.auto_scalling._logic.config.config import AutoScaling
from qwak_sdk.commands.models.deployments.deploy._logic.get_latest_successful_build import (
    get_latest_successful_build_from_model,
)
from qwak_sdk.common.run_config import (
    ConfigCliMap,
    QwakConfigBase,
    YamlConfigMixin,
    validate_bool,
    validate_enum,
    validate_int,
    validate_list_of_strings,
    validate_string,
)
from qwak_sdk.common.run_config.utils import validate_float, validate_variations


class CompressionTypes(Enum):
    UNCOMPRESSED = "uncompressed"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class AutoOffsetReset(Enum):
    UNSET = "unset"
    LATEST = "latest"
    EARLIEST = "earliest"


class PurchaseOption(Enum):
    SPOT = "spot"
    ONDEMAND = "ondemand"


CONFIG_MAPPING: List[ConfigCliMap] = [
    # Stream
    # General
    ConfigCliMap("model_id", "model_id", validate_string, True),
    ConfigCliMap("build_id", "build_id", validate_string, False),
    ConfigCliMap("environment_name", "environments", validate_list_of_strings, False),
    # Kafka
    ConfigCliMap(
        "bootstrap_server", "stream.kafka.bootstrap_servers", validate_list_of_strings
    ),
    # Kafka - Consumer
    ConfigCliMap(
        "consumer_bootstrap_server",
        "stream.kafka.consumer.bootstrap_servers",
        validate_list_of_strings,
        False,
    ),
    ConfigCliMap(
        "consumer_topic", "stream.kafka.consumer.topic", validate_string, True
    ),
    ConfigCliMap(
        "consumer_group", "stream.kafka.consumer.group", validate_string, True
    ),
    ConfigCliMap(
        "consumer_auto_offset_reset",
        "stream.kafka.consumer.auto_offset_reset",
        validate_enum(AutoOffsetReset),
        False,
    ),
    ConfigCliMap(
        "consumer_timeout", "stream.kafka.consumer.timeout", validate_int, True
    ),
    ConfigCliMap(
        "consumer_max_batch_size",
        "stream.kafka.consumer.max_batch_size",
        validate_int,
        True,
    ),
    ConfigCliMap(
        "consumer_max_poll_latency",
        "stream.kafka.consumer.max_poll_latency",
        validate_float,
        True,
    ),
    # Kafka - Producer
    ConfigCliMap(
        "producer_bootstrap_server",
        "stream.kafka.producer.bootstrap_servers",
        validate_list_of_strings,
        False,
    ),
    ConfigCliMap(
        "producer_topic", "stream.kafka.producer.topic", validate_string, True
    ),
    ConfigCliMap(
        "producer_compression_type",
        "stream.kafka.producer.compression_type",
        validate_enum(CompressionTypes),
        False,
    ),
    ConfigCliMap("env_vars", "stream.env_vars", validate_list_of_strings),
    # Realtime
    ConfigCliMap("timeout", "realtime.timeout", validate_int, False),
    ConfigCliMap("workers", "realtime.workers", validate_int, False),
    ConfigCliMap("daemon_mode", "realtime.daemon_mode", validate_bool, False),
    ConfigCliMap("server_workers", "realtime.workers", validate_int, False),
    ConfigCliMap("max_batch_size", "realtime.max_batch_size", validate_int, False),
    ConfigCliMap(
        "deployment_timeout", "realtime.deployment_timeout", validate_int, False
    ),
    ConfigCliMap("variation_name", "realtime.variation_name", validate_string, False),
    ConfigCliMap(
        "variation_protected_state",
        "realtime.variation_protected_state",
        validate_bool,
        False,
    ),
    ConfigCliMap(
        "environment_name", "realtime.environments", validate_list_of_strings, False
    ),
    ConfigCliMap(
        "variations",
        "realtime.variations",
        validate_variations,
        False,
    ),
    ConfigCliMap("env_vars", "realtime.env_vars", validate_list_of_strings),
    # Batch
    # Resources
    ConfigCliMap("pods", "resources.pods", validate_int, False),
    ConfigCliMap("cpus", "resources.cpus", validate_int, False),
    ConfigCliMap("memory", "resources.memory", validate_int, False),
    ConfigCliMap("gpu_type", "resources.gpu_type", validate_string, False),
    ConfigCliMap("gpu_amount", "resources.gpu_amount", validate_int, False),
    ConfigCliMap("instance", "resources.instance_size", validate_string, False),
    # Permissions
    ConfigCliMap(
        "iam_role_arn", "advanced_options.iam_role_arn", validate_string, False
    ),
    ConfigCliMap(
        "purchase_option",
        "advanced_options.purchase_option",
        validate_enum(PurchaseOption),
        False,
    ),
]


@dataclass
class DeployConfig(YamlConfigMixin, QwakConfigBase):
    def _post_merge_cli(self):
        if self.stream.kafka.bootstrap_servers:
            if not self.stream.kafka.consumer.bootstrap_servers:
                self.stream.kafka.consumer.bootstrap_servers = (
                    self.stream.kafka.bootstrap_servers
                )
            if not self.stream.kafka.producer.bootstrap_servers:
                self.stream.kafka.producer.bootstrap_servers = (
                    self.stream.kafka.bootstrap_servers
                )
        if not self.build_id:
            self.build_id = get_latest_successful_build_from_model(self.model_id)

    @property
    def _config_mapping(self) -> List[ConfigCliMap]:
        return CONFIG_MAPPING

    @dataclass
    class Stream:
        @dataclass
        class Kafka:
            @dataclass
            class Consumer:
                bootstrap_servers: List[str] = field(default_factory=list)
                topic: str = field(default="")
                group: str = field(default="")
                auto_offset_reset: str = field(default=AutoOffsetReset.LATEST.name)
                timeout: int = field(default=20)
                max_batch_size: int = field(default=1)
                max_poll_latency: float = field(default=1.0)

            @dataclass
            class Producer:
                bootstrap_servers: List[str] = field(default_factory=list)
                topic: str = field(default="")
                compression_type: str = field(default=CompressionTypes.GZIP.name)

            bootstrap_servers: List[str] = field(default_factory=list)
            consumer: Consumer = field(default_factory=Consumer)
            producer: Producer = field(default_factory=Producer)

        kafka: Kafka = field(default_factory=Kafka)
        env_vars: Optional[Union[List[str], Dict[str, str], Tuple]] = field(
            default_factory=list
        )

    @dataclass
    class Realtime:
        @dataclass
        class VariationConfig:
            @dataclass
            class TrafficConfig:
                percentage: int = field(default=100)
                shadow: bool = field(default=False)

            name: str = field(default="default")
            traffic: TrafficConfig = field(default_factory=TrafficConfig)

        timeout: int = field(default=6000)
        workers: int = field(default=2)
        deployment_timeout: int = field(default=0)
        daemon_mode: bool = field(default=True)
        max_batch_size: int = field(default=1)
        variation_name: str = field(default="")
        variations: List[VariationConfig] = field(default_factory=list)
        audiences: List[AudienceConfig] = field(default_factory=list)
        environments: List[str] = field(default_factory=list)
        fallback_variation: str = field(default="")
        variation_protected_state: bool = field(default=False)
        env_vars: Optional[Union[List[str], Dict[str, str], Tuple]] = field(
            default_factory=list
        )

    @dataclass
    class Batch:
        pass

    @dataclass
    class Resources:
        pods: int = field(default=1)
        cpus: int = field(default=2)
        memory: int = field(default=512)
        gpu_type: str = field(default=None)
        gpu_amount: int = field(default=0)
        instance_size: str = field(default="")

    @dataclass
    class AdvancedOptions:
        iam_role_arn: str = field(default=None)
        purchase_option: str = field(default=PurchaseOption.SPOT.value)

    model_id: str = field(default="")
    build_id: str = field(default="")
    environments: List[str] = field(default_factory=list)

    stream: Stream = field(default_factory=Stream)
    auto_scaling: AutoScaling = field(default=None)
    realtime: Realtime = field(default_factory=Realtime)
    batch: Batch = field(default_factory=Batch)
    resources: Resources = field(default_factory=Resources)
    advanced_options: AdvancedOptions = field(default_factory=AdvancedOptions)
