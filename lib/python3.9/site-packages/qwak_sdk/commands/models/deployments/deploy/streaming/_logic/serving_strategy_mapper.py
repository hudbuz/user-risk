from __future__ import annotations

from _qwak_proto.qwak.deployment.deployment_pb2 import (
    KafkaConfig,
    ServingStrategy,
    StreamConfig,
)

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)

DEFAULT_VARIATION_NAME = "default"


def create_streaming_serving_strategy_from_deploy_config(
    deploy_config: DeployConfig,
) -> ServingStrategy:
    return ServingStrategy(
        stream_config=StreamConfig(
            kafka=KafkaConfig(
                consumer=KafkaConfig.Consumer(
                    bootstrap_server=deploy_config.stream.kafka.consumer.bootstrap_servers,
                    topic=deploy_config.stream.kafka.consumer.topic,
                    group=deploy_config.stream.kafka.consumer.group,
                    timeout=deploy_config.stream.kafka.consumer.timeout,
                    auto_offset_type=deploy_config.stream.kafka.consumer.auto_offset_reset.upper(),
                    max_batch_size=deploy_config.stream.kafka.consumer.max_batch_size,
                    max_poll_latency=deploy_config.stream.kafka.consumer.max_poll_latency,
                ),
                producer=KafkaConfig.Producer(
                    bootstrap_server=deploy_config.stream.kafka.producer.bootstrap_servers,
                    topic=deploy_config.stream.kafka.producer.topic,
                    compression_type=deploy_config.stream.kafka.producer.compression_type.upper(),
                ),
            )
        )
    )
