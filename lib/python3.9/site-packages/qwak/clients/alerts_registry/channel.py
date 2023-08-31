from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.monitoring.v0.alerting_channel_pb2 import (
    AlertingChannelDescription,
    AlertingChannelMetadata,
    AlertingChannelSpec,
)
from _qwak_proto.qwak.monitoring.v0.alerting_channel_pb2 import (
    ChannelConfiguration as ChannelConfigurationProto,
)
from _qwak_proto.qwak.monitoring.v0.alerting_channel_pb2 import (
    OpsgenieType,
    PagerDutyType,
    SecretStringValue,
    SlackType,
)
from qwak.exceptions import QwakException


@dataclass
class ChannelConfiguration(ABC):
    @abstractmethod
    def to_proto(self) -> "ChannelConfigurationProto":
        pass

    @classmethod
    def from_proto(cls, proto: "ChannelConfigurationProto") -> "ChannelConfiguration":
        channel_type: str = proto.WhichOneof("channel_type")

        if channel_type == "slack":
            return SlackChannel.from_proto(proto=proto.slack)
        if channel_type == "pagerduty":
            return PagerDutyChannel.from_proto(proto=proto.pagerduty)
        if channel_type == "opsgenie":
            return OpsGenieChannel.from_proto(proto=proto.opsgenie)

        raise QwakException(f"Got unsupported channel type: {channel_type}")


@dataclass
class Channel:
    name: str
    channel_conf: ChannelConfiguration

    def __post_init__(self):
        import re

        allowed_pattern = "^[a-z]+[a-z0-9_]*$"
        result = re.match(allowed_pattern, self.name)
        if not result:
            raise QwakException(
                "Channel name is limited to lower case letters and numbers, must start with a letter."
            )

    def to_proto(self) -> AlertingChannelDescription:
        metadata = AlertingChannelMetadata(name=self.name, id="")
        return AlertingChannelDescription(
            metadata=metadata,
            spec=AlertingChannelSpec(configuration=self.channel_conf.to_proto()),
        )

    @classmethod
    def from_proto(cls, proto: "AlertingChannelDescription") -> "Channel":
        channel_conf: ChannelConfiguration = ChannelConfiguration.from_proto(
            proto.spec.configuration
        )
        return cls(
            name=proto.metadata.name,
            channel_conf=channel_conf,
        )


@dataclass
class SlackChannel(ChannelConfiguration):
    api_url: str

    def to_proto(self) -> ChannelConfigurationProto:
        return ChannelConfigurationProto(
            slack=SlackType(api_url=SecretStringValue(value=self.api_url))
        )

    @classmethod
    def from_proto(cls, proto: "SlackType") -> "SlackChannel":
        return cls(api_url=proto.api_url.value)


@dataclass
class PagerDutyChannel(ChannelConfiguration):
    url: str
    routing_key: Optional[str] = None
    service_key: Optional[str] = None

    def __post_init__(self):
        if (not self.routing_key and not self.service_key) or (
            self.routing_key and self.service_key
        ):
            raise QwakException("Routing and service key are mutually exclusive.")

    def to_proto(self) -> ChannelConfigurationProto:
        return ChannelConfigurationProto(
            pagerduty=PagerDutyType(
                routing_key=SecretStringValue(value=self.routing_key),
                service_key=SecretStringValue(value=self.service_key),
                url=self.url,
            )
        )

    @classmethod
    def from_proto(cls, proto: "PagerDutyType") -> "PagerDutyChannel":
        return cls(
            routing_key=proto.routing_key.value,
            service_key=proto.service_key.value,
            url=proto.url,
        )


@dataclass
class OpsGenieChannel(ChannelConfiguration):
    api_key: str

    def to_proto(self) -> ChannelConfigurationProto:
        return ChannelConfigurationProto(
            opsgenie=OpsgenieType(
                api_key=SecretStringValue(value=self.api_key),
            )
        )

    @classmethod
    def from_proto(cls, proto: "OpsgenieType") -> "OpsGenieChannel":
        return cls(api_key=proto.api_key.value)
