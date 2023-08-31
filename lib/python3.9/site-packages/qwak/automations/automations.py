from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from _qwak_proto.qwak.automation.v1.automation_pb2 import (
    Automation as AutomationProto,
    AutomationAudit as AutomationAuditProto,
    AutomationSpec as AutomationSpecProto,
)
from _qwak_proto.qwak.automation.v1.common_pb2 import (
    Metric as MetricProto,
    SqlMetric as SqlMetricProto,
)
from _qwak_proto.qwak.automation.v1.notification_pb2 import (
    CustomWebhook as CustomWebhookProto,
    HttpMethodType as HttpMethodTypeProto,
    Notification as NotificationProto,
    PostSlackNotification as PostSlackNotificationProto,
)
from _qwak_proto.qwak.automation.v1.trigger_pb2 import (
    MetricBasedTrigger as MetricBasedTriggerProto,
    NoneTrigger as NoneTriggerProto,
    OnBoardingTrigger as OnBoardingTriggerProto,
    ScheduledTrigger as ScheduledTriggerProto,
    Trigger as TriggerProto,
)
from google.protobuf.timestamp_pb2 import Timestamp
from qwak.automations.batch_execution_action import BatchExecution
from qwak.automations.build_and_deploy_action import QwakBuildDeploy
from qwak.automations.common import (
    Action,
    ThresholdDirection,
    map_proto_threshold_to_direction,
    map_threshold_direction_to_proto,
)


@dataclass
class Trigger(ABC):
    def to_proto(self):
        # abstract method
        pass

    @staticmethod
    @abstractmethod
    def from_proto(message: TriggerProto):
        # abstract method
        pass


@dataclass
class Metric(ABC):
    @abstractmethod
    def to_proto(self):
        # abstract method
        pass

    @staticmethod
    @abstractmethod
    def from_proto(message: MetricProto):
        # abstract method
        pass


@dataclass
class ScheduledTrigger(Trigger):
    cron: str = field(default="")
    interval: str = field(default="")

    def to_proto(self):
        return (
            TriggerProto(
                scheduled_trigger=ScheduledTriggerProto(interval=self.interval)
            )
            if self.interval
            else TriggerProto(scheduled_trigger=ScheduledTriggerProto(cron=self.cron))
        )

    @staticmethod
    def from_proto(message: TriggerProto):
        return ScheduledTrigger(
            cron=message.scheduled_trigger.cron,
            interval=message.scheduled_trigger.interval,
        )


@dataclass
class MetricBasedTrigger(Trigger):
    name: str = field(default="")
    metric: Metric = field(default="")
    direction: ThresholdDirection = field(default=ThresholdDirection.ABOVE)
    threshold: str = field(default="")
    override_cron: str = field(default="")

    def to_proto(self):
        return TriggerProto(
            metric_based_trigger=MetricBasedTriggerProto(
                name=self.name,
                threshold=self.threshold,
                metric=self.metric.to_proto(),
                threshold_direction=map_threshold_direction_to_proto(self.direction),
                override_cron=self.override_cron,
            )
        )

    @staticmethod
    def from_proto(message: TriggerProto):
        metric = map_metric_name_to_class(
            message.metric_based_trigger.metric.WhichOneof("metric")
        )
        return MetricBasedTrigger(
            name=message.metric_based_trigger.name,
            metric=metric.from_proto(message.metric_based_trigger.metric)
            if metric
            else None,
            threshold=message.metric_based_trigger.threshold,
            direction=map_proto_threshold_to_direction(
                message.metric_based_trigger.threshold_direction
            ),
            override_cron=message.metric_based_trigger.override_cron,
        )


@dataclass
class NoneTrigger(Trigger):
    def to_proto(self):
        return TriggerProto(none_trigger=NoneTriggerProto())

    @staticmethod
    def from_proto(message: TriggerProto):
        return NoneTrigger()


@dataclass
class OnBoardingTrigger(Trigger):
    def to_proto(self):
        return TriggerProto(on_boarding_trigger=OnBoardingTriggerProto())

    @staticmethod
    def from_proto(message: TriggerProto):
        return OnBoardingTrigger()


@dataclass
class SqlMetric(Metric):
    sql_query: str = field(default="")

    def to_proto(self):
        return MetricProto(sql_metric=SqlMetricProto(sql_query=self.sql_query))

    @staticmethod
    def from_proto(message: MetricProto):
        return SqlMetric(sql_query=message.sql_metric.sql_query)


@dataclass
class Notification(ABC):
    def to_proto(self):
        # abstract method
        pass

    @staticmethod
    @abstractmethod
    def from_proto(message: TriggerProto):
        # abstract method
        pass


@dataclass
class SlackNotification(Notification):
    webhook: str = field(default="")

    def to_proto(self):
        return NotificationProto(
            post_slack_notification=PostSlackNotificationProto(
                webhook=self.webhook,
            )
        )

    @staticmethod
    def from_proto(message: NotificationProto):
        return SlackNotification(
            webhook=message.post_slack_notification.webhook,
        )

    def __str__(self):
        return f"Slack Notification:\n webhook:{self.webhook}\n"


@dataclass
class CustomWebhook(Notification):
    url: str = field(default="")
    http_method: str = field(default="GET")
    headers: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, str] = field(default_factory=dict)

    _HTTP_NAME_TO_STATUS_MAPPING = {
        "GET": HttpMethodTypeProto.HTTP_METHOD_TYPE_GET,
        "POST": HttpMethodTypeProto.HTTP_METHOD_TYPE_POST,
        "PUT": HttpMethodTypeProto.HTTP_METHOD_TYPE_PUT,
        "PATCH": HttpMethodTypeProto.HTTP_METHOD_TYPE_PATCH,
        "DELETE": HttpMethodTypeProto.HTTP_METHOD_TYPE_DELETE,
        "HEAD": HttpMethodTypeProto.HTTP_METHOD_TYPE_HEAD,
    }

    _HTTP_STATUS_TO_NAME_MAPPING = {
        v: k for k, v in _HTTP_NAME_TO_STATUS_MAPPING.items()
    }

    def __post_init__(self):
        if self.http_method.upper() not in self._HTTP_NAME_TO_STATUS_MAPPING.keys():
            raise ValueError(
                f"HTTP method {self.http_method} is not a valid method. Available options are "
                f"{list(self._HTTP_NAME_TO_STATUS_MAPPING.keys())}"
            )

    def to_proto(self):
        return NotificationProto(
            custom_webhook=CustomWebhookProto(
                url=self.url,
                http_method=self._HTTP_NAME_TO_STATUS_MAPPING.get(
                    self.http_method.upper()
                ),
                headers=self.headers,
                data=self.data,
            )
        )

    @staticmethod
    def from_proto(message: NotificationProto):
        return CustomWebhook(
            url=message.custom_webhook.url,
            data=message.custom_webhook.data,
            headers=message.custom_webhook.headers,
            http_method=CustomWebhook._HTTP_STATUS_TO_NAME_MAPPING.get(
                message.custom_webhook.http_method
            ),
        )

    def __str__(self):
        return f"Custom webhook:\n url:{self.url}\n data:{self.data}\n headers:{self.headers}\n http method:{self.http_method}\n"


@dataclass
class AutomationAudit(ABC):
    date: datetime = field(default=datetime.now())
    user_id: str = field(default="")

    def to_proto(self):
        timestamp = Timestamp()
        timestamp.FromDatetime(self.date)
        return AutomationAuditProto(user_id=self.user_id, date=timestamp)

    @staticmethod
    def from_proto(message: AutomationAuditProto):
        return AutomationAudit(
            user_id=message.user_id,
            date=datetime.fromtimestamp(
                message.date.seconds + message.date.nanos / 1e9
            ),
        )


@dataclass
class Automation:
    id: str = field(default="")
    name: str = field(default="")
    model_id: str = field(default="")
    execute_immediately: bool = field(default=False)
    trigger: Trigger = field(default_factory=Trigger)
    action: Action = field(default_factory=Action)
    description: str = field(default="")
    environment: str = field(default="")
    is_enabled: bool = field(default=True)
    is_deleted: bool = field(default=False)
    is_sdk_v1: bool = field(default=False)
    create_audit: AutomationAudit = field(default_factory=AutomationAudit)
    on_error: Notification = field(default=None)
    on_success: Notification = field(default=None)

    def to_proto(self):
        return AutomationProto(
            automation_id=self.id,
            automation_spec=AutomationSpecProto(
                automation_name=self.name,
                is_enabled=self.is_enabled,
                is_sdk_v1=self.is_sdk_v1,
                execute_immediately=self.execute_immediately,
                model_id=self.model_id,
                action=self.action.to_proto(),
                trigger=self.trigger.to_proto() if self.trigger is not None else None,
                on_error=self.on_error.to_proto()
                if self.on_error is not None
                else None,
                on_success=self.on_success.to_proto()
                if self.on_success is not None
                else None,
            ),
            qwak_environment_id=self.environment,
            create_audit=self.create_audit.to_proto(),
            is_deleted=self.is_deleted,
        )

    @staticmethod
    def from_proto(message: AutomationProto):
        action = map_action_name_to_class(
            message.automation_spec.action.WhichOneof("action")
        )
        trigger = map_trigger_name_to_class(
            message.automation_spec.trigger.WhichOneof("trigger")
        )
        on_error_notification = map_notification_name_to_class(
            message.automation_spec.on_error.WhichOneof("notification")
        )
        on_success_notification = map_notification_name_to_class(
            message.automation_spec.on_success.WhichOneof("notification")
        )

        return Automation(
            id=message.automation_id,
            name=message.automation_spec.automation_name,
            description=message.automation_spec.automation_description,
            execute_immediately=message.automation_spec.execute_immediately,
            model_id=message.automation_spec.model_id,
            is_enabled=message.automation_spec.is_enabled,
            is_sdk_v1=message.automation_spec.is_sdk_v1,
            is_deleted=message.is_deleted,
            action=action.from_proto(message.automation_spec.action)
            if action
            else None,
            trigger=trigger.from_proto(message.automation_spec.trigger)
            if trigger
            else None,
            environment=message.qwak_environment_id,
            create_audit=AutomationAudit.from_proto(message.create_audit),
            on_error=on_error_notification.from_proto(message.automation_spec.on_error)
            if on_error_notification
            else None,
            on_success=on_success_notification.from_proto(
                message.automation_spec.on_success
            )
            if on_success_notification
            else None,
        )

    def __str__(self):
        on_error = f"\nOn Error: {self.on_error}" if self.on_error else ""
        on_success = f"\nOn Success: {self.on_success}" if self.on_success else ""
        return f"Id: {self.id}\tName: {self.name}\tModel: {self.model_id}\nDescription: {self.description}\nAction: {self.action}\nTrigger: {self.trigger}{on_error}{on_success}"


def map_notification_name_to_class(notification_name: str):
    mapping = {
        "post_slack_notification": SlackNotification,
        "custom_webhook": CustomWebhook,
    }
    return mapping.get(notification_name)


def map_action_name_to_class(action_name: str):
    mapping = {"build_deploy": QwakBuildDeploy, "batch_execution": BatchExecution}
    return mapping.get(action_name)


def map_trigger_name_to_class(trigger_name: str):
    mapping = {
        "scheduled_trigger": ScheduledTrigger,
        "metric_based_trigger": MetricBasedTrigger,
        "none_trigger": NoneTrigger,
        "on_boarding_trigger": OnBoardingTrigger,
    }

    return mapping.get(trigger_name)


def map_metric_name_to_class(metric_name: str):
    mapping = {"sql_metric": SqlMetric}
    return mapping.get(metric_name)
