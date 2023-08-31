import os
from datetime import datetime
from typing import Optional

from _qwak_proto.qwak.inference.feedback.feedback_pb2 import (
    Actuals,
    ActualValuesRequest,
    Entity,
)
from _qwak_proto.qwak.inference.feedback.feedback_pb2_grpc import FeedbackServiceStub
from google.protobuf.internal.well_known_types import Timestamp
from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.inner.tool.grpc.grpc_tools import create_grpc_channel


class FeedbackClient:
    def __init__(
        self, model_id: Optional[str] = None, environment_name: Optional[str] = None
    ) -> None:
        """Construct feedback client.

        Args:
            model_id: The model id that you want to submit a feedback for.
            environment_name: Specify the environment the model is hosted on.
        """

        self._model_id = model_id or os.getenv("QWAK_MODEL_ID")
        model_url = EcosystemClient().get_environment_model_api(
            environment_name=environment_name
        )
        self.channel = create_grpc_channel(
            url=model_url,
            enable_ssl=True,
            enable_auth=True,
        )

    def actual(
        self,
        entity_name: str,
        entity_value: str,
        tag: str,
        actuals: list,
        timestamp: datetime = datetime.utcnow(),
    ):
        """

        Args:
            entity_name:
            entity_value:
            tag:
            actuals:
            timestamp:

        Returns:

        """
        proto_timestamp = Timestamp()
        proto_timestamp.FromDatetime(timestamp)
        feedback = FeedbackServiceStub(self.channel)
        feedback_response = feedback.PostFeedback(
            ActualValuesRequest(
                model_id=self._model_id,
                entity=Entity(name=entity_name, value=entity_value),
                actuals=Actuals(tag=tag, value=actuals),
                timestamp=proto_timestamp,
            )
        )
        return feedback_response
