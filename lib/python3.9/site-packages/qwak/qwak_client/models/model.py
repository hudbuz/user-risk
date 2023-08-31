from dataclasses import dataclass, field
from datetime import datetime

from _qwak_proto.qwak.models.models_pb2 import Model as ModelProto
from google.protobuf.timestamp_pb2 import Timestamp


@dataclass
class Model:
    model_id: str = field(default=None)
    uuid: str = field(default=None)
    model_name: str = field(default=None)
    model_description: str = field(default=None)
    project_id: str = field(default=None)
    created_by: str = field(default="")
    created_at: datetime = field(default=datetime.now())
    last_modified_by: str = field(default="")
    last_modified_at: datetime = field(default=datetime.now())

    def to_proto(self):
        timestamp = Timestamp()
        return ModelProto(
            model_id=self.model_id,
            uuid=self.uuid,
            display_name=self.model_name,
            model_description=self.model_description,
            project_id=self.project_id,
            created_by=self.created_by,
            created_at=timestamp.FromDatetime(self.created_at),
            last_modified_by=self.last_modified_by,
            last_modified_at=timestamp.FromDatetime(self.last_modified_at),
        )

    @staticmethod
    def from_proto(model_proto: ModelProto):
        return Model(
            model_id=model_proto.model_id,
            uuid=model_proto.uuid,
            model_name=model_proto.display_name,
            model_description=model_proto.model_description,
            project_id=model_proto.project_id,
            created_by=model_proto.created_by,
            created_at=datetime.fromtimestamp(
                model_proto.created_at.seconds + model_proto.created_at.nanos / 1e9
            ),
            last_modified_by=model_proto.last_modified_by,
            last_modified_at=datetime.fromtimestamp(
                model_proto.last_modified_at.seconds
                + model_proto.last_modified_at.nanos / 1e9
            ),
        )
