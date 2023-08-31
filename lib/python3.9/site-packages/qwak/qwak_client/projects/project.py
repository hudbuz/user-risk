from dataclasses import dataclass, field
from datetime import datetime

from _qwak_proto.qwak.projects.projects_pb2 import Project as ProjectProto
from _qwak_proto.qwak.projects.projects_pb2 import ProjectSpec as ProjectSpecProto
from google.protobuf.timestamp_pb2 import Timestamp


@dataclass
class Project:
    project_id: str = field(default=None)
    project_name: str = field(default=None)
    project_description: str = field(default=None)
    models_count: int = field(default=0)
    models_active: int = field(default=0)
    created_by: str = field(default="")
    created_at: datetime = field(default=datetime.now())
    last_modified_by: str = field(default="")
    last_modified_at: datetime = field(default=datetime.now())

    def to_proto(self):
        timestamp = Timestamp()
        return ProjectProto(
            spec=ProjectSpecProto(
                project_id=self.project_id,
                project_name=self.project_name,
                project_description=self.project_description,
                models_count=self.models_count,
                models_active=self.models_active,
                created_by=self.created_by,
                created_at=timestamp.FromDatetime(self.created_at),
                last_modified_by=self.last_modified_by,
                last_modified_at=timestamp.FromDatetime(self.last_modified_at),
            )
        )

    @staticmethod
    def from_proto(project_spec_proto: ProjectSpecProto):
        return Project(
            project_id=project_spec_proto.project_id,
            project_name=project_spec_proto.project_name,
            project_description=project_spec_proto.project_description,
            models_count=project_spec_proto.models_count,
            models_active=project_spec_proto.models_active,
            created_by=project_spec_proto.created_by,
            created_at=datetime.fromtimestamp(
                project_spec_proto.created_at.seconds
                + project_spec_proto.created_at.nanos / 1e9
            ),
            last_modified_by=project_spec_proto.last_modified_by,
            last_modified_at=datetime.fromtimestamp(
                project_spec_proto.last_modified_at.seconds
                + project_spec_proto.last_modified_at.nanos / 1e9
            ),
        )
