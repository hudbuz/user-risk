from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List

from _qwak_proto.qwak.build.v1.build_pb2 import Audit as AuditProto
from _qwak_proto.qwak.build.v1.build_pb2 import Build as BuildProto
from _qwak_proto.qwak.builds.builds_pb2 import Build as BuildsManagementBuild
from google.protobuf.timestamp_pb2 import Timestamp


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


@dataclass
class Build:
    model_id: str = field(default=None)
    build_id: str = field(default=None)
    commit_id: str = field(default=None)
    build_status: BuildStatus = field(default=BuildStatus.INVALID)
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_by: str = field(default="")
    created_at: datetime = field(default=datetime.now())
    last_modified_by: str = field(default="")
    last_modified_at: datetime = field(default=datetime.now())

    def to_proto(self):
        timestamp = Timestamp()
        return BuildProto(
            buildId=self.build_id,
            commitId=self.commit_id,
            build_status=self.build_status.value,
            tags=self.tags,
            params=self.parameters,
            metrics=self.metrics,
            audit=AuditProto(
                created_by=self.created_by,
                created_at=timestamp.FromDatetime(self.created_at),
                last_modified_by=self.last_modified_by,
                last_modified_at=timestamp.FromDatetime(self.last_modified_at),
            ),
        )

    @staticmethod
    def from_proto(build_proto: BuildProto):
        return Build(
            build_id=build_proto.buildId,
            commit_id=build_proto.commitId,
            build_status=BuildStatus(build_proto.build_status),
            tags=build_proto.tags,
            parameters=build_proto.params,
            metrics=build_proto.metrics,
            created_by=build_proto.audit.created_by,
            created_at=datetime.fromtimestamp(
                build_proto.audit.created_at.seconds
                + build_proto.audit.created_at.nanos / 1e9
            ),
            last_modified_by=build_proto.audit.last_modified_by,
            last_modified_at=datetime.fromtimestamp(
                build_proto.audit.last_modified_at.seconds
                + build_proto.audit.last_modified_at.nanos / 1e9
            ),
        )

    @staticmethod
    def from_builds_management(build_proto: BuildsManagementBuild):
        return Build(
            model_id=build_proto.build_spec.model_id,
            build_id=build_proto.build_spec.build_id,
            commit_id=build_proto.build_spec.commit_id,
            build_status=BuildStatus(build_proto.build_status),
            tags=build_proto.build_spec.tags,
            parameters=build_proto.build_spec.experiment_tracking_values.params,
            metrics=build_proto.build_spec.experiment_tracking_values.metrics,
            created_by=build_proto.created_by,
            created_at=datetime.fromtimestamp(
                build_proto.created_at.seconds + build_proto.created_at.nanos / 1e9
            ),
            last_modified_by=build_proto.last_modified_by,
            last_modified_at=datetime.fromtimestamp(
                build_proto.last_modified_at.seconds
                + build_proto.last_modified_at.nanos / 1e9
            ),
        )
