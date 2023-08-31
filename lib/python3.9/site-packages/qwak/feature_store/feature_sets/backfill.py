from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    Backfill as ProtoBackfill,
)
from google.protobuf.timestamp_pb2 import Timestamp


class FillupMethodABC(ABC):
    @abstractmethod
    def to_proto(self):
        pass


class AsScheduled(FillupMethodABC):
    @classmethod
    def to_proto(cls):
        return ProtoBackfill.AS_SCHEDULED


class Snapshot(FillupMethodABC):
    @classmethod
    def to_proto(cls):
        return ProtoBackfill.SNAPSHOT


class BackfillType:
    Snapshot = Snapshot
    AsScheduled = AsScheduled


@dataclass
class Backfill:
    start_date: datetime
    fill_up_method: FillupMethodABC = field(default_factory=BackfillType.AsScheduled)

    def to_proto(self):
        proto_timestamp = Timestamp()
        proto_timestamp.FromDatetime(self.start_date)
        return ProtoBackfill(
            start_date=proto_timestamp, fillup_method=self.fill_up_method.to_proto()
        )

    @classmethod
    def from_proto(cls, proto: "ProtoBackfill"):
        backfill_mapping = {
            ProtoBackfill.FillUpMethod.AS_SCHEDULED: BackfillType.AsScheduled(),
            ProtoBackfill.FillUpMethod.SNAPSHOT: BackfillType.Snapshot(),
        }

        return cls(
            start_date=datetime.utcfromtimestamp(
                proto.start_date.seconds + proto.start_date.nanos / 1e9,
            ),
            fill_up_method=backfill_mapping.get(proto.fillup_method),
        )
