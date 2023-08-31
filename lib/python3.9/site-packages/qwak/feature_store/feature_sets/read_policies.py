import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    Aggregation as AggregationProto,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    AggregationPopulation as AggregationPopulationProto,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    DataSourceReadPolicy as ProtoDataSourceReadPolicy,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    FullRead as ProtoFullRead,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    NewOnly as ProtoNewOnly,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    PopulationTimeframe as PopulationTimeframeProto,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    PopulationTimeframeNewOnly as PopulationTimeframeNewOnlyProto,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    TimeFrame as ProtoTimeFrame,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    Vanilla as VanillaProto,
)
from qwak.exceptions import QwakException


class ReadPolicyABC(ABC):
    @abstractmethod
    def to_proto(self):
        pass

    @classmethod
    def from_proto(cls, proto: "ProtoDataSourceReadPolicy"):
        read_policy_mapping = {
            "time_frame": TimeFrame,
            "new_only": NewOnly,
            "full_read": FullRead,
        }

        read_policy_type: str = proto.WhichOneof("type")
        if read_policy_type in read_policy_mapping:
            read_policy_class = read_policy_mapping.get(read_policy_type)
            return read_policy_class.from_proto(proto)

        raise QwakException(f"Got unsupported read policy type: {read_policy_type}")


class NewOnly(ReadPolicyABC):
    def to_proto(self):
        return ProtoDataSourceReadPolicy(new_only=ProtoNewOnly())

    @classmethod
    def from_proto(cls, proto):
        return cls()


class TimeFrameFlavor(ABC):
    @abstractmethod
    def to_proto(self):
        pass


class Vanilla(TimeFrameFlavor):
    def to_proto(self):
        return VanillaProto()

    @classmethod
    def from_proto(cls, proto):
        return cls()


class Aggregation(TimeFrameFlavor):
    @abstractmethod
    def to_proto(self):
        pass

    @classmethod
    def from_proto(cls, proto):
        return cls()


class AggregationPopulation(Aggregation):
    def to_proto(self):
        return AggregationProto(aggregation_population=AggregationPopulationProto())

    @classmethod
    def from_proto(cls, proto):
        return cls()


@dataclass
class TimeFrame(ReadPolicyABC):
    days: int = 0
    hours: int = 0
    minutes: int = 0
    flavor: Optional[TimeFrameFlavor] = field(default_factory=lambda: Vanilla())

    def to_proto(self):
        if isinstance(self.flavor, Aggregation):
            return ProtoDataSourceReadPolicy(
                time_frame=ProtoTimeFrame(
                    minutes=self._get_time_frame_total_minutes(),
                    aggregation=self.flavor.to_proto(),
                )
            )
        return ProtoDataSourceReadPolicy(
            time_frame=ProtoTimeFrame(
                minutes=self._get_time_frame_total_minutes(),
                vanilla=self.flavor.to_proto(),
            )
        )

    @classmethod
    def from_proto(cls, proto):
        flavor_mapping = {
            "vanilla": Vanilla,
            "aggregation": AggregationPopulation,
        }

        minutes_raw = proto.time_frame.minutes

        days = minutes_raw // 1440  # (24 * 60)
        leftover_minutes = minutes_raw % 1440
        hours = leftover_minutes // 60
        mins = minutes_raw - (days * 1440) - (hours * 60)

        flavor_type = proto.time_frame.WhichOneof("flavor")
        return cls(
            days=days,
            hours=hours,
            minutes=mins,
            flavor=flavor_mapping.get(flavor_type).from_proto(proto),
        )

    def __post_init__(self):
        self._validate_positive_integer(self.days, "days")
        self._validate_positive_integer(self.hours, "hours")
        self._validate_positive_integer(self.minutes, "minutes")
        if self._get_time_frame_total_minutes() <= 0:
            raise QwakException(
                "Time frame must have a positive amount of time, "
                "it is mandatory to set one of the fields"
            )
        if inspect.isclass(self.flavor):
            self.flavor = self.flavor()

    def _get_time_frame_total_minutes(self):
        return self.days * 24 * 60 + self.hours * 60 + self.minutes

    @staticmethod
    def _validate_positive_integer(int_value, field_name):
        if not isinstance(int_value, int) or int_value < 0:
            raise QwakException(f"{field_name} must be a positive integer")


class FullReadFlavor(ABC):
    @abstractmethod
    def to_proto(self):
        pass

    @classmethod
    def from_proto(cls, proto):
        return cls()


class FullReadVanilla(FullReadFlavor):
    def to_proto(self):
        return VanillaProto()

    @classmethod
    def from_proto(cls, proto):
        return cls()


class PopulationTimeframe(FullReadFlavor):
    @abstractmethod
    def to_proto(self):
        pass

    @classmethod
    def from_proto(cls, proto):
        return cls()


class PopulationTimeframeNewOnly(PopulationTimeframe):
    def to_proto(self):
        return PopulationTimeframeProto(new_only=PopulationTimeframeNewOnlyProto())

    @classmethod
    def from_proto(cls, proto):
        return cls()


@dataclass
class FullRead(ReadPolicyABC):
    flavor: Optional[FullReadFlavor] = field(default_factory=lambda: FullReadVanilla())

    def __post_init__(self):
        if inspect.isclass(self.flavor):
            self.flavor = self.flavor()

    @classmethod
    def from_proto(cls, proto):
        flavor_mapping = {
            "default": FullReadVanilla,
            "population_timeframe": PopulationTimeframeNewOnly,
        }

        flavor_type = proto.full_read.WhichOneof("flavor")
        return cls(
            flavor=flavor_mapping.get(flavor_type).from_proto(proto),
        )

    def to_proto(self):
        if isinstance(self.flavor, PopulationTimeframe):
            return ProtoDataSourceReadPolicy(
                full_read=ProtoFullRead(population_timeframe=self.flavor.to_proto())
            )

        return ProtoDataSourceReadPolicy(
            full_read=ProtoFullRead(default=self.flavor.to_proto())
        )


class Population:
    NewOnly = PopulationTimeframeNewOnly


class Aggregations:
    Population = AggregationPopulation


class ReadPolicy:
    NewOnly = NewOnly
    TimeFrame = TimeFrame
    FullRead = FullRead
