from enum import Enum, auto

from _qwak_proto.qwak.feature_store.features.execution_pb2 import LARGE as ProtoLARGE
from _qwak_proto.qwak.feature_store.features.execution_pb2 import MEDIUM as ProtoMEDIUM
from _qwak_proto.qwak.feature_store.features.execution_pb2 import SMALL as ProtoSMALL
from _qwak_proto.qwak.feature_store.features.execution_pb2 import XLARGE as ProtoXLARGE
from _qwak_proto.qwak.feature_store.features.execution_pb2 import (
    ClusterTemplate as ProtoClusterTemplate,
)


class ClusterTemplate(Enum):
    """
    TODO: add docs
    Cluster templates

    """

    def _generate_next_value_(name, start, count, last_values):
        return name

    SMALL = auto(), "TODO SMALL Cluster docs"
    MEDIUM = auto(), "TODO MEDIUM Cluster docs"
    LARGE = auto(), "TODO LARGE Cluster docs"
    XLARGE = auto(), "TODO XLARGE Cluster docs"

    _cluster_template_to_proto = {
        SMALL: ProtoSMALL,
        MEDIUM: ProtoMEDIUM,
        LARGE: ProtoLARGE,
        XLARGE: ProtoXLARGE,
    }

    _proto_to_cluster_template = {v: k for k, v in _cluster_template_to_proto.items()}

    @classmethod
    def from_proto(cls, proto: ProtoClusterTemplate):
        return ClusterTemplate._proto_to_cluster_template.value[proto]

    @staticmethod
    def to_proto(template):
        if not template:
            return None

        _cluster_template_to_proto = {
            ClusterTemplate.SMALL: ProtoSMALL,
            ClusterTemplate.MEDIUM: ProtoMEDIUM,
            ClusterTemplate.LARGE: ProtoLARGE,
            ClusterTemplate.XLARGE: ProtoXLARGE,
        }
        return _cluster_template_to_proto[template]
