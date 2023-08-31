from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.builds.builds_pb2 import (
    BatchFeature as ProtoBatchFeature,
    BatchFeatureV1 as ProtoBatchFeatureV1,
    Feature as ProtoFeature,
)
from _qwak_proto.qwak.feature_store.entities.entity_pb2 import EntitySpec
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    BatchFeatureSet,
    BatchFeatureSetV1,
    FeatureSetType,
)
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.exceptions import QwakException
from qwak.model.schema_entities import BaseFeature, Entity, FeatureStoreInput


@dataclass(unsafe_hash=True)
class BatchFeature(BaseFeature):
    entity: Optional[Entity] = None

    def to_proto(self):
        return ProtoFeature(
            batch_feature=ProtoBatchFeature(
                name=self.name, entity=self.entity.to_proto()
            )
        )


@dataclass(unsafe_hash=True)
class BatchFeatureV1(BaseFeature):
    entity: Optional[Entity] = None

    def to_proto(self):
        return ProtoFeature(
            batch_feature_v1=ProtoBatchFeatureV1(
                name=self.name, entity=self.entity.to_proto()
            )
        )


@dataclass()
class FeatureSetInfo:
    entity_spec: EntitySpec
    feature_set_type: FeatureSetType
    feature_version: int


def get_feature_set_info(
    feature_manager_client: FeatureRegistryClient, feature_set_name: str
) -> FeatureSetInfo:
    """
    Get the entities by the feature set name and feature type
    Args:
        feature_manager_client: feature manager client for the grpc request
        feature_set_name: the required feature set name
    Returns: tuple of entity spec, type and feature version

    """
    feature_set_response = feature_manager_client.get_feature_set_by_name(
        feature_set_name
    )
    if not feature_set_response:
        raise QwakException(f"Feature set: {feature_set_name} does not exist")

    featureset_type: FeatureSetType = get_feature_type(
        feature_set_response.feature_set.feature_set_definition.feature_set_spec.feature_set_type
    )
    featureset_version = _get_featureset_version(featureset_type)

    return FeatureSetInfo(
        entity_spec=feature_set_response.feature_set.feature_set_definition.feature_set_spec.entity.entity_spec,
        feature_set_type=featureset_type,
        feature_version=featureset_version,
    )


def get_feature_type(feature_set_type: FeatureSetType):
    return getattr(feature_set_type, feature_set_type.WhichOneof("set_type"))


def _get_featureset_version(feature_set_type: FeatureSetType) -> int:
    """
    Get Featureset version. Return 0 if none version is set
    """
    if hasattr(feature_set_type, "qwak_internal_protocol_version"):
        return feature_set_type.qwak_internal_protocol_version
    return 0


def get_typed_feature(
    feature: FeatureStoreInput, feature_type: FeatureSetType, featureset_version: int
) -> BaseFeature:
    """
    convert InputFeature to the relevant type
    Args:
        feature: Input feature to cast to the correct type
        feature_type: the feature type as it registered
        featureset_version: the version of the featureset

    Return:
        BaseFeature with the correct type
    """
    if isinstance(feature_type, BatchFeatureSet):
        return BatchFeature(name=feature.name, entity=feature.entity)
    elif isinstance(feature_type, BatchFeatureSetV1) and featureset_version == 0:
        return BatchFeature(name=feature.name, entity=feature.entity)
    elif isinstance(feature_type, BatchFeatureSetV1) and featureset_version != 0:
        return BatchFeatureV1(name=feature.name, entity=feature.entity)
    else:
        raise ValueError(
            f"Feature set type {feature_type} with protocol version {featureset_version} is not supported for extraction"
        )


def get_entity_type(value_type: EntitySpec.ValueType):
    """
    Normalize entity by the enum
    """

    if value_type == EntitySpec.ValueType.STRING:
        return str
    elif value_type == EntitySpec.ValueType.INT:
        return int
