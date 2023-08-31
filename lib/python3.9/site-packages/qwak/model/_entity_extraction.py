from typing import Dict, List

from qwak.clients.feature_store import FeatureRegistryClient
from qwak.exceptions import QwakException
from qwak.feature_store._common.feature_set_utils import (
    FeatureSetInfo,
    get_entity_type,
    get_feature_set_info,
    get_typed_feature,
)
from qwak.model.schema_entities import BaseFeature, Entity, FeatureStoreInput


def get_feature_set_name(feature: BaseFeature) -> str:
    feature_full_name: List[str] = feature.name.split(".")
    return feature_full_name[0].lower()


def normalize_features(
    features: List[BaseFeature], feature_set_to_entities: Dict[str, FeatureSetInfo]
) -> List[BaseFeature]:
    """
    Adding the relevant entity to the feature set features
    Args:
        features: list of features
        feature_set_to_entities: dict of feature set name and it's entity

    return normalize features - the features with the entities
    """
    new_normalize_features = []
    for feature in features:
        if isinstance(feature, FeatureStoreInput):
            feature_set_name = get_feature_set_name(feature)
            entity_spec = feature_set_to_entities.get(feature_set_name).entity_spec
            retrieved_entity = Entity(
                name=entity_spec.keys[
                    0
                ],  # currently support only one entity key per feature set
                type=get_entity_type(entity_spec.value_type),
            )
            if (
                feature.entity
                and retrieved_entity.name.lower() != feature.entity.name.lower()
            ):
                raise QwakException(
                    f"Explicitly supplied with an invalid entity: {feature.entity}, "
                    f"actual: {retrieved_entity}"
                )
            feature.entity = retrieved_entity
            feature.entity.name = feature.entity.name.lower()
            if isinstance(feature, FeatureStoreInput):
                feature_type = feature_set_to_entities.get(
                    feature_set_name
                ).feature_set_type
                feature_version = feature_set_to_entities.get(
                    feature_set_name
                ).feature_version
                feature = get_typed_feature(feature, feature_type, feature_version)
        new_normalize_features.append(feature)

    return new_normalize_features


def adding_entities_to_schema(entities: List[Entity], features: List[BaseFeature]):
    """
    Adding the new entities for feature store feature to the entities list from schema
    Args:
        entities: list of entities
        features: list of features
    Returns: the entities with the new entities from the feature store features
    """
    entity_name_to_entity = {entity.name: entity for entity in entities}
    for feature in features:
        if isinstance(feature, FeatureStoreInput):
            entity_name_to_entity[feature.entity.name] = feature.entity

    for entity in entity_name_to_entity.values():
        entity.name = entity.name.lower()

    return list(set(entity_name_to_entity.values()))


def normalize_schema(features: List[BaseFeature], entities: List[Entity]):
    """
    Normalize schema - will add entity to each features in schema if not exists
    Arg:
        model: the model with the required schema
    Return:
        updated model
    """

    normalized_features = features
    normalized_entities = entities
    if features and hasattr(features, "__iter__"):
        feature_set_enrichment: Dict[str, FeatureSetInfo] = dict()
        features_manager_client = FeatureRegistryClient()
        validation_errors = []
        for feature in features:
            try:
                if isinstance(feature, FeatureStoreInput):
                    feature_set_name = get_feature_set_name(feature)
                    feature_set_enrichment[
                        feature_set_name
                    ] = feature_set_enrichment.get(
                        feature_set_name,
                        get_feature_set_info(features_manager_client, feature_set_name),
                    )
                else:
                    feature_set_enrichment[feature.name] = feature
            except QwakException as e:
                validation_errors.append(e.message)

        if validation_errors:
            raise QwakException(f"Schema validation error: {validation_errors}")
        normalized_features = normalize_features(features, feature_set_enrichment)
        normalized_entities = adding_entities_to_schema(entities, features)

    return normalized_features, normalized_entities


def enrich_schema(features: List[BaseFeature], entities: List[Entity]):
    normalized_features, normalized_entities = normalize_schema(features, entities)
    return normalized_features, normalized_entities
