import dataclasses
import itertools
import re
from typing import Callable, Dict, List

from qwak.feature_store.offline import OfflineClient
from qwak.model.schema_entities import BaseFeature, Entity, FeatureStoreInput


def is_asterisk(feature: BaseFeature) -> bool:
    """
    Checks whether the given Feature is a FeatureStoreInput with an '*'
    Args:
        feature: input feature
    Return
    """
    p = "^[^.]+.\\*$"
    # if this is not a FeatureStoreInput, no need to check
    if isinstance(feature, FeatureStoreInput):
        return bool(re.match(p, feature.name))
    return False


def inflate_feature(feature: BaseFeature, offline_fs: OfflineClient) -> List:
    """
    Inflates a Feature into a list. if the feature contains an '*', a list
        a feature for each feature in the associate FeatureSet is returned. else -
        a list containing the original feature is returned.
    Args:
        feature: input feature
        offline_fs: OfflineFeatureStore to query
    Return:
        if an '*' exists, returns a List of FeatureStoreInuput for each feature in the feature set
        else returns a list containing the feature.

    """
    if is_asterisk(feature):
        featureset_name = feature.name.lower().split(".")[0]

        p = f"^{featureset_name}+.*$"
        feature_names = [
            feature.lower()
            for feature in offline_fs.get_columns_by_feature_set(featureset_name)
            if re.match(p, feature.lower())
        ]
        return [dataclasses.replace(feature, name=name) for name in feature_names]

    return [feature]


def unpack_asterisk_features(
    features: List[BaseFeature],
    feature_store_generator: Callable[[], OfflineClient] = lambda: OfflineClient(),
) -> List[BaseFeature]:
    """
    Handles features with an '*'.
    If a feature of type FeatureStoreInput that matches '<featureset_name>.*' exists,
    transforms this feature into the list of features actually present in the FeatureSet.
    Other features remain unchanged.
    the original order is maintained, and the internal order (inside a FeatureStoreInput
     that has an '*') is inferred from the order of columns returned from
     OfflineFeatureStore.get_columns_by_feature_set(...)
     Args:
         features: List of features
         feature_store_generator: A function that generates an instance of OfflineFeatureStore
                                  If no feature with an '*' is found, the function is not called.

    Returns:
        the unpacked features
    """
    asterisk_found = any([is_asterisk(f) for f in features])
    if asterisk_found:
        # at least 1 feature contains an asterisk, map each feature to
        # a list of features (features with no '*' are mapped to a list of size 1), then flatten it.
        offline_feature_store = feature_store_generator()
        inflated_features = [
            inflate_feature(feature, offline_feature_store) for feature in features
        ]
        return list(itertools.chain.from_iterable(inflated_features))

    return features


def unpack_asterisk_features_from_key_mapping(
    key_to_features: Dict[str, List[str]],
    feature_store_generator: Callable[[], OfflineClient] = lambda: OfflineClient(),
) -> dict:
    """
    Handles features with an '*'.
    If a feature name that matches '<featureset_name>.*' exists in one of the lists,
    transforms this feature into the list of feature names actually present in the FeatureSet.
    Other features remain unchanged.
    the original order is maintained, and the internal order (inside a feature
     that has an '*') is inferred from the order of columns returned from
     OfflineFeatureStore.get_columns_by_feature_set(...)
     Args:
         key_to_features: dictionary of entity keys to requested features (same as in the offline store)
             >>> key_to_features = {'uuid': ['user_purchases.*',
             >>>                             'user_purchases.avg_purchase_amount']}
         feature_store_generator: A function that generates an instance of OfflineFeatureStore
                                  If no feature with an '*' is found, the function is not called.
    Returns:
        the unpacked features
    """
    inflated_key_to_features = {}

    for entity, features in key_to_features.items():
        feature_inputs = [
            FeatureStoreInput(name=feature, entity=Entity(name=entity, type=str))
            for feature in features
        ]
        unpacked = unpack_asterisk_features(feature_inputs, feature_store_generator)
        feature_list = [feature.name for feature in unpacked]
        inflated_key_to_features[entity] = feature_list
    return inflated_key_to_features
