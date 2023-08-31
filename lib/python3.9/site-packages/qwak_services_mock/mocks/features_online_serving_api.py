import json
from datetime import datetime

import grpc
import pandas as pd
from _qwak_proto.qwak.feature_store.serving.serving_pb2 import (
    MultiFeatureValuesResponse,
)
from _qwak_proto.qwak.feature_store.serving.serving_pb2_grpc import (
    ServingServiceServicer,
)
from qwak.exceptions import QwakException


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)


class ServingServiceMock(ServingServiceServicer):
    def __init__(self):
        self.features_by_entity_df = {}
        self._fail_request_ledger = 0

    def fail_next_request(self):
        self._fail_request_ledger += 1

    def upsert_features(self, entity_name, features_df):
        if entity_name not in features_df.columns:
            raise QwakException("Entity name does not exist in the data frame columns")

        entity_feature_df = self.features_by_entity_df.get(
            entity_name, pd.DataFrame(columns=[entity_name])
        )

        self.features_by_entity_df[entity_name] = (
            features_df.set_index(entity_name)
            .combine_first(entity_feature_df.set_index(entity_name))
            .reset_index()
            .rename({"index": entity_name})
        )

    def GetMultiFeatures(self, request, context) -> MultiFeatureValuesResponse:
        if self._fail_request_ledger > 0:
            self._fail_request_ledger -= 1
            raise ValueError("Test failing feature request!")

        if not self._validate_supported_features(request):
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Received an unimplemented one-of type for feature")
            return MultiFeatureValuesResponse()

        indexes, all_values, columns = [], [], []
        for row_index, entity_values in enumerate(request.entity_values_matrix.rows):
            values = []
            for index, entity_name in enumerate(
                request.entity_values_matrix.header.entity_names
            ):
                entity_feature_df = self.features_by_entity_df.get(entity_name)
                if entity_feature_df is not None:
                    all_features = self._filter_features(
                        entity_feature_df, entity_name, entity_values, index
                    )

                    if len(all_features) > 1:
                        context.set_code(grpc.StatusCode.INTERNAL)
                        context.set_details(
                            "Invalid state, a single entity has multiple feature rows - bug in the store :~("
                        )
                        return MultiFeatureValuesResponse()

                    elif len(all_features) == 1:
                        columns, values = self._populate_data(
                            all_features, columns, entity_name, request, values
                        )

            indexes.append(row_index)
            all_values.append(values)
        result = {"data": all_values, "columns": columns, "index": indexes}

        return MultiFeatureValuesResponse(
            pandas_df_as_json=json.dumps(result, cls=NpEncoder)
        )

    def _populate_data(self, all_features, columns, entity_name, request, values):
        requested_features = set()
        self._populate_requested_features(entity_name, request, requested_features)
        available_features = set(all_features.columns)
        existing_features = requested_features.intersection(available_features)
        non_existing_features = requested_features - available_features
        values = values + (
            [all_features.loc[0, feature_name] for feature_name in existing_features]
        )

        columns = self._populate_existing_values(columns, existing_features)
        columns, values = self._populate_non_existing_values(
            columns, non_existing_features, values
        )
        return columns, values

    def _populate_existing_values(self, columns, existing_features):
        if not any(x in columns for x in list(existing_features)):
            columns = columns + (list(existing_features))
        return columns

    @staticmethod
    def _populate_non_existing_values(columns, non_existing_features, values):
        if not any(x in columns for x in list(non_existing_features)):
            columns = columns + list(non_existing_features)
            for _ in non_existing_features:
                values = values + [None]
        return columns, values

    @staticmethod
    def _populate_requested_features(entity_name, request, requested_features):
        for entity_to_features in request.entities_to_features:
            if entity_to_features.entity_name == entity_name:
                for feature in entity_to_features.features:
                    one_of = feature.WhichOneof("type")
                    if one_of == "batch_feature":
                        requested_features.add(feature.batch_feature.name)
                    elif one_of == "batch_v1_feature":
                        requested_features.add(feature.batch_v1_feature.name)

    @staticmethod
    def _validate_supported_features(request):
        return all(
            [
                (
                    feature.WhichOneof("type") == "batch_v1_feature"
                    or feature.WhichOneof("type") == "batch_feature"
                )
                for entities in request.entities_to_features
                for feature in entities.features
            ]
        )

    @staticmethod
    def _filter_features(entity_feature_df, entity_name, entity_values, index):
        return (
            entity_feature_df[
                entity_feature_df[entity_name].astype(str)
                == str(entity_values.entity_values[index])
            ]
            .drop([entity_name], axis=1)
            .reset_index(drop=True)
        )
