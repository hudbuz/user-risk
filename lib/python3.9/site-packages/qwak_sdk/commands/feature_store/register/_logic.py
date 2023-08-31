from typing import List, Optional, Tuple

from _qwak_proto.qwak.feature_store.features.feature_set_pb2 import Feature
from _qwak_proto.qwak.features_operator.v3.features_operator_pb2 import (
    SparkColumnDescription,
    ValidationSuccessResponse,
)
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.clients.feature_store.operator_client import FeaturesOperatorClient
from qwak.exceptions import QwakException
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource
from qwak.feature_store.entities.entity import Entity
from qwak.feature_store.feature_sets.batch import BatchFeatureSetV1
from tabulate import tabulate

from qwak_sdk.inner.file_registry import extract_class_objects
from qwak_sdk.inner.tools.cli_tools import ask_yesno
from qwak_sdk.tools.utils import qwak_spinner

DELIMITER = "----------------------------------------"


def _register_entities(
    qwak_python_files: List[str], registry: FeatureRegistryClient, force: bool
):
    """
    Register Feature Store Entity Objects

    Args:
        qwak_python_files: a list of python files containing qwak package imports
        registry: FeatureRegistryClient
        force: boolean determining if to force register all encountered Entity objects
    """
    with qwak_spinner(begin_text="Finding Entities to register", print_callback=print):
        qwak_entities: List[Tuple[Entity, str]] = extract_class_objects(
            qwak_python_files, Entity
        )

    print(f"👀 Found {len(qwak_entities)} Entities")
    for entity, source_file_path in qwak_entities:
        existing_entity = registry.get_entity_by_name(entity.name)
        if existing_entity:
            if ask_yesno(
                f"Update existing Entity '{entity.name}' from source file '{source_file_path}'?",
                force,
            ):
                registry.update_entity(
                    existing_entity.entity.entity_definition.entity_id,
                    entity._to_proto(),
                )
        else:
            if ask_yesno(
                f"Create new Entity '{entity.name}' from source file '{source_file_path}'?",
                force,
            ):
                registry.create_entity(entity._to_proto())
    print(DELIMITER)


def _register_data_sources(
    qwak_python_files: List[str],
    registry: FeatureRegistryClient,
    operator_client: FeaturesOperatorClient,
    force: bool,
    no_validation: bool,
    ignore_validation_errors: bool,
):
    """
    Register Feature Store Data Source Objects

    Args:
        qwak_python_files: a list of python files containing qwak package imports
        registry: FeatureRegistryClient
        operator_client: features operator grpc client
        force: boolean determining if to force register all encountered Data Source objects
        no_validation: whether to validate entities
        ignore_validation_errors: whether to ignore and continue registering objects after encountering validation errors
    """
    with qwak_spinner(
        begin_text="Finding Data Sources to register", print_callback=print
    ):
        qwak_sources = extract_class_objects(qwak_python_files, BaseBatchSource)

    print(f"👀 Found {len(qwak_sources)} Data Sources")
    for data_source, source_file_path in qwak_sources:
        validation_failed = False

        if no_validation:
            print("Skipping data source validation")
        else:
            try:
                _handle_data_source_validation(operator_client, data_source)
            except Exception as e:
                print(str(e))
                validation_failed = True

        if validation_failed and not ignore_validation_errors:
            print("Not continuing to registration due to failure in validation")
            exit(1)

        existing_source = registry.get_data_source_by_name(data_source.name)
        if existing_source:
            if ask_yesno(
                f"Update existing Data Source '{data_source.name}' from source file '{source_file_path}'?",
                force,
            ):
                registry.update_data_source(
                    existing_source.data_source.data_source_definition.data_source_id,
                    data_source._to_proto(),
                )
        else:
            if ask_yesno(
                f"Create Data Source '{data_source.name}' from source file '{source_file_path}'?",
                force,
            ):
                registry.create_data_source(data_source._to_proto())
    print(DELIMITER)


def _handle_data_source_validation(
    operator_client: FeaturesOperatorClient,
    data_source: BaseBatchSource,
):
    print(f"Validating '{data_source.name}' data source")
    with qwak_spinner(begin_text="", print_callback=print):
        result = operator_client.validate_data_source_blocking(
            data_source_spec=data_source._to_proto(), num_samples=10
        )

    response = getattr(result, result.WhichOneof("type"))
    if isinstance(response, ValidationSuccessResponse):
        print("✅ Validation completed successfully, got data source columns:")

        table = [
            (x.column_name, x.spark_type) for x in response.spark_column_description
        ]
        print(tabulate(table, headers=["column name", "type"]))
    else:
        raise QwakException(f"🧨 Validation failed:\n{response}")


def _register_features_sets(
    qwak_python_files: List[str],
    registry: FeatureRegistryClient,
    operator_client: FeaturesOperatorClient,
    force: bool,
    git_commit: str,
    no_validation: bool,
    ignore_validation_errors: bool,
):
    """
    Register Feature Store Feature Set Objects

    Args:
        qwak_python_files: a list of python files containing qwak package imports
        registry: FeatureRegistryClient
        operator_client: features operator grpc client
        force: boolean determining if to force register all encountered Feature Set objects
        git_commit: the git commit of the parent folder
        no_validation: whether to validate entities
        ignore_validation_errors: whether to ignore and continue registering objects after encountering validation errors
    """
    with qwak_spinner(
        begin_text="Finding Feature Sets to register", print_callback=print
    ):
        qwak_feature_sets = extract_class_objects(qwak_python_files, BatchFeatureSetV1)

    print(f"👀 Found {len(qwak_feature_sets)} Feature Set(s)")

    for feature_set, source_file_path in qwak_feature_sets:
        existing_feature_set = registry.get_feature_set_by_name(feature_set.name)

        registration: bool = False
        if existing_feature_set:
            registration = ask_yesno(
                f"Update existing feature set '{feature_set.name}' from source file '{source_file_path}'?",  # nosec B608
                force,
            )
        else:
            registration = ask_yesno(
                f"Create new feature set '{feature_set.name}' from source file '{source_file_path}'?",
                force,
            )

        if registration:
            spark_columns_features = _batch_v1_feature_set_validation(
                feature_set=feature_set,
                no_validation=no_validation,
                operator=operator_client,
                registry=registry,
            )

            proto_feature_set = feature_set._to_proto(
                git_commit, spark_columns_features, registry
            )

            if existing_feature_set:
                registry.update_feature_set(
                    existing_feature_set.feature_set.feature_set_definition.feature_set_id,
                    proto_feature_set,
                )
            else:
                registry.create_feature_set(proto_feature_set)


def _handle_batch_v1_feature_set_validation(
    operator_client: FeaturesOperatorClient,
    registry_client: FeatureRegistryClient,
    feature_set: BatchFeatureSetV1,
) -> List[SparkColumnDescription]:
    print(f"Validating '{feature_set.name}' feature set")
    with qwak_spinner(begin_text="", print_callback=print):
        feature_set_spec = feature_set._to_proto(
            feature_registry=registry_client, features=None, git_commit=None
        )
        result = operator_client.validate_featureset_blocking(
            feature_set_spec, resource_path=None, num_samples=10
        )

    response = getattr(result, result.WhichOneof("type"))
    if isinstance(response, ValidationSuccessResponse):
        print("✅ Validation completed successfully, got data source columns:")
        table = [
            (x.column_name, x.spark_type) for x in response.spark_column_description
        ]
        print(tabulate(table, headers=["column name", "type"]))
        return response.spark_column_description

    else:
        raise QwakException(f"🧨 Validation failed:\n{response}")


def _get_features_from_spark_columns_description(
    spark_columns_description: List[SparkColumnDescription],
) -> Optional[List[Feature]]:
    if spark_columns_description:
        spark_features = []
        for spark_column in spark_columns_description:
            feature_col_name_list = str(spark_column.column_name).split(".")
            feature_name = (
                feature_col_name_list[1]
                if feature_col_name_list.__len__() == 2
                else spark_column.column_name
            )
            spark_features.append(
                Feature(
                    feature_name=feature_name,
                    feature_type=spark_column.spark_type,
                )
            )
        return spark_features
    else:
        return None


def _batch_v1_feature_set_validation(
    feature_set: BatchFeatureSetV1,
    no_validation: bool,
    operator: FeaturesOperatorClient,
    registry: FeatureRegistryClient,
) -> Optional[List[Feature]]:
    """
    Validates featureset transformation
    Args:
        feature_set: BatchFeatureSetV1 featureset
        no_validation: skip validation
        operator: Operator client
        registry: Registry client
    Returns:
        Optional list of features returned from validation
    """
    spark_columns_features: Optional[List[Feature]] = []

    if not no_validation:
        try:
            spark_columns_description = _handle_batch_v1_feature_set_validation(
                operator_client=operator,
                registry_client=registry,
                feature_set=feature_set,
            )
            spark_columns_features = _get_features_from_spark_columns_description(
                spark_columns_description
            )
        except Exception as e:
            print(str(e))
            print("Not continuing to registration due to failure in validation")
            exit(1)
    else:
        print(f"Skipping validation for '{feature_set.name}' feature set")
    return spark_columns_features
