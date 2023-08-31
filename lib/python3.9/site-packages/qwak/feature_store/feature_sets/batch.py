import ast
import functools
import inspect
from dataclasses import dataclass
from datetime import datetime
from inspect import Signature
from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pass

from _qwak_proto.qwak.feature_store.features.execution_pb2 import (
    ExecutionSpec as ProtoExecutionSpec,
)
from _qwak_proto.qwak.feature_store.features.feature_set_pb2 import FeatureSetSpec
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    BatchFeatureSetV1 as ProtoBatchFeatureSetV1,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    FeatureSetBatchSource as ProtoFeatureSetBatchSource,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    FeatureSetType as ProtoFeatureSetType,
)
from _qwak_proto.qwak.features_operator.v3.features_operator_pb2 import (
    ValidationSuccessResponse,
)
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.clients.feature_store.operator_client import FeaturesOperatorClient
from qwak.exceptions import QwakException
from qwak.feature_store.entities.entity import Entity
from qwak.feature_store.feature_sets.backfill import Backfill
from qwak.feature_store.feature_sets.context import Context
from qwak.feature_store.feature_sets.execution_spec import ClusterTemplate
from qwak.feature_store.feature_sets.metadata import Metadata
from qwak.feature_store.feature_sets.read_policies import ReadPolicy, ReadPolicyABC
from qwak.feature_store.feature_sets.transformations import BatchTransformation
from typeguard import typechecked

# decorator attributes
_BACKFILL_POLICY_ATTRIBUTE = "_qwak_backfill_policy"
_EXECUTION_SPECIFICATION_ATTRIBUTE = "_qwak_execution_specification"
_SCHEDULING_POLICY_ATTRIBUTE = "_qwak_scheduling_policy"
_METADATA_ATTRIBUTE = "_qwak_metadata"

# Default timestamp column name for TimeFrame
_DEFAULT_TIMEFRAME_TS_COL_NAME = "qwak_window_end_ts"


def feature_set(
    *,
    entity: str,
    data_sources: Union[Dict[str, ReadPolicyABC], List[str]],
    timestamp_column_name: str = None,
    name: str = None,
):
    """
    Define a batch scheduled feature set. Default scheduling policy is every 4 hours.

    :param name: The name of the feature set. If not defined, taken from the function name
    :type entity: a string reference to a Qwak entity, or the entity definition itself
    :param data_sources: a string reference to a Qwak data source, or the data source definition itself
    :param timestamp_column_name: Timestamp column the feature set should consider when ingesting records.
    If more than one data source is specified - the field is mandatory. If only one data source is specified - the
    field will be taken by from the data source definition.

    Example:

    .. code-block:: python
        @batch.feature_set(entity="users", data_sources=["snowflake_users_table"])
        def user_features():
            return SparkSqlTransformation("SELECT user_id, age FROM data_source")

    """

    def decorator(function):
        sig: Signature = inspect.signature(function)
        if "context" in sig.parameters:
            user_transformation = function(context=Context())
        else:
            user_transformation = function()
        if not isinstance(user_transformation, BatchTransformation):
            raise ValueError(
                "Function must return a valid batch transformation function"
            )

        fs_name = name or function.__name__
        batch_feature_set = BatchFeatureSetV1(
            name=fs_name,
            entity=entity,
            data_sources=data_sources,
            timestamp_column_name=timestamp_column_name,
            transformation=user_transformation,
            scheduling_policy=getattr(
                function, _SCHEDULING_POLICY_ATTRIBUTE, "0 */4 * * *"
            ),
            metadata=getattr(
                function,
                _METADATA_ATTRIBUTE,
                Metadata(
                    description=fs_name,
                    display_name=fs_name,
                ),
            ),
            cluster_template=getattr(
                function,
                _EXECUTION_SPECIFICATION_ATTRIBUTE,
                ClusterTemplate.MEDIUM,
            ),
            backfill=getattr(function, _BACKFILL_POLICY_ATTRIBUTE, None),
            __instance_module_path__=inspect.stack()[1].filename,
        )

        functools.update_wrapper(batch_feature_set, user_transformation)
        return batch_feature_set

    return decorator


@typechecked
def scheduling(*, cron_expression: Optional[str]):
    """
    Sets the scheduling policy of the batch feature set according to the given cron expression
    A cron expression is a string consisting of six or seven subexpressions (fields) that describe individual details
    of the schedule. These fields, separated by white space, can contain any of the allowed values with various
    combinations of the allowed characters for that field.

    If None was passed instead of cron expression, the feature set will not be scheduled,
    and would be triggered only by user request

    Some more examples:

    Expression	Means
    0 0 12 * * ?	Fire at 12:00 PM (noon) every day
    0 15 10 ? * *	Fire at 10:15 AM every day
    0 15 10 * * ?	Fire at 10:15 AM every day
    0 15 10 * * ? *	Fire at 10:15 AM every day
    0 15 10 * * ? 2005	Fire at 10:15 AM every day during the year 2005
    0 * 14 * * ?	Fire every minute starting at 2:00 PM and ending at 2:59 PM, every day

    :param cron_expression: cron expression

    Example:

    .. code-block:: python

        @batch.feature_set(entity="users", data_sources=["snowflake_users_table"])
        @batch.scheduling(cron_expression="0 8 * * *")
        def user_features():
            return SparkSqlTransformation("SELECT user_id, age FROM data_source")

    """

    # TODO: add backend validation on cron expression
    def decorator(user_transformation):
        _validate_decorator_ordering(user_transformation)
        setattr(user_transformation, _SCHEDULING_POLICY_ATTRIBUTE, cron_expression)

        return user_transformation

    return decorator


@typechecked
def metadata(*, owner: str, description: str, display_name: str):
    """
    Sets additional user provided metadata

    :param owner: feature set owner
    :param description: General description of the feature set
    :param display_name: Human readable name of the feature set

    Example:

    .. code-block:: python

        @batch.feature_set(entity="users", data_sources=["snowflake_users_table"])
        @batch.metadata(
            owner="datainfra@qwak.com",
            display_name="User Batch Features",
            description="Users feature from the Snowflake replica of the production users table",
        )
        def user_features():
            return SparkSqlTransformation("SELECT user_id, age FROM data_source")

    """

    def decorator(user_transformation):
        _validate_decorator_ordering(user_transformation)
        setattr(
            user_transformation,
            _METADATA_ATTRIBUTE,
            Metadata(owner=owner, description=description, display_name=display_name),
        )

        return user_transformation

    return decorator


@typechecked
def backfill(
    *,
    start_date: datetime,
):
    """
    Set the backfill policy of the feature set.

    :param start_date: Start date of the backfill process. Data is extracted starting from this date
    :param end_date: Optionally the end date of the backfill process. Cuts off the backfill according to this date

    Example:

    .. code-block:: python
        @batch.feature_set(entity="users", data_sources=["snowflake_users_table"])
        @batch.backfill(start_date=datetime(2022, 1, 1))
        def user_features():
            return SparkSqlTransformation("SELECT user_id, age FROM data_source")

    """

    def decorator(user_transformation):
        _validate_decorator_ordering(user_transformation)
        setattr(
            user_transformation,
            _BACKFILL_POLICY_ATTRIBUTE,
            Backfill(start_date=start_date),
        )

        return user_transformation

    return decorator


@typechecked
def execution_specification(
    *,
    cluster_template: ClusterTemplate = None,
):
    """
    Set the execution specification of the cluster running the feature set

    :param cluster_template: Predefined template sizes

    Cluster template example:

    .. code-block:: python
        @batch.feature_set(entity="users", data_sources=["snowflake_users_table"])
        @batch.execution_specification(cluster_template=ClusterTemplate.MEDIUM)
        def user_features():
            return SparkSqlTransformation("SELECT user_id, age FROM data_source"

    """

    def decorator(user_transformation):
        _validate_decorator_ordering(user_transformation)
        setattr(
            user_transformation, _EXECUTION_SPECIFICATION_ATTRIBUTE, cluster_template
        )

        return user_transformation

    return decorator


def _validate_decorator_ordering(user_transformation):
    if isinstance(user_transformation, BatchFeatureSetV1):
        raise ValueError(
            "Wrong decorator ordering - @batch.feature_set should be to top most decorator"
        )


@dataclass
class BatchFeatureSetV1:
    name: str
    entity: str
    data_sources: List[str]
    timestamp_column_name: str = None
    scheduling_policy: str = None
    transformation: BatchTransformation = None
    metadata: Metadata = None
    cluster_template: Optional[ClusterTemplate] = None
    backfill: Optional[Backfill] = None
    __instance_module_path__: Optional[str] = None

    def _get_entity_definition(self, feature_registry):
        feature_set_entity = feature_registry.get_entity_by_name(self.entity)
        if not feature_set_entity:
            raise QwakException(
                f"Trying to register a feature set with a non existing entity -: {self.entity}"
            )
        return feature_set_entity.entity.entity_definition

    def _get_batch_source(self, batch_ds_name, feature_registry):
        batch_ds = feature_registry.get_data_source_by_name(batch_ds_name)
        if not batch_ds:
            raise QwakException(
                f"Trying to register a feature set with a non registered data source -: {batch_ds_name}"
            )

        ds_spec = batch_ds.data_source.data_source_definition.data_source_spec
        if ds_spec.WhichOneof("type") != "batch_source":
            raise ValueError(
                f"Can only register batch feature sets with batch sources. Source {batch_ds_name} is of type {ds_spec.WhichOneof('type')}"
            )

        return ds_spec.batch_source

    @classmethod
    def from_proto(cls, proto: FeatureSetSpec):
        batch_v1_def = proto.feature_set_type.batch_feature_set_v1

        return cls(
            name=proto.name,
            entity=Entity._from_proto(proto.entity),
            data_sources=[
                ds.name
                for ds in proto.feature_set_type.streaming_feature_set.data_sources
            ],
            timestamp_column_name=batch_v1_def.timestamp_column_name,
            scheduling_policy=batch_v1_def.scheduling_policy,
            transformation=BatchTransformation._from_proto(batch_v1_def.transformation),
            metadata=Metadata.from_proto(proto.metadata),
            cluster_template=ClusterTemplate.from_proto(batch_v1_def.execution_spec),
            backfill=Backfill.from_proto(batch_v1_def.backfill),
        )

    def _get_data_sources(self, feature_registry) -> List[ProtoFeatureSetBatchSource]:
        actual_data_sources = {}
        if isinstance(self.data_sources, list):
            actual_data_sources = {
                ds_name: ReadPolicy.NewOnly for ds_name in self.data_sources
            }
        else:
            actual_data_sources = self.data_sources

        return [
            ProtoFeatureSetBatchSource(
                data_source=self._get_batch_source(ds, feature_registry),
                read_policy=read_policy().to_proto()
                if inspect.isclass(read_policy)
                else read_policy.to_proto(),
            )
            for ds, read_policy in actual_data_sources.items()
        ]

    def _get_timestamp_column(
        self, data_sources: List[ProtoFeatureSetBatchSource]
    ) -> str:
        if self.timestamp_column_name:
            return self.timestamp_column_name

        # if no explicit timestamp column was set AND there's a TimeFrame
        # datasource, set the default column name.
        has_timeframe: bool = any(
            [ds.read_policy.WhichOneof("type") == "time_frame" for ds in data_sources]
        )
        if has_timeframe:
            return _DEFAULT_TIMEFRAME_TS_COL_NAME

        if len(data_sources) >= 2:
            raise ValueError(
                "If more than one data source is defined - `timestamp_column_name` on the feature set decorator must be defined"
            )

        # if we got here - only one data source is defined
        return data_sources[0].data_source.date_created_column

    def _to_proto(
        self,
        git_commit,
        features,
        feature_registry,
    ) -> FeatureSetSpec:
        data_sources = self._get_data_sources(feature_registry)

        return FeatureSetSpec(
            name=self.name,
            metadata=self.metadata.to_proto(),
            git_commit=git_commit,
            features=features,
            entity=self._get_entity_definition(feature_registry),
            feature_set_type=ProtoFeatureSetType(
                batch_feature_set_v1=ProtoBatchFeatureSetV1(
                    online_sink=True,
                    offline_sink=True,
                    timestamp_column_name=self._get_timestamp_column(data_sources),
                    scheduling_policy=self.scheduling_policy,
                    feature_set_batch_sources=data_sources,
                    execution_spec=ProtoExecutionSpec(
                        cluster_template=ClusterTemplate.to_proto(self.cluster_template)
                        if self.cluster_template
                        else None
                    ),
                    transformation=self.transformation._to_proto()
                    if self.transformation
                    else None,
                    backfill=self.backfill.to_proto() if self.backfill else None,
                )
            ),
        )

    def get_sample(self, number_of_rows: int = 10) -> "pd.DataFrame":
        """
        Fetches a sample of the Feature set transformation by loading requested sample of data from the data source
        and executing the transformation on that data.

        :param number_of_rows: number of rows requests
        :returns Sample Pandas Dataframe

        Example:

        .. code-block:: python
            @batch.feature_set(entity="users", data_sources=["snowflake_users_table"])
            @batch.backfill(start_date=datetime(2022, 1, 1))
            def user_features():
                return SparkSqlTransformation("SELECT user_id, age FROM data_source")

            sample_features = user_features.get_sample()
            print(sample_feature)
            #	    user_id	         timestamp	        user_features.age
            # 0	      1	        2021-01-02 17:00:00	              23
            # 1	      1	        2021-01-01 12:00:00	              51
            # 2	      2	        2021-01-02 12:00:00	              66
            # 3	      2	        2021-01-01 18:00:00	              34
        """

        """
        Fetches a sample of the Featureset transformation by loading requested sample of data from the data source
        and executing the transformation on that data.
        """

        try:
            import pandas as pd
        except ImportError:
            raise QwakException("Missing Pandas dependency required for getting sample")

        if 0 >= number_of_rows > 1000:
            raise ValueError(
                f"`number_rows` must be under 1000 and positive, got: {number_of_rows}"
            )

        operator_client = FeaturesOperatorClient()
        registry_client = FeatureRegistryClient()

        featureset_spec = self._to_proto(
            git_commit=None, features=None, feature_registry=registry_client
        )
        result = operator_client.validate_featureset_blocking(
            featureset_spec=featureset_spec,
            resource_path=None,
            num_samples=number_of_rows,
        )

        response = getattr(result, result.WhichOneof("type"))
        if isinstance(response, ValidationSuccessResponse):
            return pd.read_json(
                path_or_buf=ast.literal_eval(response.sample),
                dtype=response.spark_column_description,
            )
        else:
            raise QwakException(f"Sampling failed:\n{response}")
