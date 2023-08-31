from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BigquerySource as ProtoBigquerySource,
)
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.exceptions import QwakException
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource


@dataclass
class BigQuerySource(BaseBatchSource):
    credentials_secret_name: str
    project: Optional[str] = None
    dataset: Optional[str] = None
    table: Optional[str] = None
    parent_project: Optional[str] = None
    sql: Optional[str] = None
    views_enabled: Optional[bool] = False
    materialization_dataset: Optional[str] = None
    materialization_project: Optional[str] = None

    def __post_init__(self):
        if not (self.sql or (self.table and self.dataset)):
            raise QwakException("Either SQL or (table and dataset) must be provided.")

        if self.sql:
            if not self.materialization_project:
                raise QwakException(
                    "For SQL query mode, materialization project setting must be set."
                )
            if not self.materialization_dataset:
                raise QwakException(
                    "For SQL query mode, materialization dataset setting must be set."
                )
            self.views_enabled = True

    @classmethod
    def _from_proto(cls, proto):
        bigquery = proto.bigquerySource
        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            credentials_secret_name=bigquery.credentials_secret_name,
            dataset=bigquery.dataset,
            table=bigquery.table,
            project=bigquery.project,
            parent_project=bigquery.parent_project,
            sql=bigquery.sql,
            views_enabled=bigquery.views_enabled,
            materialization_project=bigquery.materialization_project,
            materialization_dataset=bigquery.materialization_dataset,
        )

    def _to_proto(self):
        return ProtoDataSourceSpec(
            batch_source=ProtoBatchSource(
                name=self.name,
                description=self.description,
                date_created_column=self.date_created_column,
                bigquerySource=ProtoBigquerySource(
                    credentials_secret_name=self.credentials_secret_name,
                    dataset=self.dataset,
                    table=self.table,
                    project=self.project,
                    parent_project=self.parent_project,
                    sql=self.sql,
                    views_enabled=self.views_enabled,
                    materialization_dataset=self.materialization_dataset,
                    materialization_project=self.materialization_project,
                ),
            )
        )
