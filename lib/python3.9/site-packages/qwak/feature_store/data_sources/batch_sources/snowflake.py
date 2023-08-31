from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    SnowflakeSource as ProtoSnowflakeSource,
)
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.exceptions import QwakException
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource


@dataclass
class SnowflakeSource(BaseBatchSource):
    host: str
    username_secret_name: str
    password_secret_name: str
    database: str
    schema: str
    warehouse: str
    table: Optional[str] = None
    query: Optional[str] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.table and self.query:
            raise QwakException("Only one of query and table may be set")
        if not self.table and not self.query:
            raise QwakException("One of table or query must be set")

        if not self.username_secret_name:
            raise QwakException("username_secret_name must be set!")

        if not self.password_secret_name:
            raise QwakException("password_secret_name must be set!")

    def _to_proto(self):
        return ProtoDataSourceSpec(
            batch_source=ProtoBatchSource(
                name=self.name,
                description=self.description,
                date_created_column=self.date_created_column,
                snowflakeSource=ProtoSnowflakeSource(
                    host=self.host,
                    username_secret_name=self.username_secret_name,
                    password_secret_name=self.password_secret_name,
                    database=self.database,
                    schema=self.schema,
                    warehouse=self.warehouse,
                    table=self.table,
                    query=self.query,
                ),
            )
        )

    @classmethod
    def _from_proto(cls, proto):
        snowflake = proto.snowflakeSource
        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            host=snowflake.host,
            username_secret_name=snowflake.username_secret_name,
            password_secret_name=snowflake.password_secret_name,
            database=snowflake.database,
            schema=snowflake.schema,
            warehouse=snowflake.warehouse,
            table=snowflake.table,
            query=snowflake.query,
        )
