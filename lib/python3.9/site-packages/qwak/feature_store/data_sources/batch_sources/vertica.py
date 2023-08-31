from dataclasses import dataclass

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    VerticaSource as ProtoVerticaSource,
)
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource


@dataclass
class VerticaSource(BaseBatchSource):
    host: str
    port: int
    database: str
    schema: str
    table: str
    username_secret_name: str
    password_secret_name: str

    @classmethod
    def _from_proto(cls, proto):
        vertica = proto.verticaSource
        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            host=vertica.host,
            username_secret_name=vertica.username_secret_name,
            password_secret_name=vertica.password_secret_name,
            database=vertica.database,
            schema=vertica.schema,
            port=vertica.port,
            table=vertica.table,
        )

    def _to_proto(self):
        return ProtoDataSourceSpec(
            batch_source=ProtoBatchSource(
                name=self.name,
                description=self.description,
                date_created_column=self.date_created_column,
                verticaSource=ProtoVerticaSource(
                    host=self.host,
                    username_secret_name=self.username_secret_name,
                    password_secret_name=self.password_secret_name,
                    database=self.database,
                    schema=self.schema,
                    port=self.port,
                    table=self.table,
                ),
            )
        )
