from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    ElasticsearchSource as ProtoElasticsearchSource,
)
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource


@dataclass
class ElasticSearchSource(BaseBatchSource):
    nodes: str
    resource: str
    port: Optional[int] = None
    query: Optional[str] = None
    username_secret_name: Optional[str] = None
    password_secret_name: Optional[str] = None
    exclude_fields: Optional[str] = None
    parse_dates: Optional[bool] = True

    @classmethod
    def _from_proto(cls, proto):
        elastic = proto.elasticsearchSource
        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            nodes=elastic.nodes,
            resource=elastic.resource,
            port=elastic.port,
            query=elastic.query,
            username_secret_name=elastic.username_secret_name,
            password_secret_name=elastic.password_secret_name,
            exclude_fields=elastic.exclude_fields,
            parse_dates=elastic.parse_dates,
        )

    def _to_proto(self):
        return ProtoDataSourceSpec(
            batch_source=ProtoBatchSource(
                name=self.name,
                description=self.description,
                date_created_column=self.date_created_column,
                elasticsearchSource=ProtoElasticsearchSource(
                    nodes=self.nodes,
                    resource=self.resource,
                    port=self.port,
                    query=self.query,
                    username_secret_name=self.username_secret_name,
                    password_secret_name=self.password_secret_name,
                    exclude_fields=self.exclude_fields,
                    parse_dates=self.parse_dates,
                ),
            )
        )
