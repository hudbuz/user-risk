from dataclasses import dataclass

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    MongoSource as ProtoMongoSource,
)
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource


@dataclass
class MongoDbSource(BaseBatchSource):
    hosts: str
    username_secret_name: str
    password_secret_name: str
    database: str
    collection: str
    connection_params: str
    protocol: str = "mongodb"

    @classmethod
    def _from_proto(cls, proto):
        mongo = proto.mongoSource
        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            hosts=mongo.hosts,
            username_secret_name=mongo.username_secret_name,
            password_secret_name=mongo.password_secret_name,
            database=mongo.database,
            collection=mongo.collection,
            connection_params=mongo.connection_params,
            protocol=mongo.protocol,
        )

    def _to_proto(self):
        return ProtoDataSourceSpec(
            batch_source=ProtoBatchSource(
                name=self.name,
                description=self.description,
                date_created_column=self.date_created_column,
                mongoSource=ProtoMongoSource(
                    hosts=self.hosts,
                    username_secret_name=self.username_secret_name,
                    password_secret_name=self.password_secret_name,
                    database=self.database,
                    collection=self.collection,
                    connection_params=self.connection_params,
                    protocol=self.protocol,
                ),
            )
        )
