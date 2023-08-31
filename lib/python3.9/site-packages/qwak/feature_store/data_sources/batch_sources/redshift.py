from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    JdbcSource as ProtoJdbcSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    RedshiftSource as ProtoRedshiftSource,
)
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.exceptions import QwakException
from qwak.feature_store.data_sources.batch_sources._jdbc import JdbcSource


@dataclass
class RedshiftSource(JdbcSource):
    iam_role_arn: Optional[str] = None
    db_user: Optional[str] = None

    # these options are not yet supported
    access_key: Optional[str] = None
    secret_access_key: Optional[str] = None
    query_group: Optional[str] = "_qwak_featurestore"

    def _validate(self):
        authentication_methods = sum(
            [
                bool(self.access_key and self.secret_access_key),
                bool(self.iam_role_arn),
                bool(self.username_secret_name and self.password_secret_name),
            ]
        )
        if authentication_methods > 1:
            raise QwakException(
                "Only one connection method must be set, access key id and secret access key id, IAM role arn"
                "or user id secret name and password secret name"
            )
        if authentication_methods < 1:
            raise QwakException(
                "A connection method must be set, either access key id and secret access key id, IAM role arn"
                "or user id secret name and password secret name"
            )

    def _to_proto(self):
        return ProtoDataSourceSpec(
            batch_source=ProtoBatchSource(
                name=self.name,
                description=self.description,
                date_created_column=self.date_created_column,
                jdbcSource=ProtoJdbcSource(
                    url=self.url,
                    username_secret_name=self.username_secret_name,
                    password_secret_name=self.password_secret_name,
                    db_table=self.db_table,
                    query=self.query,
                    redshiftSource=ProtoRedshiftSource(
                        db_user=self.db_user,
                        iam_role_arn=self.iam_role_arn,
                        access_key_id=self.access_key,
                        secret_access_key=self.secret_access_key,
                    ),
                ),
            )
        )

    @classmethod
    def _from_proto(cls, proto: ProtoBatchSource):
        redshift = proto.jdbcSource
        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            url=redshift.url,
            username_secret_name=redshift.username_secret_name,
            password_secret_name=redshift.password_secret_name,
            db_table=redshift.db_table,
            query=redshift.query,
            iam_role_arn=redshift.redshiftSource.iam_role_arn,
            db_user=redshift.redshiftSource.db_user,
        )
