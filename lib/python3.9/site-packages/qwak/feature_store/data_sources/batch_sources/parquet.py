from dataclasses import dataclass, field

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    ParquetSource as ProtoParquetSource,
)
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource
from qwak.feature_store.data_sources.batch_sources.filesystem_config import (
    AnonymousS3Configuration,
    FileSystemConfiguration,
    get_fs_config_from_proto,
)


@dataclass
class ParquetSource(BaseBatchSource):
    path: str
    filesystem_configuration: FileSystemConfiguration = field(
        default_factory=lambda: AnonymousS3Configuration()
    )

    @classmethod
    def _from_proto(cls, proto):
        parquet: ProtoParquetSource = proto.parquetSource
        fs_conf = get_fs_config_from_proto(parquet.filesystem_configuration)

        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            path=parquet.path,
            filesystem_configuration=fs_conf,
        )

    def _to_proto(self):
        fs_conf = None
        if self.filesystem_configuration:
            fs_conf = self.filesystem_configuration._to_proto()

        return ProtoDataSourceSpec(
            batch_source=ProtoBatchSource(
                name=self.name,
                description=self.description,
                date_created_column=self.date_created_column,
                parquetSource=ProtoParquetSource(
                    path=self.path, filesystem_configuration=fs_conf
                ),
            )
        )
