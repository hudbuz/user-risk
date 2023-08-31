from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import CsvSource as ProtoCsvSource
from _qwak_proto.qwak.feature_store.sources.data_source_pb2 import (
    DataSourceSpec as ProtoDataSourceSpec,
)
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource
from qwak.feature_store.data_sources.batch_sources.filesystem_config import (
    FileSystemConfiguration,
    get_fs_config_from_proto,
)


@dataclass
class CsvSource(BaseBatchSource):
    path: str
    quote_character: str = '"'
    escape_character: str = '"'
    filesystem_configuration: Optional[FileSystemConfiguration] = None

    @classmethod
    def _from_proto(cls, proto):
        csv = proto.csvSource

        fs_conf = get_fs_config_from_proto(csv.filesystem_configuration)

        return cls(
            name=proto.name,
            date_created_column=proto.date_created_column,
            description=proto.description,
            path=csv.path,
            quote_character=csv.quote_character,
            escape_character=csv.escape_character,
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
                csvSource=ProtoCsvSource(
                    path=self.path,
                    quote_character=self.quote_character,
                    escape_character=self.escape_character,
                    filesystem_configuration=fs_conf,
                ),
            )
        )
