from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    AnonymousS3Configuration as ProtoAnonymousS3Configuration,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    AwsS3FileSystemConfiguration as ProtoAwsS3FileSystemConfiguration,
)
from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    FileSystemConfiguration as FileSystemConfigurationProto,
)
from qwak.exceptions import QwakException


@dataclass
class FileSystemConfiguration(ABC):
    @abstractmethod
    def _to_proto(self):
        pass

    @abstractmethod
    def _from_proto(self, proto):
        pass


@dataclass
class AnonymousS3Configuration(FileSystemConfiguration):
    def _to_proto(self):
        return FileSystemConfigurationProto(
            aws_s3_anonymous=ProtoAnonymousS3Configuration()
        )

    @classmethod
    def _from_proto(cls, proto):
        return cls()


@dataclass
class AwsS3FileSystemConfiguration(FileSystemConfiguration):
    access_key_secret_name: str
    secret_key_secret_name: str
    bucket: str
    session_token_secret_name: Optional[str] = ""

    def __post_init__(self):
        self._validate()

    def _validate(self):
        error_msg = "{field} field is mandatory"
        if not self.access_key_secret_name:
            raise QwakException(error_msg.format(field="access_key"))
        if not self.secret_key_secret_name:
            raise QwakException(error_msg.format(field="secret_key"))
        if not self.bucket:
            raise QwakException(error_msg.format(field="bucket"))

    def _to_proto(self):
        return FileSystemConfigurationProto(
            aws_s3_configuration=ProtoAwsS3FileSystemConfiguration(
                access_key_secret_name=self.access_key_secret_name,
                secret_key_secret_name=self.secret_key_secret_name,
                bucket=self.bucket,
                session_token_secret_name=self.session_token_secret_name,
            )
        )

    @classmethod
    def _from_proto(cls, proto):
        return AwsS3FileSystemConfiguration(
            access_key_secret_name=proto.access_key_secret_name,
            secret_key_secret_name=proto.secret_key_secret_name,
            bucket=proto.bucket,
            session_token_secret_name=proto.session_token_secret_name,
        )


def get_fs_config_from_proto(filesystem_conf):
    if not filesystem_conf:
        return {}

    fs_conf_type = filesystem_conf.WhichOneof("type")

    if not fs_conf_type:
        return {}

    fs_conf = getattr(filesystem_conf, fs_conf_type)
    if isinstance(fs_conf, ProtoAwsS3FileSystemConfiguration):
        return AwsS3FileSystemConfiguration._from_proto(fs_conf)

    elif isinstance(fs_conf, ProtoAnonymousS3Configuration):
        return AnonymousS3Configuration._from_proto(fs_conf)
    else:
        raise QwakException(f"Unsupported FileSystemConfiguration: {fs_conf_type}")
