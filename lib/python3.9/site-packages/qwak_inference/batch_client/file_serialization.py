import gzip
from abc import ABC, abstractmethod
from io import BytesIO, StringIO

import pandas as pd


class SerializationFormat(ABC):
    """
    Pandas serialization format - as a file transfer format between the batch client and workers
    """

    @property
    def format_key(self) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_bytes(df: pd.DataFrame) -> bytes:
        pass

    @staticmethod
    @abstractmethod
    def read_df(data: BytesIO) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def get_file_type() -> str:
        pass


class FeatherSerialization(SerializationFormat):
    format_key = "FEATHER"

    @staticmethod
    def get_bytes(df: pd.DataFrame) -> bytes:
        bytes_buffer = BytesIO()
        df.reset_index(drop=True).to_feather(bytes_buffer)

        return bytes_buffer.getvalue()

    @staticmethod
    def read_df(data: BytesIO) -> pd.DataFrame:
        return pd.read_feather(data)

    @staticmethod
    def get_file_type() -> str:
        return "feather"


class ParquetSerialization(SerializationFormat):
    format_key = "PARQUET"

    @staticmethod
    def get_bytes(df: pd.DataFrame) -> bytes:
        bytes_buffer = BytesIO()
        df.to_parquet(bytes_buffer)

        return bytes_buffer.getvalue()

    @staticmethod
    def read_df(data: BytesIO) -> pd.DataFrame:
        return pd.read_parquet(data)

    @staticmethod
    def get_file_type() -> str:
        return "parquet"


class CSVSerialization(SerializationFormat):
    format_key = "CSV"

    @staticmethod
    def get_bytes(df: pd.DataFrame) -> bytes:
        string_buffer = StringIO()
        df.to_csv(string_buffer, index=False)

        return gzip.compress(bytes(string_buffer.getvalue(), "utf-8"))

    @staticmethod
    def read_df(data: BytesIO) -> pd.DataFrame:
        return pd.read_csv(data)

    @staticmethod
    def get_file_type() -> str:
        return "csv.gz"


SERIALIZATION_HANDLER_MAP = {
    FeatherSerialization.format_key: FeatherSerialization(),
    ParquetSerialization.format_key: ParquetSerialization(),
    CSVSerialization.format_key: CSVSerialization(),
}
