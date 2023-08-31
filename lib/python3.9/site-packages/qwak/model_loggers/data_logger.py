import gzip
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from _qwak_proto.qwak.builds.build_pb2 import (
    CsvFormat,
    DataColumnDefinition,
    DataColumnType,
    DataFormat,
    DataTableDefinition,
)
from qwak.clients.build_orchestrator.client import BuildOrchestratorClient
from qwak.clients.data_versioning.client import DataVersioningManagementClient
from qwak.exceptions import QwakException
from qwak.inner.model_loggers_utils import (
    fetch_build_id,
    upload_data,
    validate_model,
    validate_tag,
)

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pass


def log_data(
    dataframe: "pd.DataFrame",
    tag: str,
    model_id: Optional[str] = None,
    build_id: Optional[str] = None,
    encoding: str = "utf-8",
) -> None:
    """
    Log data by specific parameters

    Args:
        dataframe: data to log
        tag: tag to save data with
        model_id: optional model id to save data with - if not given found from environment
        build_id: optional build id - if not given found from environment.
        encoding: Encoding for the dataframe - utf-8 is default.
    """
    if not validate_tag(tag):
        raise QwakException(
            "Tag should contain only letters, numbers, underscore or hyphen"
        )

    data_file_extension = "data.csv.gz"
    data_tag = Path(tag) / data_file_extension
    model_id = validate_model(model_id)

    if not build_id:
        # If called inside a model - build id saved as environment variable
        build_id = fetch_build_id()

    client = BuildOrchestratorClient()
    upload_url_response = client.get_build_versioning_upload_url(
        build_id=build_id, model_id=model_id, tag=str(data_tag)
    )

    string_buffer = StringIO()
    dataframe.to_csv(string_buffer, index=False, escapechar="\\", encoding=encoding)
    upload_data(
        upload_url=upload_url_response.upload_url,
        data=gzip.compress(bytes(string_buffer.getvalue(), "utf-8")),
        content_type="text/plain",
    )

    dataframe_definition = DataTableDefinition(
        columns=[
            DataColumnDefinition(name=str(name), type=map_column_type(str(col_type)))
            for name, col_type in dataframe.dtypes.to_dict().items()
        ],
        data_format=DataFormat(
            csv=CsvFormat(delimiter=",", escape_char="\\", quote_char='"')
        ),
    )

    client.define_build_data_table(
        build_id=build_id, model_id=model_id, tag=tag, table=dataframe_definition
    )

    DataVersioningManagementClient().register_data_tag(
        model_id=model_id, build_id=build_id, tag=tag, extension=data_file_extension
    )


def load_data(
    tag: str,
    model_id: Optional[str] = None,
    build_id: Optional[str] = None,
    encoding: str = "utf-8",
    compressed: bool = True,
) -> "pd.DataFrame":
    """
    Load data by specific parameters

    Args:
        tag: tag to load data from
        model_id: model id to load data with - if not given found from environment
        build_id: optional build id - if not given found from environment.
        encoding: Encoding for the dataframe - utf-8 is default.
        compressed: read a compressed dataframe. Since Qwak version 0.9.39 all logged data frames are compressed by
         default. In order to read a dataframe persisted with an SDK version prior to 0.9.39, add compressed=False.

    Returns:
        wanted dataframe
    """
    if not validate_tag(tag):
        raise QwakException(
            "Tag should contain only letters, numbers, underscore or hyphen"
        )

    file_extension = "csv.gz" if compressed else "csv"
    data_tag = f"{tag}/data.{file_extension}"

    model_id = validate_model(model_id)

    client = BuildOrchestratorClient()
    download_url_response = client.get_build_versioning_download_url(
        build_id=build_id, model_id=model_id, tag=data_tag
    )

    additional_args = {"compression": "gzip"} if compressed else {}
    try:
        import pandas as pd
    except ImportError:
        raise QwakException(
            "Missing Pandas dependency required for logging a dataframe"
        )
    try:
        return pd.read_csv(
            download_url_response.download_url, encoding=encoding, **additional_args
        )
    except Exception as error:
        raise QwakException("Unable to load data: %s" % str(error))


def map_column_type(type_name: str) -> DataColumnType:
    data_type = type_mapper.get(type_name, DataColumnType.INVALID_COLUMN_TYPE)
    if data_type == DataColumnType.INVALID_COLUMN_TYPE:
        raise QwakException(
            f"Failed to map data type {type_name} to Qwak type, which is required for table registration."
            f" Supported types are: {list(type_mapper.keys())}"
        )

    return data_type


type_mapper = {
    "object": DataColumnType.OBJECT,
    "uint8": DataColumnType.INT,
    "int64": DataColumnType.INT,
    "float64": DataColumnType.FLOAT,
    "datetime64": DataColumnType.DATETIME,
    "datetime64[ns]": DataColumnType.DATETIME,
    "datetime64[ns, UTC]": DataColumnType.DATETIME,
    "bool": DataColumnType.BOOLEAN,
}
