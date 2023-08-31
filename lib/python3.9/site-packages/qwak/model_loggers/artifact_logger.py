import urllib.request
from typing import Optional

from qwak.clients.build_orchestrator.client import BuildOrchestratorClient
from qwak.clients.file_versioning.client import FileVersioningManagementClient
from qwak.exceptions import QwakException
from qwak.inner.model_loggers_utils import (
    fetch_build_id,
    upload_data,
    validate_model,
    validate_tag,
)


def log_file(
    from_path: str,
    tag: str = "artifact",
    model_id: Optional[str] = None,
    build_id: Optional[str] = None,
) -> None:
    """
    Log a file by a given tag

    Args:
        from_path: file path to log
        tag: tag to save the file with
        model_id: optional model id to save data with - if not given found from environment
        build_id: optional build id - if not given found from environment.
    """
    if not validate_tag(tag):
        raise QwakException(
            "Tag should contain only letters, numbers, underscore or hyphen"
        )

    model_id = validate_model(model_id)
    if not build_id:
        # Checking if called inside a model - then build id saved as environment variable or stays
        build_id = fetch_build_id()

    upload_url_response = BuildOrchestratorClient().get_build_versioning_upload_url(
        build_id=build_id, model_id=model_id, tag=tag
    )

    FileVersioningManagementClient().register_file_tag(
        model_id, tag, from_path, build_id
    )

    with open(from_path, mode="rb") as f:
        upload_data(upload_url_response.upload_url, f.read())


def load_file(
    to_path: str,
    tag: str = "artifact",
    model_id: Optional[str] = None,
    build_id: Optional[str] = None,
) -> str:
    """
    Load a file by a given tag

    Args:
        to_path: the local path the downloaded file will be written to
        tag: load the artifact with this given tag
        model_id: optional model id to save data with - if not given found from environment
        build_id: optional build id - if not given found from environment.

    Returns:
        the path to the newly created data file
    """
    if not validate_tag(tag):
        raise QwakException(
            "Tag should contain only letters, numbers, underscore or hyphen"
        )

    model_id = validate_model(model_id)
    download_url_response = BuildOrchestratorClient().get_build_versioning_download_url(
        build_id=build_id, model_id=model_id, tag=tag
    )

    try:
        filename, headers = urllib.request.urlretrieve(  # nosec B310
            url=download_url_response.download_url, filename=to_path
        )

        return filename
    except Exception as error:
        raise QwakException(f"Unable to load save artifact locally: {str(error)}")
