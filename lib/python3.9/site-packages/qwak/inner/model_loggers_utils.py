import os
import re
from typing import Optional

import requests
from qwak.clients.model_management.client import ModelsManagementClient
from qwak.exceptions import QwakException

TAG_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_tag(tag: str) -> bool:
    """
    Check if tag exists.

    Args:
        tag: tag to check

    Returns: If tag is valid
    """
    return re.match(TAG_REGEX, tag) is not None


def fetch_model_id() -> Optional[str]:
    """
    Get model id from environment.

    Returns: model id if found.

    Notes:
        1. Checking if called inside a model - then model id saved as environment variable.
    """
    # Checking if called inside a model - then model id saved as environment variable
    return os.getenv("QWAK_MODEL_ID", None)


def validate_model(model_id: str) -> str:
    """
    Validate a model ID validity and existence
    """
    if not model_id:
        model_id = fetch_model_id()
        if not model_id:
            raise QwakException("Failed to determined model ID.")

    try:
        ModelsManagementClient().get_model(model_id=model_id)
    except Exception:
        raise QwakException("Failed to find model.")

    return model_id


def fetch_build_id() -> Optional[str]:
    """
    Get Build id from environment

    Returns: Build id if found

    Notes:
        1. Checking if called inside a model - then build id saved as environment variable.
    """
    # Checking if called inside a model - then model id saved as environment variable
    return os.getenv("QWAK_BUILD_ID", None)


def upload_data(
    upload_url: str,
    data: bytes,
    content_type: str = "text/plain",
) -> None:
    """
    Upload data
    Args:
        upload_url: the url to upload to.
        data: the data to upload
        content_type: Uploaded content-type
    """
    try:
        http_response = requests.put(  # nosec B113
            upload_url,
            data=data,
            headers={"content-type": content_type},
        )

        if http_response.status_code != 200:
            raise QwakException(
                f"Failed to upload data. "
                f"Status: [{http_response.status_code}], "
                f"reason: [{http_response.reason}]"
            )
    except Exception as e:
        raise QwakException(f"Failed to upload data. Error is {e}")
