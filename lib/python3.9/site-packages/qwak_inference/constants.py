from os import getenv
from pathlib import Path


class QwakConstants:
    """
    Qwak Configuration settings
    """

    QWAK_HOME = (
        getenv("QWAK_HOME")
        if getenv("QWAK_HOME") is not None
        else f"{str(Path.home())}"
    )

    QWAK_CONFIG_FOLDER: str = f"{QWAK_HOME}/.qwak"

    QWAK_CONFIG_FILE: str = f"{QWAK_CONFIG_FOLDER}/config"

    QWAK_AUTHORIZATION_FILE: str = f"{QWAK_CONFIG_FOLDER}/auth"

    QWAK_DEFAULT_SECTION: str = "default"

    QWAK_AUTHENTICATION_URL = "https://grpc.qwak.ai/api/v1/authentication/qwak-api-key"

    QWAK_AUTHENTICATED_USER_ENDPOINT: str = (
        "https://grpc.qwak.ai/api/v0/runtime/get-authenticated-user-context"
    )
