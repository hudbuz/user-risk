from os import getenv
from pathlib import Path

from _qwak_proto.qwak.batch_job.v1.batch_job_resources_pb2 import GpuType


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

    QWAK_LOGGER_FOLDER: str = f"{QWAK_CONFIG_FOLDER}/logs"

    QWAK_CONFIG_FILE: str = f"{QWAK_CONFIG_FOLDER}/config"

    QWAK_AUTHORIZATION_FILE: str = f"{QWAK_CONFIG_FOLDER}/auth"

    QWAK_DEFAULT_SECTION: str = "default"

    AUTH0_JWKS_URI = "https://dev-qwak.us.auth0.com/.well-known/jwks.json"

    AUTH0_ALGORITHMS = ["RS256"]

    GPU_TYPES = list(set(GpuType.DESCRIPTOR.values_by_name) - {"INVALID_GPU"})

    TOKEN_AUDIENCE: str = "https://auth-token.qwak.ai/"  # nosec B105

    QWAK_AUTHENTICATION_URL = "https://grpc.qwak.ai/api/v1/authentication/qwak-api-key"
