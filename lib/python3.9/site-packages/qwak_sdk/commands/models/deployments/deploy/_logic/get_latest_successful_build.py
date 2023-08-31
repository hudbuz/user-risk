from copy import deepcopy

from _qwak_proto.qwak.builds.builds_pb2 import SUCCESSFUL
from qwak.clients.build_management.client import BuildsManagementClient
from qwak.clients.model_management.client import ModelsManagementClient
from qwak.exceptions import QwakException


def get_latest_successful_build_from_model(model_id: str) -> str:
    """Get the latest successful build from model

    Args:
        model_id: Model id to search build for.

    Returns:
        str: Latest build id.

    Raises:
        QwakException: if no successful build exists in model.
    """
    model = ModelsManagementClient().get_model(model_id)
    model_uuid = model.uuid
    last_successful_build = BuildsManagementClient().list_builds(model_uuid)
    builds = deepcopy(last_successful_build.builds)
    builds = [build for build in builds if build.build_status == SUCCESSFUL]
    builds.sort(key=lambda r: r.created_at.seconds)

    if not builds:
        raise QwakException(f"Unable to find successful build for model {model_id}")

    return builds[-1].build_spec.build_id
