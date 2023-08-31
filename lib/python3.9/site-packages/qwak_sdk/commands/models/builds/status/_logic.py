from _qwak_proto.qwak.builds.builds_pb2 import BuildStatus
from qwak.clients.build_management import BuildsManagementClient


def execute_get_build_status(build_id) -> BuildStatus:
    return BuildsManagementClient().get_build(build_id).build.build_status
