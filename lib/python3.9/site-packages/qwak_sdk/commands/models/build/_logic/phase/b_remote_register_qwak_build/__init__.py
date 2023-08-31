from .cleanup_step import CleanupStep
from .start_remote_build_step import StartRemoteBuildStep
from .upload_step import UploadStep


def get_remote_register_qwak_build_steps():
    return [
        UploadStep(),
        StartRemoteBuildStep(),
        CleanupStep(),
    ]
