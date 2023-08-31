from .quiet_error import QuietError


class QwakQuietBuildException(QuietError):
    def __init__(
        self, message, build_id, dev, builds_management_proxy, qwak_user_id=""
    ):
        self.message = message
        if not dev:
            builds_management_proxy.update_build_status(
                build_id, "FAILED", qwak_user_id
            )
