from .qwak_exception import QwakException


class QwakBuildException(QwakException):
    """
    Raise an error in the general case of a build failure
    """

    def __init__(
        self, message, build_id, dev, builds_management_proxy, qwak_user_id=""
    ):
        super().__init__(message)
        self.message = message
        if not dev:
            builds_management_proxy.update_build_status(
                build_id, "FAILED", qwak_user_id
            )
