import grpc
from qwak.inner.tool.auth import Auth0ClientBase

_SIGNATURE_HEADER_KEY = "authorization"


class Auth0Client(grpc.AuthMetadataPlugin, Auth0ClientBase):
    def __init__(self):
        from qwak.inner.di_configuration import UserAccountConfiguration

        user_account = UserAccountConfiguration().get_user_config()
        api_key = user_account.api_key

        Auth0ClientBase.__init__(self, api_key=api_key)

    def __call__(self, context, callback):
        callback(((_SIGNATURE_HEADER_KEY, "Bearer {}".format(self.get_token())),), None)
