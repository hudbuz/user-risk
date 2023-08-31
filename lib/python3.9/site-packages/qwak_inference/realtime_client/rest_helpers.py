import socket
from datetime import datetime

import requests

from qwak_inference.configuration import AuthClient
from qwak_inference.configuration.account import UserAccountConfiguration


def _get_authorization():
    user_account_configuration = UserAccountConfiguration()
    auth_client = AuthClient(api_key=user_account_configuration.get_user_apikey())

    return f"Bearer {auth_client.get_token()}", auth_client.token_expiration()


class HTTPAdapterWithSocketOptions(requests.adapters.HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.socket_options = kwargs.pop("socket_options", None)
        super(HTTPAdapterWithSocketOptions, self).__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if self.socket_options is not None:
            kwargs["socket_options"] = self.socket_options
        super(HTTPAdapterWithSocketOptions, self).init_poolmanager(*args, **kwargs)


# Configure only TCP attributes that are available in the OS
def validate_socket_config(socket_options):
    config = []
    for line in socket_options:
        if hasattr(socket, line[0]) and hasattr(socket, line[1]):
            config.append((getattr(socket, line[0]), getattr(socket, line[1]), line[2]))
    return config


class RestSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.headers.update({"Content-Type": "application/json"})
        socket_options = [
            ("SOL_SOCKET", "SO_KEEPALIVE", 1),
            ("SOL_TCP", "TCP_KEEPIDLE", 120),
            ("SOL_TCP", "TCP_KEEPINTVL", 75),
            ("SOL_TCP", "TCP_KEEPCNT", 9),
        ]
        socket_options = validate_socket_config(socket_options)
        adapter = HTTPAdapterWithSocketOptions(socket_options=socket_options)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

    def prepare_request(self, request):
        if "Authorization" not in self.headers:
            (
                self.headers["Authorization"],
                self.jwt_expiration,
            ) = _get_authorization()
        else:
            if self.jwt_expiration <= datetime.utcnow():
                (
                    self.headers["Authorization"],
                    self.jwt_expiration,
                ) = _get_authorization()

        return super().prepare_request(request)
