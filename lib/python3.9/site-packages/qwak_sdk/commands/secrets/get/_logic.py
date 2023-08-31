from qwak.clients.secret_service import SecretServiceClient


def execute_get_secret(name):
    return SecretServiceClient().get_secret(name)
