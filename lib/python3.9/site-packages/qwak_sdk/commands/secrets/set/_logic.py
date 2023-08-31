from qwak.clients.secret_service import SecretServiceClient


def execute_set_secret(name, value):
    SecretServiceClient().set_secret(name, value)
