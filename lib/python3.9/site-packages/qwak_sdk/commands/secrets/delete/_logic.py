from qwak.clients.secret_service import SecretServiceClient


def execute_delete_secret(name):
    SecretServiceClient().delete_secret(name)
