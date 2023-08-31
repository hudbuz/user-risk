from qwak_sdk.cli import create_qwak_cli


def qwak_cli():
    return create_qwak_cli()()


if __name__ == "__main__":
    qwak_cli()()
