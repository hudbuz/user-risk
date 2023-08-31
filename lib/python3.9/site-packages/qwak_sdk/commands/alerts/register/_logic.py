import os
from pathlib import Path
from typing import List

from qwak.clients.alerts_registry import AlertingRegistryClient
from qwak.clients.alerts_registry.channel import Channel

from qwak_sdk.inner.file_registry import extract_class_objects, list_qwak_python_files
from qwak_sdk.inner.tools.cli_tools import ask_yesno
from qwak_sdk.tools.utils import qwak_spinner

QWAK_alerts_DELIMITER = "----------------------------------------"


def _register_channels(qwak_python_files, alerts_client, force):
    """
    Register Channels

    Args:
        qwak_python_files: a list of python files containing qwak package imports
        alerts_client: AlertingRegistryClient alerts service client
        force: boolean determining if to force register all encountered Channel objects
    """
    with qwak_spinner(
        begin_text="Looking for channels to register", print_callback=print
    ):
        qwak_channels: List[Channel] = extract_class_objects(qwak_python_files, Channel)

    print(f"👀 Found {len(qwak_channels)} Channels")
    for channel, source_file_path in qwak_channels:
        channel_id, existing_channel = alerts_client.get_alerting_channel(channel.name)
        if existing_channel:
            if ask_yesno(
                f"Update existing Channel '{channel.name}' from source file '{source_file_path}'?",
                force,
            ):
                alerts_client.update_alerting_channel(
                    channel_id=channel_id, channel=channel
                )
        else:
            if ask_yesno(
                f"Create new Channel '{channel.name}' from source file '{source_file_path}'?",
                force,
            ):
                alerts_client.create_alerting_channel(channel)
    print(QWAK_alerts_DELIMITER)


def execute_register_channel(path: Path, force: bool):
    if not path:
        path = Path.cwd()
    else:
        path = Path(path)

    if path.is_file():
        qwak_python_files = [(str(path), os.path.abspath(path))]
    elif Path.is_dir(path):
        with qwak_spinner(
            begin_text="Recursively looking for python files in input dir",
            print_callback=print,
        ) as sp:
            qwak_python_files = list_qwak_python_files(path, sp)
            print(qwak_python_files)
            print(sp)
            pass

    alerts_client = AlertingRegistryClient()
    _register_channels(
        qwak_python_files=qwak_python_files,
        alerts_client=alerts_client,
        force=force,
    )
