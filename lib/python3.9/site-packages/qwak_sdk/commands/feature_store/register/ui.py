import os
from pathlib import Path

import click
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.clients.feature_store.operator_client import FeaturesOperatorClient

from qwak_sdk.commands.feature_store.register._logic import (
    _register_data_sources,
    _register_entities,
    _register_features_sets,
)
from qwak_sdk.inner.file_registry import list_qwak_python_files
from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.utils import qwak_spinner


@click.command(
    "register",
    cls=QwakCommand,
    help="Register and deploy all feature store object under the given path. Registered "
    "features will be visible on the Qwak management platform after registration",
)
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    metavar="PATH",
    help="Directory / module where qwak feature store objects are stored",
)
@click.option(
    "--force",
    "-f",
    default=False,
    is_flag=True,
    metavar="FLAG",
    help="Force register all found qwak Feature Store objects",
)
@click.option(
    "--no-validation",
    "-nv",
    default=False,
    is_flag=True,
    metavar="FLAG",
    help="Skip validation for all found qwak Feature Store objects",
)
@click.option(
    "--ignore-validation-errors",
    "-ive",
    default=False,
    is_flag=True,
    metavar="FLAG",
    help="Ignore validation errors. Continue registering even if an error occurs",
)
def register_fs_objects(
    path: Path,
    force: bool,
    no_validation: bool,
    ignore_validation_errors: bool,
    **kwargs,
):
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

    try:
        import git

        git_commit = git.Repo(path, search_parent_directories=True).head.commit.hexsha
    except Exception:
        # be super defensive on Git errors. Failing to fetch anything git related should not fail the registration
        git_commit = None

    registry_client = FeatureRegistryClient()
    operator_client = FeaturesOperatorClient()
    _register_entities(qwak_python_files, registry_client, force)

    _register_data_sources(
        qwak_python_files,
        registry_client,
        operator_client,
        force,
        no_validation,
        ignore_validation_errors,
    )

    _register_features_sets(
        qwak_python_files,
        registry_client,
        operator_client,
        force,
        git_commit,
        no_validation,
        ignore_validation_errors,
    )
