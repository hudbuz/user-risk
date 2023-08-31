import click
from qwak.inner.const import QwakConstants

from qwak_sdk.commands.models.build._logic.client_logs.cli_ui import (
    execute_build_pipeline,
)
from qwak_sdk.commands.models.build._logic.config.config_v1 import ConfigV1
from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.inner.tools.config_handler import config_handler


@click.command("build", cls=QwakCommand)
@click.option("--model-id", metavar="NAME", required=False, help="Model ID")
@click.option(
    "--main-dir",
    metavar="NAME",
    help=f"Model main directory name, [Default: {ConfigV1.BuildProperties.ModelUri.main_dir}]",
)
@click.option(
    "-P",
    "--param-list",
    required=False,
    metavar="NAME=VALUE",
    multiple=True,
    help="A parameter for the build, of the form -P name=value. the params will be saved and can be viewed later",
)
@click.option(
    "-E",
    "--env-vars",
    required=False,
    metavar="NAME=VALUE",
    multiple=True,
    help="A parameter for the build, of the form -E name=value",
)
@click.option(
    "-T",
    "--tags",
    required=False,
    multiple=True,
    help="A tag for the model build",
)
@click.option(
    "--git-credentials",
    required=False,
    metavar="USERNAME:ACCESS_TOKEN",
    help="Access credentials for private repositories listed in the python dependencies file",
)
@click.option(
    "--git-branch",
    metavar="NAME",
    required=False,
    help=f"Branch to use for git repo model code if defined."
    f"\n[Default: {ConfigV1.BuildProperties.ModelUri.git_branch}]",
)
@click.option(
    "--git-credentials-secret",
    metavar="NAME",
    required=False,
    help="[REMOTE BUILD] Predefined Qwak secret secret name, that contains access credentials to private repositories"
    + "Secrets should be of the form USERNAME:ACCESS_TOKEN. For info regarding defining Qwak Secrets using the"
    + "`qwak secret` command",
)
@click.option(
    "--cpus",
    metavar="NAME",
    required=False,
    help="[REMOTE BUILD] Number of cpus to use on the remote build. [Default (If GPU not configured): 2] "
    "(DO NOT CONFIGURE GPU AND CPU TOGETHER)",
    type=click.FLOAT,
)
@click.option(
    "--memory",
    metavar="NAME",
    required=False,
    help="[REMOTE BUILD] Memory to use on the remote build. [Default (If GPU not configured): 4Gi] "
    "(DO NOT CONFIGURE GPU AND CPU TOGETHER)",
)
@click.option(
    "--gpu-type",
    metavar="NAME",
    required=False,
    help=f"[REMOTE BUILD] Type of GPU to use on the remote build ({', '.join([x for x in QwakConstants.GPU_TYPES])})."
    f"\n[Default: {ConfigV1.BuildEnv.RemoteConf.RemoteBuildResources.gpu_type}]"
    "(DO NOT CONFIGURE GPU AND CPU TOGETHER)",
    type=click.STRING,
)
@click.option(
    "--gpu-amount",
    metavar="NAME",
    required=False,
    type=int,
    help=f"[REMOTE BUILD] Amount of GPU to use on the remote build."
    f"\n[Default: {ConfigV1.BuildEnv.RemoteConf.RemoteBuildResources.gpu_amount}] "
    "(DO NOT CONFIGURE GPU AND CPU TOGETHER)",
)
@click.option(
    "--gpu-compatible",
    help=f"[REMOTE BUILD] Whether to build an image that is compatible to be deployd on a GPU instance."
    f"\n[Default: {ConfigV1.BuildProperties.gpu_compatible}] ",
    default=False,
    is_flag=True,
)
@click.option(
    "--iam-role-arn",
    required=False,
    type=str,
    help="[REMOTE BUILD] Custom IAM Role ARN.",
)
@click.option(
    "--cache/--no-cache",
    default=None,
    help="Disable docker build cache. [Default: Cache enabled]",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    default=None,
    help="Log verbosity level - v: INFO, vv: DEBUG [default: WARNING], Default ERROR",
)
@click.option(
    "--base-image",
    help="Used for customizing the docker container image built for train, build and deploy."
    "Docker images should be based on qwak images, The entrypoint or cmd of the docker "
    "image should not be changed."
    f"\n[Default: {ConfigV1.BuildEnv.DockerConf.base_image}]",
    required=False,
)
@click.option(
    "-f",
    "--from-file",
    help="Build by run_config file, Command arguments will overwrite any run_config.",
    required=False,
    type=click.Path(exists=True, resolve_path=True, dir_okay=False),
)
@click.option(
    "--out-conf",
    help="Extract models build conf from command arguments, the command will not run it wil only output valid yaml "
    "structure",
    default=False,
    is_flag=True,
)
@click.option(
    "--json-logs",
    help="Output logs as json for easier parsing",
    default=False,
    is_flag=True,
)
@click.option(
    "--programmatic",
    help="Run the _logic without the UI and receive the build id and any exception as return values",
    default=False,
    is_flag=True,
)
@click.option(
    "--validate-build-artifact/--no-validate-build-artifact",
    help="Skip validate build artifact step",
    default=None,
)
@click.option(
    "--tests/--no-tests",
    help="Skip tests step",
    default=None,
)
@click.option(
    "--dependency-file-path",
    help="Custom dependency file path",
    default=None,
)
@click.option(
    "--validate-build-artifact-timeout",
    help="Timeout in seconds for the validation step",
    default=120,
)
@click.option(
    "--dependency-required-folders",
    help="Comma separated list of folders to be copied into the build",
    default=None,
    required=False,
    multiple=True,
)
@click.option(
    "--deploy",
    help="Whether you want to deploy the build if it finishes successfully. "
    "Choosing this will follow the build process in the terminal and will trigger a deployment when the "
    "build finishes.",
    default=False,
    is_flag=True,
)
@click.option(
    "--instance",
    required=False,
    type=str,
    help="The instance size to build on - 'small', 'medium', 'large', etc...",
    default=None,
)
@click.option(
    "--deployment-instance",
    required=False,
    type=str,
    help="The instance size to deploy on - 'small', 'medium', 'large', etc...",
    default=None,
)
@click.argument("uri", required=False)
def models_build(**kwargs):
    return build(**kwargs)


def build(
    from_file: str,
    out_conf: bool,
    json_logs: bool,
    programmatic: bool,
    **kwargs,
):
    # If QWAK_DEBUG=true is set then the artifacts will not be deleted, all intermediate files located in ~/.qwak/builds
    # Including all intermediate images
    config: ConfigV1 = config_handler(
        config=ConfigV1,
        from_file=from_file,
        out_conf=out_conf,
        **kwargs,
    )
    if out_conf:
        return
    else:
        use_git_credentials = kwargs.get("use_git_credentials")
        return execute_build_pipeline(
            config=config,
            json_logs=json_logs,
            programmatic=programmatic,
            use_git_credentials=use_git_credentials,
        )
