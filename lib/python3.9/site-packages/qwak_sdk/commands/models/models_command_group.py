import click

from qwak_sdk.commands.auto_scalling.autoscaling_commands_group import (
    autoscaling_commands_group,
)

from .build.ui import models_build
from .builds.builds_commands_group import builds_commands_group
from .create.ui import model_create
from .delete.ui import model_delete
from .deployments.deploy.deploy_commands_group import deploy_group
from .deployments.undeploy.ui import models_undeploy
from .describe.ui import model_describe
from .executions.execution_commands_group import execution_commands_group
from .init.ui import model_init
from .list.ui import model_list
from .runtime.runtime_commands_group import runtime_commands_group


@click.group(
    name="models",
    help="Commands for interacting with Qwak based models",
)
def models_command_group():
    # Click group injection
    pass


models_command_group.add_command(autoscaling_commands_group)
models_command_group.add_command(builds_commands_group)
models_command_group.add_command(execution_commands_group)
models_command_group.add_command(model_init)
models_command_group.add_command(model_create)
models_command_group.add_command(model_delete)
models_command_group.add_command(model_list)
models_command_group.add_command(models_build)
models_command_group.add_command(deploy_group, "deploy")
models_command_group.add_command(models_undeploy, "undeploy")
models_command_group.add_command(model_describe)
models_command_group.add_command(runtime_commands_group)
