from click import group

from qwak_sdk.commands.models.deployments.deploy.batch.ui import batch
from qwak_sdk.commands.models.deployments.deploy.realtime.ui import realtime
from qwak_sdk.commands.models.deployments.deploy.streaming.ui import stream


@group(
    name="deploy",
    help="Model deployments",
)
def deploy_group():
    # Click group injection
    pass


deploy_group.add_command(realtime, "realtime")
deploy_group.add_command(stream, "stream")
deploy_group.add_command(batch, "batch")
