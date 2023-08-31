import click

from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.log_handling import QwakLogHandling


@click.command("logs", cls=QwakCommand)
@click.option("-f", "--follow", is_flag=True, default=False, help="Follow log tail")
@click.option(
    "-b", "--build-id", required=True, help="Runtime model Build ID to show logs of"
)
@click.option(
    "-s",
    "--since",
    required=False,
    metavar="X [TYPE] ago",
    help=f"Get logs from X1 [type] X2 [type]... Xn [type] ago. "
    f"(x = INT; type = {QwakLogHandling.TIME_UNITS}) i.e. [1 day 2 hours ago]",
)
@click.option(
    "-n",
    "--number-of-results",
    required=False,
    type=int,
    help="Maximum number of results per query",
)
@click.option(
    "-g", "--grep", required=False, help="Filter by log content contains [expression]"
)
def build_logs(
    follow=True, build_id=None, since=None, number_of_results=None, grep=None, **kwargs
):
    QwakLogHandling().get_logs(
        follow, since, number_of_results, grep, {"build_id": build_id}, "build"
    )
