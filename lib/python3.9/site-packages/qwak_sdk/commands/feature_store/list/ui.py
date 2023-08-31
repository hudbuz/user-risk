from datetime import datetime
from typing import List, Tuple

import click
from _qwak_proto.qwak.feature_store.features.feature_set_pb2 import (
    FeaturesetSchedulingState,
    FeatureStatus,
)
from _qwak_proto.qwak.feature_store.features.feature_set_service_pb2 import (
    GetFeaturesetSchedulingStateResponse,
    ListFeatureSetsResponse,
)
from qwak.clients.feature_store import FeatureRegistryClient
from tabulate import tabulate

from qwak_sdk.inner.tools.cli_tools import QwakCommand


def _pts_to_datetime(pts) -> datetime:
    return datetime.fromtimestamp(pts.seconds + pts.nanos / 1e9)


def _datetime_to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%b-%d %H:%M:%S")


def get_featureset_scheduling_state_as_str(
    registry_client: FeatureRegistryClient,
    featureset_name: str,
) -> str:
    try:
        resp: GetFeaturesetSchedulingStateResponse = (
            registry_client.get_feature_set_scheduling_state(
                featureset_name=featureset_name
            )
        )
        if resp.state == FeaturesetSchedulingState.SCHEDULING_STATE_PAUSED:
            return "Paused"
        elif resp.state == FeaturesetSchedulingState.SCHEDULING_STATE_ENABLED:
            return "Enabled"
        else:
            return "Unknown"
    except Exception:
        return "Unknown"


def _list_feature_sets_verbose(
    resp: ListFeatureSetsResponse, registry_client: FeatureRegistryClient
) -> Tuple[List[str], List[str]]:
    columns = [
        "Name",
        "Type",
        "Status",
        "Scheduling state",
        "Owner",
        "Description",
        "Creation date",
        "Last updated",
        "Featureset id",
    ]
    data = []
    for feature_set in resp.feature_families:
        definition = feature_set.feature_set_definition
        spec = definition.feature_set_spec
        metadata = feature_set.metadata
        if not (spec.feature_set_type.WhichOneof("set_type")):
            feature_set_type = ""
        else:
            feature_set_type = (
                spec.feature_set_type.WhichOneof("set_type")
                .replace("_", " ")
                .capitalize()
            )

        sched_state: str = "Enabled"
        if spec.feature_set_type.WhichOneof("set_type") in {
            "batch_feature_set",
            "batch_feature_set_v1",
        }:
            sched_state = get_featureset_scheduling_state_as_str(
                registry_client, spec.name
            )

        data.append(
            [
                spec.name,
                feature_set_type,
                FeatureStatus.Name(definition.status),
                sched_state,
                spec.metadata.owner,
                spec.metadata.description,
                _datetime_to_str(_pts_to_datetime(metadata.created_at)),
                _datetime_to_str(_pts_to_datetime(metadata.last_modified_at)),
                definition.feature_set_id,
            ]
        )

    return (data, columns)


def _list_feature_sets(
    resp: ListFeatureSetsResponse, registry_client: FeatureRegistryClient
) -> Tuple[List[str], List[str]]:
    desired_columns = [
        "Name",
        "Type",
        "Status",
        "Scheduling state",
        "Last updated",
    ]
    verbose_data, verbose_columns = _list_feature_sets_verbose(resp, registry_client)
    col_indexes = [verbose_columns.index(c) for c in desired_columns]

    def project_lst(lst: List[str]) -> List[str]:
        return [lst[idx] for idx in col_indexes]

    data = [project_lst(d) for d in verbose_data]

    return (data, desired_columns)


@click.command("list", cls=QwakCommand, help="List registered feature sets")
@click.option(
    "--verbose",
    "-v",
    default=False,
    is_flag=True,
    metavar="FLAG",
    help="Verbose output",
)
def list_feature_sets(verbose: bool, **kwargs):
    registry_client = FeatureRegistryClient()
    response = registry_client.list_feature_sets()

    if verbose:
        data, columns = _list_feature_sets_verbose(response, registry_client)
    else:
        data, columns = _list_feature_sets(response, registry_client)

    print(tabulate(data, headers=columns))
