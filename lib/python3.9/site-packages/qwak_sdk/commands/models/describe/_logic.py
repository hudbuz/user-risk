import json
from datetime import datetime

from _qwak_proto.qwak.builds.builds_pb2 import BuildStatus, ValueType
from _qwak_proto.qwak.deployment.deployment_pb2 import ModelDeploymentStatus
from google.protobuf.json_format import MessageToDict
from qwak.clients.build_management import BuildsManagementClient
from qwak.clients.deployment.client import DeploymentManagementClient
from qwak.clients.model_management import ModelsManagementClient
from tabulate import tabulate


def execute_model_describe(model_id, interface, show_list_builds, format):
    model = _model_data(model_id)
    list_builds = _builds_data(show_list_builds, model)
    interface_build = _interface_data(interface, model)
    if format == "text":
        print_text_data(model, list_builds, interface_build)
    elif format == "json":
        print_json_data(model, list_builds, interface_build)


def _model_data(model_id):
    models_management = ModelsManagementClient()
    return models_management.get_model(model_id)


def _builds_data(list_builds, model):
    if list_builds:
        builds_management = BuildsManagementClient()
        return builds_management.list_builds(model.uuid)
    return None


def _interface_data(interface, model):
    if interface:
        deployment_client = DeploymentManagementClient()
        deployment_details = deployment_client.get_deployment_details(
            model_id=model.model_id, model_uuid=model.uuid
        )
        if deployment_details.current_deployment_details.build_id:
            builds_management = BuildsManagementClient()
            build_response = builds_management.get_build(
                deployment_details.current_deployment_details.build_id
            )
            if build_response.build.HasField("model_schema"):
                return build_response
    return None


def print_json_data(model, list_builds, interface_build):
    output = MessageToDict(model)
    if list_builds:
        output["builds"] = MessageToDict(list_builds)
    if interface_build:
        output["interface"] = MessageToDict(interface_build)["build"]["modelSchema"]
    print(json.dumps(output, indent=4, sort_keys=True))


def print_text_data(model, list_builds, interface_build):
    print(
        f"Model id: {model.model_id}\nDisplay name: {model.display_name}\nDescription: {model.model_description}\n"
        + f'Creation Date: {datetime.fromtimestamp(model.created_at.seconds + model.created_at.nanos / 1e9).strftime("%A, %B %d, %Y %I:%M:%S")}\n'
        + f'Last update: {datetime.fromtimestamp(model.created_at.seconds + model.last_modified_at.nanos / 1e9).strftime("%A, %B %d, %Y %I:%M:%S")}'
    )
    if list_builds:
        columns = [
            "Build id",
            "Commit id",
            "Last modified date",
            "Build Status",
            "Deployment build status",
        ]
        data = []
        for build in list_builds.builds:
            deployment_status = (
                ModelDeploymentStatus.Name(number=build.deployment_build_status)
                if build.deployment_build_status != 0
                else ""
            )
            data.append(
                [
                    build.build_spec.build_id,
                    build.build_spec.commit_id,
                    datetime.fromtimestamp(
                        build.last_modified_at.seconds
                        + build.last_modified_at.nanos / 1e9
                    ).strftime("%A, %B %d, %Y %I:%M:%S"),
                    BuildStatus.Name(number=build.build_status),
                    deployment_status,
                ]
            )
        print("\n" + tabulate(data, headers=columns))
    if interface_build:
        model_schema = interface_build.build.model_schema
        columns = [
            "Parameter name",
            "Parameter type",
            "Parameter source",
            "Parameter category",
        ]
        data = []
        for entity in model_schema.entities:
            data.append(
                [
                    entity.name,
                    ValueType.Types.Name(entity.type.type),
                    None,
                    "Input",
                ]
            )
        for feature in model_schema.features:
            if feature.HasField("explicit_feature"):
                data.append(
                    [
                        feature.explicit_feature.name,
                        ValueType.Types.Name(feature.explicit_feature.type.type),
                        None,
                        "Input",
                    ]
                )
            elif feature.HasField("batch_feature"):
                data.append(
                    [
                        feature.batch_feature.name,
                        None,
                        feature.batch_feature.entity.name,
                        "Batch Feature",
                    ]
                )
            elif feature.HasField("on_the_fly_feature"):
                data.append(
                    [
                        feature.on_the_fly_feature.name,
                        None,
                        str(
                            [
                                source.explicit_feature.name
                                for source in feature.on_the_fly_feature.source_features
                            ]
                        ),
                        "On-The-Fly Feature",
                    ]
                )
        for prediction in model_schema.predictions:
            data.append(
                [
                    prediction.name,
                    ValueType.Types.Name(prediction.type.type),
                    None,
                    "Output",
                ]
            )
        print("\n" + tabulate(data, headers=columns))
