from _qwak_proto.qwak.projects.projects_pb2 import (
    CreateProjectResponse,
    GetProjectResponse,
)
from qwak.clients.model_management import ModelsManagementClient
from qwak.clients.project.client import ProjectsManagementClient
from qwak.exceptions import QwakException

from qwak_sdk.exceptions import QwakCommandException


def execute_model_create(model_name, model_description, project, project_id):
    if not (project or project_id):
        raise QwakCommandException("You nust supply either project or project_id")

    if project_id:
        resolved_project_id = project_id
    else:
        try:
            project: GetProjectResponse = ProjectsManagementClient().get_project(
                project_id="", project_name=project
            )
            resolved_project_id = project.project.spec.project_id
        except QwakException:
            print(f"Project with name {project} doesn't exist. Creating it.")
            project_creation: CreateProjectResponse = (
                ProjectsManagementClient().create_project(
                    project_name=project, project_description=""
                )
            )
            resolved_project_id = project_creation.project.project_id

    return ModelsManagementClient().create_model(
        resolved_project_id, model_name, model_description
    )
