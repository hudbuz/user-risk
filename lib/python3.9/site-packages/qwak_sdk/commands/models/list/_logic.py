from qwak.clients.project.client import ProjectsManagementClient


def execute_models_list(project_id):
    return ProjectsManagementClient().get_project(project_id)
