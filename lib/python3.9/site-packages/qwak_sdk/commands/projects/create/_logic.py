from qwak.clients.project.client import ProjectsManagementClient


def execute(project_name: str, project_description: str):
    projects_management = ProjectsManagementClient()
    return projects_management.create_project(project_name, project_description)
