from qwak.clients.project.client import ProjectsManagementClient


def execute(project_id):
    projects_management = ProjectsManagementClient()
    return projects_management.delete_project(project_id)
