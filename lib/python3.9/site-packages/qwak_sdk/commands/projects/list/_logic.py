from qwak.clients.project.client import ProjectsManagementClient


def execute():
    projects_management = ProjectsManagementClient()
    return projects_management.list_projects()
