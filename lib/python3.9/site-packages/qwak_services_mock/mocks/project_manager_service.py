import uuid
from typing import Dict

import grpc
from _qwak_proto.qwak.projects.projects_pb2 import (
    CreateProjectRequest,
    CreateProjectResponse,
    DeleteProjectRequest,
    DeleteProjectResponse,
    GetProjectRequest,
    GetProjectResponse,
    ListProjectsRequest,
    ListProjectsResponse,
    Project,
    ProjectSpec,
)
from _qwak_proto.qwak.projects.projects_pb2_grpc import (
    ProjectsManagementServiceServicer,
)
from google.protobuf.timestamp_pb2 import Timestamp


class ProjectManagerServiceMock(ProjectsManagementServiceServicer):
    def __init__(self):
        self.projects: Dict[str, ProjectSpec] = dict()

    def ListProjects(
        self, request: ListProjectsRequest, context
    ) -> ListProjectsResponse:
        return ListProjectsResponse(
            projects=[project for project in self.projects.values()]
        )

    def CreateProject(
        self, request: CreateProjectRequest, context
    ) -> CreateProjectResponse:
        if request.project_name in self.projects.keys():
            description = f"Project with name {request.project_name} already exists"
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details(description)
            raise Exception(description)

        now = Timestamp()
        now.GetCurrentTime()

        project_spec = ProjectSpec(
            project_id=str(uuid.uuid4()),
            project_name=request.project_name,
            project_description=request.project_description,
            project_status=ProjectSpec.Status.ACTIVE,
            created_at=now,
            last_modified_at=now,
            models_active=0,
            models_count=0,
        )

        self.projects[request.project_name] = project_spec

        return CreateProjectResponse(project=project_spec)

    def GetProject(self, request: GetProjectRequest, context) -> GetProjectResponse:
        matched_projects = list(
            filter(
                lambda project: request.project_id == project.project_id,
                self.projects.values(),
            )
        )

        if not matched_projects:
            description = f"Project with id {request.project_name} was not found"
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(description)
            raise Exception(description)

        return GetProjectResponse(project=Project(spec=matched_projects[0]))

    def DeleteProject(
        self, request: DeleteProjectRequest, context
    ) -> DeleteProjectResponse:
        matched_projects = list(
            filter(
                lambda project: request.project_id == project.project_id,
                self.projects.values(),
            )
        )

        if not matched_projects:
            description = f"Project with id {request.project_name} was not found"
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(description)
            raise Exception(description)

        del self.projects[matched_projects[0].project_name]

        return DeleteProjectResponse(info="Deleted")
