import grpc
from _qwak_proto.qwak.builds.build_url_pb2 import (
    BuildVersioningTagsDefinition,
    BuildVersioningTagsProperties,
    BuildVersioningTagsType,
)
from _qwak_proto.qwak.builds.build_pb2 import (
    BaseDockerImageType,
)
from _qwak_proto.qwak.builds.builds_orchestrator_service_pb2 import (
    BuildModelResponse,
    CancelBuildModelResponse,
    CreateDataTableResponse,
    GetBuildVersioningDownloadURLResponse,
    GetBuildVersioningUploadURLResponse,
    ListBuildVersioningTagsResponse,
    GetBaseDockerImageNameResponse,
)
from _qwak_proto.qwak.builds.builds_orchestrator_service_pb2_grpc import (
    BuildsOrchestratorServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class BuildOrchestratorServiceApiMock(BuildsOrchestratorServiceServicer):
    UPLOAD_URL_FORMAT = "/{build_id}/upload_urls/file.zip"
    DOWNLOAD_URL_FORMAT = "/{build_id}/download_urls/file.zip"

    def __init__(self):
        super(BuildOrchestratorServiceApiMock, self).__init__()

    def BuildModel(self, request, context):
        """Build a serving model image"""
        try:
            return BuildModelResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def CancelBuildModel(self, request, context):
        """Cancel an ongoing build"""
        try:
            return CancelBuildModelResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def CreateUploadURL(self, request, context):
        """Request the service for a path ready for upload in purpose of uploading local model code available to the service
        and the download link corresponding
        """
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details("Deprecated")
        raise DeprecationWarning("This api is deprecated and should not be used")

    def GetBuildVersioningUploadURL(self, request, context):
        """Request the service for a path ready for upload in purpose of uploading local model code or data"""
        try:
            return GetBuildVersioningUploadURLResponse(
                upload_url=BuildOrchestratorServiceApiMock.UPLOAD_URL_FORMAT.format(
                    build_id=request.params.build_id
                ),
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetBuildVersioningDownloadURL(self, request, context):
        """Request the service for a path ready for download the artifact"""
        try:
            return GetBuildVersioningDownloadURLResponse(
                download_url=BuildOrchestratorServiceApiMock.DOWNLOAD_URL_FORMAT.format(
                    build_id=request.params.build_id
                ),
                file_size=1000,
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ListBuildVersioningTags(self, request, context):
        """Request the service for a list of paths ready for download the artifact"""
        try:
            return ListBuildVersioningTagsResponse(
                build_versioning_tags_properties=[
                    self.__map_versioning_tag_to_property(
                        build_id=request.build_versioning_tags.build_id,
                        versioning_tag=tag,
                    )
                    for tag in request.build_versioning_tags.tags_definition
                ]
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def CreateDataTable(self, request, context):
        """Create data table for build."""
        try:
            return CreateDataTableResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetBaseDockerImageName(self, request, context):
        try:
            if request.base_docker_image_type == BaseDockerImageType.GPU:
                return GetBaseDockerImageNameResponse(
                    base_docker_image_name="mocked_gpu_image"
                )
            elif request.base_docker_image_type == BaseDockerImageType.CPU:
                return GetBaseDockerImageNameResponse(
                    base_docker_image_name="mocked_cpu_image"
                )
            else:
                raise ValueError(
                    f"Unknown base docker image type: {request.base_docker_image_type}"
                )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    @staticmethod
    def __map_versioning_tag_to_property(
        build_id: str, versioning_tag: BuildVersioningTagsDefinition
    ) -> BuildVersioningTagsProperties:
        return BuildVersioningTagsProperties(
            tag=versioning_tag.tags[0],
            download_url=BuildOrchestratorServiceApiMock.DOWNLOAD_URL_FORMAT.format(
                build_id=build_id
            ),
            file_size=1000,
            tag_type=BuildVersioningTagsType.FILE_TAG_TYPE,
        )
