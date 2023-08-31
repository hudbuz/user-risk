from _qwak_proto.qwak.instance_template.instance_template_pb2 import (
    InstanceTemplateSpec,
    InstanceType,
)
from _qwak_proto.qwak.instance_template.instance_template_service_pb2 import (
    GetInstanceTemplateRequest,
    GetInstanceTemplateResponse,
    ListInstanceTemplatesRequest,
    ListInstanceTemplatesResponse,
)
from _qwak_proto.qwak.instance_template.instance_template_service_pb2_grpc import (
    InstanceTemplateManagementServiceServicer,
)
from _qwak_proto.qwak.user_application.common.v0.resources_pb2 import (
    CpuResources,
    GpuResources,
    GpuType,
    MemoryUnit,
    NodeOptimizationType,
    PodComputeResources,
)
from qwak.exceptions import QwakException
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


def _create_instance_template_cpu(
    instance_id: str, name: str, order: int, cpu: float, memory_amount: int
) -> InstanceTemplateSpec:
    return InstanceTemplateSpec(
        id=instance_id,
        display_name=name,
        order=order,
        enabled=True,
        pod_compute_resources=PodComputeResources(
            optimization_type=NodeOptimizationType.NODE_OPTIMIZATION_NONE,
            cpu_resources=CpuResources(
                cpu=cpu,
                memory_amount=memory_amount,
                memory_units=MemoryUnit.GIB,
            ),
        ),
        instance_type=InstanceType.INSTANCE_TYPE_CPU,
    )


def _create_instance_template_gpu(
    instance_id: str,
    name: str,
    order: int,
    gpu: int,
    gpu_type: GpuType,
    cpu: float,
    memory_amount: int,
) -> InstanceTemplateSpec:
    return InstanceTemplateSpec(
        id=instance_id,
        display_name=name,
        order=order,
        enabled=True,
        pod_compute_resources=PodComputeResources(
            optimization_type=NodeOptimizationType.NODE_OPTIMIZATION_NONE,
            gpu_resources=GpuResources(
                gpu_amount=gpu,
                gpu_type=gpu_type,
            ),
            cpu_resources=CpuResources(
                cpu=cpu,
                memory_amount=memory_amount,
                memory_units=MemoryUnit.GIB,
            ),
        ),
        instance_type=InstanceType.INSTANCE_TYPE_GPU,
    )


class InstanceTemplateManagementServiceMock(InstanceTemplateManagementServiceServicer):
    INSTANCES = {
        "small": _create_instance_template_cpu("small", "Small", 1, 1, 8),
        "medium": _create_instance_template_cpu("medium", "Medium", 2, 2, 16),
        "large": _create_instance_template_cpu("large", "Large", 3, 4, 32),
        "xlarge": _create_instance_template_cpu("xlarge", "XLarge", 4, 8, 64),
        "g5_xlarge": _create_instance_template_gpu(
            "g5_xlarge", "G5 XLarge", 1, 1, GpuType.NVIDIA_A10G, 1, 8
        ),
        "g5_2xlarge": _create_instance_template_gpu(
            "g5_2xlarge", "G5 2XLarge", 2, 2, GpuType.NVIDIA_A10G, 2, 16
        ),
        "g5_4xlarge": _create_instance_template_gpu(
            "g5_4xlarge", "G5 4XLarge", 2, 4, GpuType.NVIDIA_A10G, 4, 32
        ),
    }

    def ListInstanceTemplates(
        self, request: ListInstanceTemplatesRequest, context
    ) -> ListInstanceTemplatesResponse:
        try:
            return ListInstanceTemplatesResponse(
                instance_template_list=list(self.INSTANCES.values())
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetInstanceTemplate(
        self, request: GetInstanceTemplateRequest, context
    ) -> GetInstanceTemplateResponse:
        try:
            if request.id in self.INSTANCES:
                return GetInstanceTemplateResponse(
                    instance_template=self.INSTANCES[request.id]
                )
            else:
                raise QwakException("Instance template not found")
        except Exception as e:
            raise_internal_grpc_error(context, e)
