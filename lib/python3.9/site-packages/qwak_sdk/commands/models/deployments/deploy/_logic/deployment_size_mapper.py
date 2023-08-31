from __future__ import annotations

from _qwak_proto.qwak.deployment.deployment_pb2 import DeploymentSize, MemoryUnit
from _qwak_proto.qwak.user_application.common.v0.resources_pb2 import (
    ClientPodComputeResources,
    CpuResources,
    GpuResources,
    PodComputeResourceTemplateSpec,
)
from qwak.clients.instance_template.client import InstanceTemplateManagementClient

from qwak_sdk.commands._logic.instance_template import verify_template_id
from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)


def deployment_size_from_deploy_config(
    deploy_config: DeployConfig,
    instance_template_client: InstanceTemplateManagementClient,
) -> DeploymentSize:
    if deploy_config.resources.instance_size:
        deploy_config.resources.instance_size = (
            deploy_config.resources.instance_size.lower()
        )
        verify_template_id(
            deploy_config.resources.instance_size, instance_template_client
        )

        return DeploymentSize(
            number_of_pods=deploy_config.resources.pods,
            client_pod_compute_resources=ClientPodComputeResources(
                template_spec=PodComputeResourceTemplateSpec(
                    template_id=deploy_config.resources.instance_size
                )
            ),
        )
    elif deploy_config.resources.gpu_type:
        return DeploymentSize(
            number_of_pods=deploy_config.resources.pods,
            client_pod_compute_resources=ClientPodComputeResources(
                gpu_resources=GpuResources(
                    gpu_type=deploy_config.resources.gpu_type,
                    gpu_amount=deploy_config.resources.gpu_amount,
                )
            ),
        )
    else:
        return DeploymentSize(
            number_of_pods=deploy_config.resources.pods,
            client_pod_compute_resources=ClientPodComputeResources(
                cpu_resources=CpuResources(
                    cpu=deploy_config.resources.cpus,
                    memory_amount=deploy_config.resources.memory,
                    memory_units=MemoryUnit.MIB,
                )
            ),
        )
