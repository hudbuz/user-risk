from _qwak_proto.qwak.instance_template.instance_template_pb2 import InstanceType
from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from qwak.exceptions import QwakException

INVALID_TEMPLATE_ID_ERROR_FORMAT = (
    "Invalid instance: {template_id}. Valid instances are: {cpu_templates}. "
    "For GPU instances you can use {gpu_templates}."
)


def verify_template_id(
    template_id: str, instance_template_client: InstanceTemplateManagementClient
) -> None:
    all_templates = instance_template_client.list_instance_templates()
    valid_templates = [template for template in all_templates if template.enabled]
    existing_templates = {template.id: template for template in valid_templates}
    if template_id not in existing_templates.keys():
        cpu_template_ids = [
            template.id
            for template in sorted(valid_templates, key=lambda template: template.order)
            if template.instance_type == InstanceType.INSTANCE_TYPE_CPU
        ]
        gpu_template_ids = [
            template.id
            for template in sorted(valid_templates, key=lambda template: template.order)
            if template.instance_type == InstanceType.INSTANCE_TYPE_GPU
        ]
        raise QwakException(
            INVALID_TEMPLATE_ID_ERROR_FORMAT.format(
                template_id=template_id,
                cpu_templates=", ".join(cpu_template_ids),
                gpu_templates=", ".join(gpu_template_ids),
            )
        )
