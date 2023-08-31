from typing import Any, Tuple, Union

from qwak.clients.build_orchestrator import BuildOrchestratorClient
from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from qwak.inner.tool.run_config import QwakConfigBase, YamlConfigMixin

from qwak_sdk.commands.models.build._logic.config.config_v1 import ConfigV1


def config_handler(
    config: Union[QwakConfigBase, YamlConfigMixin, Any],
    from_file: str,
    out_conf: bool,
    sections: Tuple[str, ...] = (),
    **kwargs,
) -> Any:
    conf: Union[QwakConfigBase, YamlConfigMixin] = config.from_yaml(from_file)
    conf.merge_cli_argument(sections=sections, **kwargs)

    if isinstance(conf, ConfigV1):
        conf.fetch_base_docker_image_name(
            BuildOrchestratorClient(), InstanceTemplateManagementClient()
        )

    if out_conf:
        print(conf.to_yaml())

    return conf
