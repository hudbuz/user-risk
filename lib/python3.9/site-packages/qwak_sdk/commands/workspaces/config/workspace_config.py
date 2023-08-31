from dataclasses import dataclass, field
from typing import List

from qwak_sdk.common.run_config import (
    ConfigCliMap,
    QwakConfigBase,
    YamlConfigMixin,
    validate_string,
)

CONFIG_MAPPING: List[ConfigCliMap] = [
    ConfigCliMap("workspace_id", "workspace.workspace_id", validate_string, False),
    ConfigCliMap("name", "workspace.name", validate_string, False),
    ConfigCliMap("instance", "workspace.instance", validate_string, False),
    ConfigCliMap("image", "workspace.image", validate_string, False),
]


@dataclass
class WorkspaceConfig(YamlConfigMixin, QwakConfigBase):
    def _post_merge_cli(self):
        pass

    @property
    def _config_mapping(self) -> List[ConfigCliMap]:
        return CONFIG_MAPPING

    @dataclass
    class Workspace:
        workspace_id: str = field(default_factory=str)
        name: str = field(default_factory=str)
        instance: str = field(default_factory=str)
        image: str = field(default_factory=str)

    workspace: Workspace = field(default_factory=Workspace)
