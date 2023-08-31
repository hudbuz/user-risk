from pathlib import Path
from typing import Optional, Union

import yaml

from qwak_sdk.commands.audience._logic.config.config_base import ConfigBase
from qwak_sdk.commands.audience._logic.config.v1.config_v1 import ConfigV1
from qwak_sdk.inner.tools.dataclasses_utils import create_dataclass_instance

VERSION_CONFIG_MAPPING = {"v1": ConfigV1}


def parse_audience_from_yaml(
    file_path: Optional[Union[Path, str]] = None
) -> ConfigBase:
    if file_path:
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            audience_dict = yaml.safe_load(file_path_obj.open("r"))
            audience_config: ConfigBase = VERSION_CONFIG_MAPPING.get(
                audience_dict.get("api_version")
            )

            return create_dataclass_instance(audience_config, audience_dict)
        else:
            raise FileNotFoundError(
                f"Audience file {file_path_obj} definition isn't found"
            )
    else:
        return ConfigV1()
