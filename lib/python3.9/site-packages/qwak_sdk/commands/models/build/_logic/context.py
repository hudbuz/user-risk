from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from qwak.clients.build_orchestrator import BuildOrchestratorClient
from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from qwak.clients.model_management import ModelsManagementClient
from qwak.inner.di_configuration import UserAccountConfiguration
from qwak.inner.di_configuration.account import UserAccount


class DependencyManagerType(Enum):
    UNKNOWN = 0
    PIP = 1
    POETRY = 2
    CONDA = 3


@dataclass
class Context:
    # Clients
    client_builds_orchestrator: BuildOrchestratorClient = field(
        default_factory=BuildOrchestratorClient
    )
    client_models_management: ModelsManagementClient = field(
        default_factory=ModelsManagementClient
    )
    client_instance_template: InstanceTemplateManagementClient = field(
        default_factory=InstanceTemplateManagementClient
    )

    # General
    user_account: UserAccount = field(
        default_factory=UserAccountConfiguration().get_user_config
    )

    # Pre fetch validation
    build_id: str = field(default="")
    model_id: str = field(default="")
    project_uuid: str = field(default="")
    host_temp_local_build_dir: Path = field(default=None)
    model_uri: Path = field(default=None)
    git_credentials: str = field(default="")

    # Fetch model
    git_commit_id: Optional[str] = field(default=None)

    # Post fetch validation
    dependency_manager_type: DependencyManagerType = field(
        default=DependencyManagerType.UNKNOWN
    )
    model_relative_dependency_file: Path = field(default=None)
    model_relative_dependency_lock_file: Path = field(default=None)

    # Upload model
    model_code_remote_url: Optional[str] = field(default=None)

    # Image
    base_image: Optional[str] = field(default=None)

    # Custom runtime wheel
    custom_runtime_wheel: Optional[Path] = field(default=None)
    custom_core_wheel: Optional[Path] = field(default=None)
