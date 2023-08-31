from dataclasses import dataclass, field
from typing import List

from qwak.qwak_client.builds.build import Build
from qwak.qwak_client.deployments.deployment import Deployment, EnvironmentAudienceRoute
from qwak.qwak_client.models.model import Model


@dataclass
class ModelMetadata:
    model: Model = field(default=None)
    deployments: List[Deployment] = field(default_factory=list)
    audience_routes: List[EnvironmentAudienceRoute] = field(default_factory=list)
    deployed_builds: List[Build] = field(default_factory=list)
