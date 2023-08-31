from .build_polling_status import BuildPollingStatusStep
from .deploy_build import DeployBuildStep


def get_deploy_steps():
    return [BuildPollingStatusStep(), DeployBuildStep()]
