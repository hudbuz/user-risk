from enum import Enum

FETCHING_MODEL_CODE_PHASE_DESCRIPTION = "Fetching Model Code"
REGISTER_QWAK_BUILD_PHASE_DESCRIPTION = "Registering Qwak Build"
DEPLOY_PHASE_DESCRIPTION = "Deploying"


class BuildPhase(Enum):
    FETCHING_MODEL_CODE = 1
    REGISTERING_QWAK_BUILD = 2
    DEPLOY_PHASE = 3


class PhaseDetails:
    def __init__(self, build_phase: BuildPhase):
        self.phase = build_phase

    def get_id(self):
        return str(self.phase.name)

    def get_description(self):
        if self.phase == BuildPhase.FETCHING_MODEL_CODE:
            return FETCHING_MODEL_CODE_PHASE_DESCRIPTION
        elif self.phase == BuildPhase.REGISTERING_QWAK_BUILD:
            return REGISTER_QWAK_BUILD_PHASE_DESCRIPTION
        elif self.phase == BuildPhase.DEPLOY_PHASE:
            return DEPLOY_PHASE_DESCRIPTION
        else:
            raise ValueError("Invalid PhaseId")
