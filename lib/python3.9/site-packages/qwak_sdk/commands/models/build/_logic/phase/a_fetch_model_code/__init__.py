from typing import List

from ...interface.step_inteface import Step
from .fetch_model_step.fetch_model_step import FetchModelStep
from .post_fetch_validation_step import PostFetchValidationStep
from .pre_fetch_validation_step import PreFetchValidationStep


def get_fetch_model_code_steps() -> List[Step]:
    return [
        PreFetchValidationStep(),
        FetchModelStep(),
        PostFetchValidationStep(),
    ]
