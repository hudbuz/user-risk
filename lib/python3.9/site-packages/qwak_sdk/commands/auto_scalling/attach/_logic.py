from typing import Dict

import grpc
from qwak.clients.autoscaling import AutoScalingClient

from qwak_sdk.commands.auto_scalling._logic.config import Config
from qwak_sdk.commands.auto_scalling._logic.config.parser import (
    parse_autoscaling_from_yaml,
)
from qwak_sdk.exceptions import QwakCommandException


def attach_autoscaling(model_id: str, variation_name: str, file: str) -> Dict[str, str]:
    try:
        config = parse_autoscaling_from_yaml(file)
        config = merge_kw_yaml(
            config=config, model_id=model_id, variation_name=variation_name
        )
        response = AutoScalingClient().attach_autoscaling(
            model_id=config.spec.model_id,
            variation_name=config.spec.variation_name,
            auto_scaling_config=config.spec.auto_scaling.to_autoscaling_api(),
        )
        return {
            "autoscaling_id": response.auto_scaling_id,
            "model_id": config.spec.model_id,
            "variation_name": config.spec.variation_name,
        }
    except grpc.RpcError as e:
        raise QwakCommandException(e.args[0].details)


def merge_kw_yaml(
    config: Config,
    model_id: str = "",
    variation_name: str = "",
) -> Config:
    if model_id:
        config.spec.model_id = model_id
    if variation_name:
        config.spec.variation_name = variation_name

    return config
