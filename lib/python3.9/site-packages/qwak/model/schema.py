from dataclasses import dataclass
from typing import List

from _qwak_proto.qwak.builds.builds_pb2 import ModelSchema as ProtoModelSchema
from qwak.model._entity_extraction import enrich_schema
from qwak.model.schema_entities import (
    BaseFeature,
    Entity,
    ExplicitFeature,
    FeatureStoreInput,
    InferenceOutput,
)

__all__ = ["ModelSchema", "ExplicitFeature", "InferenceOutput", "FeatureStoreInput"]


@dataclass
class ModelSchema:
    """
    The ModelSchema class provides a structured representation of a Machine Learning (ML) model's schema.
     It captures the details such as input features, output predictions, and entities, providing a standard way to
    communicate these details between different components of an ML system.
    """

    def __init__(
        self,
        entities: List[Entity] = None,
        inputs: List[BaseFeature] = None,
        outputs: List[InferenceOutput] = None,
    ):
        """
        Initializes the ModelSchema object with entities, inputs, and outputs.

        Args:
            entities (List[Entity], optional): A list of Entity objects. Defaults to None.
            inputs (List[BaseFeature], optional): A list of BaseFeature objects. Defaults to None.
            outputs (List[InferenceOutput], optional): A list of InferenceOutput objects. Defaults to None.

        Raises:
            TypeError: If entities, inputs, or outputs are not provided in the correct format.
        """
        raw_entities = entities if entities else []
        raw_inputs = inputs if inputs else []
        self._inputs, self._entities = enrich_schema(raw_inputs, raw_entities)
        self._outputs = outputs if outputs else []

    def to_proto(self):
        return ProtoModelSchema(
            entities=[entity.to_proto() for entity in self._entities],
            features=[feature.to_proto() for feature in self._inputs],
            inference_output=[inference.to_proto() for inference in self._outputs],
        )
