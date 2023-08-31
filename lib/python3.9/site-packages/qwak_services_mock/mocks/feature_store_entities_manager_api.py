import grpc
from _qwak_proto.qwak.feature_store.entities.entity_pb2 import (
    Entity,
    EntityDefinition,
    EntityMetadata,
    FeatureSetBrief,
)
from _qwak_proto.qwak.feature_store.entities.entity_service_pb2 import (
    CreateEntityResponse,
    GetEntityByNameResponse,
    ListEntitiesResponse,
)
from _qwak_proto.qwak.feature_store.entities.entity_service_pb2_grpc import (
    EntityServiceServicer,
)


class EntityServiceMock(EntityServiceServicer):
    def __init__(self):
        self._entities_spec = {}

    def reset_entities(self):
        self._entities_spec.clear()

    def CreateEntity(self, request, context):
        self._entities_spec[request.entity_spec.name] = request.entity_spec
        return CreateEntityResponse()

    def GetEntityByName(self, request, context):
        if request.entity_name not in self._entities_spec:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return GetEntityByNameResponse()

        return GetEntityByNameResponse(
            entity=Entity(
                entity_definition=EntityDefinition(
                    entity_id="123",
                    entity_spec=self._entities_spec[request.entity_name],
                ),
                metadata=None,
                feature_sets=[],
            )
        )

    def ListEntities(self, request, context):
        return ListEntitiesResponse(
            entities=[
                Entity(
                    entity_definition=EntityDefinition(
                        entity_spec=entity_spec, entity_id="123"
                    ),
                    metadata=EntityMetadata(),
                    feature_sets=[FeatureSetBrief()],
                )
                for entity_spec in self._entities_spec.values()
            ]
        )
