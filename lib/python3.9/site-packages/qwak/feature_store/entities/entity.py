from _qwak_proto.qwak.feature_store.entities.entity_pb2 import EntitySpec
from qwak.clients.feature_store import FeatureRegistryClient


class Entity:
    """
    An entity is an object which features sets can be associated with.

    Features created via the feature store are tied to a specific business entity.
    For example registered users, transactions, or merchants.

    Example of an Entity definition:

    .. code-block:: python

        from qwak.feature_store import Entity

        customer = Entity(
            name='organization_id',
            description='An organization which is a customer of our platform',
        )
    """

    def __init__(self, name: str, description: str):
        """
        Create a new Entity

        :param name: The name of the entity, must be unique
        :param description: Short human-readable description of the entity
        """
        self.name = name
        self.description = description

    def _to_proto(self):
        return EntitySpec(
            name=self.name,
            keys=[self.name],
            description=self.description,
            value_type=1,  # string
        )

    @classmethod
    def _from_proto(cls, proto):
        entity_spec = proto.entity_spec
        return cls(name=entity_spec.name, description=entity_spec.description)

    def register(self):
        """
        Explicitly register this entity to Qwak's Feature Store
        """

        registry = FeatureRegistryClient()
        existing_entity = registry.get_entity_by_name(self.name)
        if existing_entity:
            registry.update_entity(
                existing_entity.entity.entity_definition.entity_id,
                self._to_proto(),
            )
        else:
            registry.create_entity(self._to_proto())
