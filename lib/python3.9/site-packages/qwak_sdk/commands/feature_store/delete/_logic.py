from qwak_sdk.tools.colors import Color

FEATURE_SET = "feature_set"
ENTITY = "entity"
DATA_SOURCE = "data_source"
All = "all"


def _inner_delete(name, object, registry_client):
    if object == ENTITY:
        feature_store_entity = registry_client.get_entity_by_name(name)
        _delete_object(
            lambda e: registry_client.delete_entity(
                e.entity.entity_definition.entity_id
            ),
            feature_store_entity,
            "entity",
            name,
        )
    if object == FEATURE_SET:
        feature_set = registry_client.get_feature_set_by_name(name)
        if feature_set:
            registry_client.delete_feature_set(
                feature_set.feature_set.feature_set_definition.feature_set_id
            )
            print(
                f"{Color.GREEN}Feature Set '{name}' deletion request is being handled by the service"
            )
        else:
            print(
                f"{Color.RED}Could not find specified feature set named '{name}' to delete{Color.END}"
            )
    if object == DATA_SOURCE:
        feature_store_ds = registry_client.get_data_source_by_name(name)
        _delete_object(
            lambda e: registry_client.delete_data_source(
                e.data_source.data_source_definition.data_source_id
            ),
            feature_store_ds,
            "data source",
            name,
        )


def _delete_object(delete_func, _object, _type, name):
    if _object:
        delete_func(_object)
        print(f"{Color.GREEN}Successfully deleted the {_type} named '{name}'")
    else:
        print(
            f"{Color.RED}Could not find specified {_type} named '{name}' to delete{Color.END}"
        )
