import uuid
from datetime import datetime

import grpc
from _qwak_proto.qwak.vectors.v1.collection.collection_pb2 import (
    COLLECTION_STATUS_CREATE_REQUESTED,
    QwakMetadata,
    VectorCollection,
    VectorCollectionDefinition,
)
from _qwak_proto.qwak.vectors.v1.collection.collection_service_pb2 import (
    CreateCollectionResponse,
    DeleteCollectionByIdResponse,
    DeleteCollectionByNameResponse,
    GetCollectionByIdResponse,
    GetCollectionByNameResponse,
    ListCollectionsResponse,
)
from _qwak_proto.qwak.vectors.v1.collection.collection_service_pb2_grpc import (
    VectorCollectionServiceServicer,
)
from google.protobuf.timestamp_pb2 import Timestamp


class VectorCollectionManagementServiceMock(VectorCollectionServiceServicer):
    def __init__(self):
        self._collections_spec_by_ids = {}
        self._collections_spec_by_name = {}

    def reset_collections(self):
        self._collections_spec_by_ids.clear()
        self._collections_spec_by_name.clear()

    def CreateCollection(self, request, context):
        timestamp = Timestamp()
        timestamp.FromDatetime(datetime.now())

        collection_id = str(uuid.uuid4())
        vector_collection = VectorCollection(
            metadata=QwakMetadata(
                created_at=timestamp,
                created_by="it@qwak.com",
                last_modified_at=timestamp,
                last_modified_by="it@qwak.com",
            ),
            definition=VectorCollectionDefinition(
                id=collection_id,
                collection_spec=request.collection_spec,
            ),
            status=COLLECTION_STATUS_CREATE_REQUESTED,
        )

        self._collections_spec_by_ids[collection_id] = vector_collection
        self._collections_spec_by_name[request.collection_spec.name] = vector_collection
        return CreateCollectionResponse(vector_collection=vector_collection)

    def GetCollectionById(self, request, context):
        if request.id in self._collections_spec_by_ids:
            return GetCollectionByIdResponse(
                vector_collection=self._collections_spec_by_ids[request.id]
            )

        context.set_details(f"Collection ID {request.id} doesn't exist'")
        context.set_code(grpc.StatusCode.NOT_FOUND)

    def GetCollectionByName(self, request, context):
        if request.name in self._collections_spec_by_name:
            return GetCollectionByNameResponse(
                vector_collection=self._collections_spec_by_name[request.name]
            )

        context.set_details(f"Collection name {request.name} doesn't exist'")
        context.set_code(grpc.StatusCode.NOT_FOUND)

    def DeleteCollectionById(self, request, context):
        if request.id in self._collections_spec_by_ids:
            self._collections_spec_by_ids.pop(request.id)
            return DeleteCollectionByIdResponse()

        context.set_details(f"Collection ID {request.id} doesn't exist'")
        context.set_code(grpc.StatusCode.NOT_FOUND)

    def DeleteCollectionByName(self, request, context):
        if request.name in self._collections_spec_by_name:
            self._collections_spec_by_name.pop(request.name)
            return DeleteCollectionByNameResponse()

        context.set_details(f"Collection name {request.name} doesn't exist'")
        context.set_code(grpc.StatusCode.NOT_FOUND)

    def ListCollections(self, request, context):
        return ListCollectionsResponse(
            vector_collections=[
                collection for collection in self._collections_spec_by_ids.values()
            ]
        )
