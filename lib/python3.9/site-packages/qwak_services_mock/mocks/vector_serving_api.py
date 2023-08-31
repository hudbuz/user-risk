import grpc
from _qwak_proto.qwak.vectors.v1.vector_pb2 import Property, SearchResult
from _qwak_proto.qwak.vectors.v1.vector_service_pb2 import (
    DeleteVectorsResponse,
    SearchSimilarVectorsResponse,
    UpsertVectorsResponse,
)
from _qwak_proto.qwak.vectors.v1.vector_service_pb2_grpc import VectorServiceServicer
from numpy import dot
from numpy.linalg import norm


class VectorServingServiceMock(VectorServiceServicer):
    def __init__(self):
        self._vector_collections = {}

    def reset_vector_store(self):
        self._vector_collections.clear()

    def get_num_of_vectors(self, collection_name):
        if collection_name not in self._vector_collections:
            return 0

        return len(self._vector_collections[collection_name].values())

    def SearchSimilarVectors(self, request, context):
        if request.collection_name not in self._vector_collections:
            context.set_details(f"Collection named {request.collection} doesn't exist'")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return

        reference_vector = list(request.reference_vector.element)
        collection_store = self._vector_collections.get(request.collection_name, dict())

        # a naive impl of a "vector similarity" search - compute pairwise cosine distance on
        # the entire set and return top results
        result_set = sorted(
            collection_store.values(),
            key=lambda b: _cos_sim(reference_vector, b["vector"]),
        )[-request.max_results :]

        return SearchSimilarVectorsResponse(
            search_results=[
                SearchResult(
                    id=result["id"] if request.include_id else None,
                    properties=[
                        p
                        for p in result_set[0]["properties"]
                        if p.name in request.properties
                    ],
                    vector=result["vector"] if request.include_vector else None,
                    distance=_cos_sim(reference_vector, result["vector"])
                    if request.include_distance
                    else None,
                )
                for result in result_set
            ]
        )

    def UpsertVectors(self, request, context):
        collection_store = self._vector_collections.get(request.collection_name, dict())
        for stored_vector in request.vector:
            collection_store[stored_vector.id] = {
                "id": stored_vector.id,
                "vector": list(stored_vector.vector.element),
                "properties": stored_vector.property,
            }

        self._vector_collections[request.collection_name] = collection_store
        return UpsertVectorsResponse()

    def DeleteVectors(self, request, context):
        if request.collection_name not in self._vector_collections:
            context.set_details(f"Collection named {request.collection} doesn't exist'")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return

        collection_store = self._vector_collections[request.collection_name]
        ids_in_collection = [
            vector_id
            for vector_id in request.vector_id
            if vector_id in collection_store
        ]
        for id in ids_in_collection:
            collection_store.pop(id)

        self._vector_collections[request.collection_name] = collection_store
        return DeleteVectorsResponse(num_vectors_deleted=len(ids_in_collection))


def _cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))
