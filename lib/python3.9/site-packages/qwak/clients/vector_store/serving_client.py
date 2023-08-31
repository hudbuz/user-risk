from typing import List, Optional

import grpc
from _qwak_proto.qwak.vectors.v1.filters_pb2 import Filter
from _qwak_proto.qwak.vectors.v1.vector_pb2 import (
    DoubleVector,
    SearchResult,
    StoredVector,
)
from _qwak_proto.qwak.vectors.v1.vector_service_pb2 import (
    DeleteVectorsRequest,
    SearchSimilarVectorsRequest,
    UpsertVectorsRequest,
)
from _qwak_proto.qwak.vectors.v1.vector_service_pb2_grpc import VectorServiceStub
from qwak.clients._inner.edge_communications import get_endpoint_url
from qwak.exceptions import QwakException
from qwak.inner.tool.grpc.grpc_tools import create_grpc_channel
from typeguard import typechecked


class VectorServingClient:
    def __init__(
        self,
        edge_services_url: Optional[str] = None,
        enable_ssl: bool = True,
        environment_id: Optional[str] = None,
    ):
        edge_services_url = get_endpoint_url(edge_services_url, environment_id)
        grpc_channel = create_grpc_channel(
            url=edge_services_url,
            enable_ssl=enable_ssl,
            status_for_retry=(
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.INTERNAL,
            ),
            backoff_options={"init_backoff_ms": 250},
        )

        self._vector_serving_service = VectorServiceStub(grpc_channel)

    @typechecked
    def search(
        self,
        collection_name: str,
        vector: List[float],
        properties: List[str],
        top_k: int = 1,
        include_id: bool = True,
        include_vector: bool = False,
        include_distance: bool = False,
        filters: Optional[Filter] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors
        """
        try:
            return list(
                self._vector_serving_service.SearchSimilarVectors(
                    SearchSimilarVectorsRequest(
                        collection_name=collection_name,
                        reference_vector=DoubleVector(element=vector),
                        properties=properties,
                        max_results=top_k,
                        filter=filters,
                        include_id=include_id,
                        include_vector=include_vector,
                        include_distance=include_distance,
                    )
                ).search_results
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to query collection '{collection_name}' for vector '{str(vector)}', error is {e.details()}"
            )

    def upsert_vectors(self, collection_name: str, vectors: List[StoredVector]) -> None:
        """
        Upsert vectors to a collection
        """
        try:
            self._vector_serving_service.UpsertVectors(
                UpsertVectorsRequest(
                    collection_name=collection_name,
                    vector=vectors,
                )
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to upsert vectors with id's '{str([v.id for v in vectors])}', error is {e.details()}"
            )

    def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> int:
        """
        Delete vectors from a collection
        """
        try:
            return self._vector_serving_service.DeleteVectors(
                DeleteVectorsRequest(
                    collection_name=collection_name, vector_id=vector_ids
                )
            ).num_vectors_deleted

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to delete vectors by id '{str(vector_ids)}', error is {e.details()}"
            )
