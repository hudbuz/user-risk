import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from _qwak_proto.qwak.feature_store.sources.batch_pb2 import (
    BatchSource as ProtoBatchSource,
)
from _qwak_proto.qwak.features_operator.v3.features_operator_pb2 import (
    ValidationSuccessResponse,
)
from qwak.clients.feature_store.operator_client import FeaturesOperatorClient
from qwak.exceptions import QwakException

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pass


@dataclass
class BaseBatchSource(ABC):
    name: str
    date_created_column: str
    description: str

    @abstractmethod
    def _to_proto(self) -> ProtoBatchSource:
        pass

    @classmethod
    @abstractmethod
    def _from_proto(cls, proto):
        pass

    def get_sample(self, number_of_rows: int = 10) -> "pd.DataFrame":
        """
        Tries to get a sample of length `number_rows` from the data source.
        Args:
            number_of_rows: number of rows to get from data source
        Returns:
            A tuple containing the resulting dataframe and a tuple of the columns names and types.
            (the types are pyspark dataframe types)
        """

        try:
            import pandas as pd
        except ImportError:
            raise QwakException("Missing Pandas dependency required for getting sample")

        if number_of_rows > 1000:
            raise ValueError(f"`number_rows` must be under 1000, got: {number_of_rows}")

        operator_client = FeaturesOperatorClient()
        ds_spec = self._to_proto()

        result = operator_client.validate_data_source_blocking(
            data_source_spec=ds_spec, num_samples=number_of_rows
        )

        response = getattr(result, result.WhichOneof("type"))
        if isinstance(response, ValidationSuccessResponse):
            return pd.read_json(
                path_or_buf=ast.literal_eval(response.sample),
                dtype=response.spark_column_description,
            )
        else:
            raise QwakException(f"Sampling failed:\n{response}")
