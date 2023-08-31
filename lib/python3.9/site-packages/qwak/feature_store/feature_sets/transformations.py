from abc import ABC, abstractmethod

from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    SqlTransformation as ProtoSqlTransformation,
)
from _qwak_proto.qwak.feature_store.features.feature_set_types_pb2 import (
    Transformation as ProtoTransformation,
)
from qwak.exceptions import QwakException


class BatchTransformation(ABC):
    @abstractmethod
    def _to_proto(self) -> ProtoTransformation:
        pass

    @classmethod
    def _from_proto(cls, proto: "ProtoTransformation"):
        function_mapping = {
            "sql_transformation": SparkSqlTransformation,
        }

        function_type: str = proto.WhichOneof("type")
        if function_type in function_mapping:
            function_class = function_mapping.get(function_type)
            return function_class.from_proto(proto)

        raise QwakException(f"Got unsupported function type: {function_type}")


class SparkSqlTransformation(BatchTransformation):
    """
    A Spark SQL transformation
    :param sql: A valid Spark SQL transformation
    Example transformation:
    .. code-block:: python
        SparkSqlTransformation("SELECT user_id, age FROM data_source")
    """

    def __init__(self, sql):
        self._sql = sql

    def _to_proto(self) -> ProtoTransformation:
        return ProtoTransformation(
            sql_transformation=ProtoSqlTransformation(sql=self._sql, function_names=[]),
        )

    @classmethod
    def _from_proto(cls, proto: "ProtoSqlTransformation"):
        return cls(sql=proto.sql_transformation.sql)
