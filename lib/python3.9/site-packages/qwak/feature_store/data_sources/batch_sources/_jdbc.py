from abc import ABC
from dataclasses import dataclass
from typing import Optional

from qwak.exceptions import QwakException
from qwak.feature_store.data_sources.batch_sources._batch import BaseBatchSource


@dataclass
class JdbcSource(BaseBatchSource, ABC):
    username_secret_name: Optional[str] = None
    password_secret_name: Optional[str] = None
    url: Optional[str] = None
    db_table: Optional[str] = None
    query: Optional[str] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not (bool(self.db_table) ^ bool(self.query)):
            raise QwakException("Only one of query and db_table must be set")
