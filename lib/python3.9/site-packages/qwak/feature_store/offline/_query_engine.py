import uuid
from abc import ABC, abstractmethod

import pandas as pd


class BaseQueryEngine(ABC):
    @abstractmethod
    def upload_table(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def run_query(self, query: str):
        pass

    @abstractmethod
    def read_pandas_from_query(self, query: str, parse_dates=None):
        pass

    @staticmethod
    @abstractmethod
    def get_quotes():
        pass

    class JoinTableSpec:
        def __init__(self, join_tables_db_name: str, quotes: str):
            self.table_name = str(uuid.uuid4())
            self.join_table_full_path = f"{quotes}{join_tables_db_name}{quotes}.{quotes}{self.table_name}{quotes}"
