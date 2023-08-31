from dataclasses import dataclass
from datetime import datetime
from typing import Union


@dataclass
class Context:
    start_time: Union[str, datetime] = "${qwak_ingestion_start_timestamp}"
    end_time: Union[str, datetime] = "${qwak_ingestion_end_timestamp}"
