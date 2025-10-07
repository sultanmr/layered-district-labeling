from ..base import DataLoader
from .sql_queries import SBAHN_QUERY


class SbahnDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return SBAHN_QUERY
