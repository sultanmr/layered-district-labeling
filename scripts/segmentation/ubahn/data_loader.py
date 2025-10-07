from ..base import DataLoader
from .sql_queries import UBAHN_QUERY


class UbahnDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return UBAHN_QUERY
