from ..base import DataLoader
from .sql_queries import POOLS_QUERY


class PoolsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return POOLS_QUERY
