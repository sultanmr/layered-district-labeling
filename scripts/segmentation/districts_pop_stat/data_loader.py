from ..base import DataLoader
from .sql_queries import DISTRICTS_POP_STAT_QUERY


class DistrictsPopStatDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return DISTRICTS_POP_STAT_QUERY
