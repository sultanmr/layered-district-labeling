from ..base import DataLoader
from .sql_queries import PARKS_QUERY


class ParksDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return PARKS_QUERY
