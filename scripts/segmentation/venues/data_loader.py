from ..base import DataLoader
from .sql_queries import VENUES_QUERY


class VenuesDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return VENUES_QUERY
