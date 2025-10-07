from ..base import DataLoader
from .sql_queries import LONG_TERM_LISTINGS_QUERY


class LongTermListingsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return LONG_TERM_LISTINGS_QUERY
