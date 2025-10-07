from ..base import DataLoader
from .sql_queries import SHORT_TERM_LISTINGS_QUERY


class ShortTermListingsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return SHORT_TERM_LISTINGS_QUERY
