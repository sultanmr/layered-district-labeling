from ..base import DataLoader
from .sql_queries import KINDERGARTENS_QUERY


class KindergartensDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return KINDERGARTENS_QUERY
