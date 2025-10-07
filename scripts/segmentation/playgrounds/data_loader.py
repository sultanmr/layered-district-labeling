from ..base import DataLoader
from .sql_queries import PLAYGROUNDS_QUERY


class PlaygroundsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return PLAYGROUNDS_QUERY
