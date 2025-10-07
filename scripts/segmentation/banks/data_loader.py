from ..base import DataLoader
from .sql_queries import BANKS_QUERY


class BanksDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return BANKS_QUERY