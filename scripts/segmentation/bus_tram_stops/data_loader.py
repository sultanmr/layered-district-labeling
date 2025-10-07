from ..base import DataLoader
from .sql_queries import BUS_TRAM_STOPS_QUERY


class BusTramStopsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return BUS_TRAM_STOPS_QUERY