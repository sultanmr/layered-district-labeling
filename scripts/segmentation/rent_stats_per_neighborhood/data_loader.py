from ..base import DataLoader
from .sql_queries import RENT_STATS_PER_NEIGHBORHOOD_QUERY


class RentStatsPerNeighborhoodDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return RENT_STATS_PER_NEIGHBORHOOD_QUERY
