from ..base import DataLoader
from .sql_queries import CRIME_STATISTICS_QUERY


class CrimeStatisticsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return CRIME_STATISTICS_QUERY
