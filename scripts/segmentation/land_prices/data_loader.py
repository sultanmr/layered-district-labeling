from ..base import DataLoader
from .sql_queries import LAND_PRICES_QUERY


class LandPricesDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return LAND_PRICES_QUERY
