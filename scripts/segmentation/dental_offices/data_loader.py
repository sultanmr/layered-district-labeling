from ..base import DataLoader
from .sql_queries import DENTAL_OFFICES_QUERY


class DentalOfficesDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return DENTAL_OFFICES_QUERY
