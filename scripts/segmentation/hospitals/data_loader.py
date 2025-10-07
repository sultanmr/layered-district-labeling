from ..base import DataLoader
from .sql_queries import HOSPITALS_QUERY


class HospitalsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return HOSPITALS_QUERY
