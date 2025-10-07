from ..base import DataLoader
from .sql_queries import UNIVERSITIES_QUERY


class UniversitiesDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return UNIVERSITIES_QUERY
