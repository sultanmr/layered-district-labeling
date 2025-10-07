from ..base import DataLoader
from .sql_queries import SCHOOLS_QUERY


class SchoolsDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return SCHOOLS_QUERY
