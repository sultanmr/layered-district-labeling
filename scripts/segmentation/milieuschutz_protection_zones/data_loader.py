from ..base import DataLoader
from .sql_queries import MILIEUSCHUTZ_PROTECTION_ZONES_QUERY


class MilieuschutzProtectionZonesDataLoader(DataLoader):

    @property
    def query(self) -> str:
        return MILIEUSCHUTZ_PROTECTION_ZONES_QUERY
