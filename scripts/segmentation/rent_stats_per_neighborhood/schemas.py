from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class TagCategory(str, Enum):
    DENSITY = "density"
    QUALITY = "quality"

class RentStatsPerNeighborhoodTag(BaseModel):
    tag: str = Field(..., description="The hashtag name")
    category: TagCategory = Field(..., description="Category of the tag")
    description: str = Field(..., description="Description of what the tag represents")
    rule: str = Field(..., description="The rule that triggers this tag")
    
    @classmethod
    def get_all_tags(cls):
        return [
            # Density tags
            RentStatsPerNeighborhoodTag(
                tag="#well-served",
                category=TagCategory.DENSITY,
                description="Above median number of rent_stats_per_neighborhood per capita",
                rule="rent_stats_per_neighborhood_per_capita > median_rent_stats_per_neighborhood_per_capita"
            ),
            RentStatsPerNeighborhoodTag(
                tag="#underserved", 
                category=TagCategory.DENSITY,
                description="Below 50% of median rent_stats_per_neighborhood per capita",
                rule="rent_stats_per_neighborhood_per_capita < 0.5 * median_rent_stats_per_neighborhood_per_capita"
            )
        ]

class TagAnalysisResult(BaseModel):
    district: str = Field(..., description="Name of the district")
    tags: List[str] = Field(..., description="List of applicable tags")
    reasoning: str = Field(..., description="Explanation of why tags were assigned")
    statistics: dict = Field(..., description="Statistical data used for decision making")
    
class ClusterTagAssignment(BaseModel):
    cluster_id: int = Field(..., description="Cluster identifier")
    tags: List[str] = Field(..., description="Tags assigned to this cluster")
    characteristics: dict = Field(..., description="Statistical characteristics of the cluster")
    reasoning: str = Field(..., description="Explanation for tag assignment")
