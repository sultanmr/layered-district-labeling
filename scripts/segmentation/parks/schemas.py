from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class TagCategory(str, Enum):
    MAINTENANCE = "maintenance_quality"
    PARK_SIZE = "park_size" 
    SPACE_AVAILABILITY = "space_availability"
    QUANTITY = "quantity"

class GreenSpaceTag(BaseModel):
    """Pydantic model for green space tags with validation rules"""
    tag: str = Field(..., description="The hashtag name")
    category: TagCategory = Field(..., description="Category of the tag")
    description: str = Field(..., description="Description of what the tag represents")
    rule: str = Field(..., description="The rule that triggers this tag")
    
    @classmethod
    def get_all_tags(cls):
        """Return all predefined green space tags with their rules"""
        return [
            # Maintenance Quality tags
            GreenSpaceTag(
                tag="#well-maintained",
                category=TagCategory.MAINTENANCE,
                description="Above median maintenance score",
                rule="maintenance_score > median_maintenance"
            ),
            GreenSpaceTag(
                tag="#needs-attention", 
                category=TagCategory.MAINTENANCE,
                description="Below 70% of median maintenance score",
                rule="maintenance_score < 0.7 * median_maintenance"
            ),
            
            # Park Size tags
            GreenSpaceTag(
                tag="#large-park",
                category=TagCategory.PARK_SIZE, 
                description="Above median average park size",
                rule="avg_park_size > median_park_size"
            ),
            
            # Space Availability tags
            GreenSpaceTag(
                tag="#spacious",
                category=TagCategory.SPACE_AVAILABILITY,
                description="Above median green space per capita", 
                rule="green_space_per_capita > median_per_capita"
            ),
            GreenSpaceTag(
                tag="#crowded",
                category=TagCategory.SPACE_AVAILABILITY,
                description="Below 50% of median green space per capita",
                rule="green_space_per_capita < 0.5 * median_per_capita"
            ),
            
            # Quantity tags
            GreenSpaceTag(
                tag="#many-parks",
                category=TagCategory.QUANTITY,
                description="Above median number of green spaces",
                rule="num_green_spaces > median_num_spaces"
            )
        ]

class TagAnalysisResult(BaseModel):
    """Pydantic model for tag analysis results"""
    district: str = Field(..., description="Name of the district")
    tags: List[str] = Field(..., description="List of applicable tags")
    reasoning: str = Field(..., description="Explanation of why tags were assigned")
    statistics: dict = Field(..., description="Statistical data used for decision making")
    
class ClusterTagAssignment(BaseModel):
    """Pydantic model for cluster-level tag assignments"""
    cluster_id: int = Field(..., description="Cluster identifier")
    tags: List[str] = Field(..., description="Tags assigned to this cluster")
    characteristics: dict = Field(..., description="Statistical characteristics of the cluster")
    reasoning: str = Field(..., description="Explanation for tag assignment")