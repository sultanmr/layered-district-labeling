from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class TagCategory(str, Enum):
    TRANSPORT_DENSITY = "transport_density"
    SERVICE_COVERAGE = "service_coverage" 
    SERVICE_QUALITY = "service_quality"
    ACCESSIBILITY = "accessibility"

class BusTramTag(BaseModel):
    """Pydantic model for bus/tram tags with validation rules"""
    tag: str = Field(..., description="The hashtag name")
    category: TagCategory = Field(..., description="Category of the tag")
    description: str = Field(..., description="Description of what the tag represents")
    rule: str = Field(..., description="The rule that triggers this tag")
    
    @classmethod
    def get_all_tags(cls):
        """Return all predefined bus/tram tags with their rules"""
        return [
            # Transport Density tags
            BusTramTag(
                tag="#well-served",
                category=TagCategory.TRANSPORT_DENSITY,
                description="Above median number of stops per capita",
                rule="stops_per_capita > median_stops_per_capita"
            ),
            BusTramTag(
                tag="#transport-desert", 
                category=TagCategory.TRANSPORT_DENSITY,
                description="Below 50% of median stops per capita",
                rule="stops_per_capita < 0.5 * median_stops_per_capita"
            ),
            
            # Service Coverage tags
            BusTramTag(
                tag="#good-coverage",
                category=TagCategory.SERVICE_COVERAGE,
                description="Above median coverage score",
                rule="coverage_score > median_coverage"
            ),
            BusTramTag(
                tag="#limited-coverage",
                category=TagCategory.SERVICE_COVERAGE,
                description="Below 50% of median coverage score",
                rule="coverage_score < 0.5 * median_coverage"
            ),
            
            # Service Quality tags
            BusTramTag(
                tag="#high-quality-service",
                category=TagCategory.SERVICE_QUALITY,
                description="Above median service quality score",
                rule="service_quality_score > median_service_quality"
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