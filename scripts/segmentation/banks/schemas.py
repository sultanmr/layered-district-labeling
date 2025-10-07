from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class TagCategory(str, Enum):
    BANK_DENSITY = "bank_density"
    SERVICE_QUALITY = "service_quality" 
    ACCESSIBILITY = "accessibility"
    ATM_AVAILABILITY = "atm_availability"

class BankTag(BaseModel):
    """Pydantic model for bank tags with validation rules"""
    tag: str = Field(..., description="The hashtag name")
    category: TagCategory = Field(..., description="Category of the tag")
    description: str = Field(..., description="Description of what the tag represents")
    rule: str = Field(..., description="The rule that triggers this tag")
    
    @classmethod
    def get_all_tags(cls):
        """Return all predefined bank tags with their rules"""
        return [
            # Bank Density tags
            BankTag(
                tag="#well-banked",
                category=TagCategory.BANK_DENSITY,
                description="Above median number of banks per capita",
                rule="banks_per_capita > median_banks_per_capita"
            ),
            BankTag(
                tag="#banking-desert", 
                category=TagCategory.BANK_DENSITY,
                description="Below 50% of median banks per capita",
                rule="banks_per_capita < 0.5 * median_banks_per_capita"
            ),
            
            # Service Quality tags
            BankTag(
                tag="#good-service",
                category=TagCategory.SERVICE_QUALITY,
                description="Above median service availability score",
                rule="service_availability_score > median_service_availability"
            ),
            
            # Accessibility tags
            BankTag(
                tag="#accessible-banking",
                category=TagCategory.ACCESSIBILITY,
                description="Above median accessibility score",
                rule="accessibility_score > median_accessibility"
            ),
            
            # ATM Availability tags
            BankTag(
                tag="#atm-rich",
                category=TagCategory.ATM_AVAILABILITY,
                description="Above median ATMs per capita",
                rule="atms_per_capita > median_atms_per_capita"
            ),
            BankTag(
                tag="#atm-poor",
                category=TagCategory.ATM_AVAILABILITY,
                description="Below 50% of median ATMs per capita",
                rule="atms_per_capita < 0.5 * median_atms_per_capita"
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