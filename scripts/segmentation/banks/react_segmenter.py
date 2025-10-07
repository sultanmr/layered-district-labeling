from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from langchain.tools import Tool
from ..react_base import ReactSegmentationStrategy
from .schemas import BankTag

# Set up logger
logger = logging.getLogger(__name__)

class BanksReactSegmenter(ReactSegmentationStrategy):
    """ReAct agent for banking services segmentation with reasoning capabilities"""
    
    def __init__(self):
        tools = self._create_all_tools()
        cols = ['banks_per_capita', 'atms_per_capita', 'accessibility_score', 'service_availability_score', 'num_banks', 'num_atms']
        # Initialize through parent class which handles LLM setup
        super().__init__(tools, cols, "gemini")

    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment districts using ReAct agent reasoning with feature storage for tools"""
        # Store features for tool access - make sure it's available before calling parent
        self.features_df = features.copy()
        
        try:
            # Create analysis prompt with the stored features
            analysis_prompt = self._create_analysis_prompt("banks", features, BankTag.get_all_tags())
            return self._get_results(analysis_prompt, features)

        except Exception as e:
            logger.error(f"ReAct agent error: {e}")

    def _create_tools(self):
        """Create banking analysis specific tools using superclass helpers"""
        return [
            self._create_density_analysis_tool(
                "analyze_bank_density",
                "Analyze bank density and categorize levels using dynamic quartile-based thresholds. Input should be a string with the column name like 'banks_per_capita'.",
                lambda data: self._analyze_density(data, "bank density")
            ),
            self._create_quality_assessment_tool(
                "evaluate_atm_availability",
                "Evaluate ATM availability and categorize levels using dynamic data-driven thresholds. Input should be a string with the column name like 'atms_per_capita'.",
                lambda data: self._evaluate_quality(data, "ATM availability")
            ),
            Tool(
                name="assess_service_quality",
                func=self._assess_service_quality_wrapper,
                description="Assess banking service quality with dynamic scoring. Input should be a string with two column names separated by comma like 'accessibility_score,service_availability_score'."
            ),
            self._create_quality_assessment_tool(
                "calculate_accessibility_index",
                "Calculate accessibility index with dynamic weighting based on data variability. Input should be a string with the column name like 'accessibility_score'.",
                self._calculate_accessibility_index
            )
        ]
    
    def _assess_service_quality_wrapper(self, columns_input: str) -> dict:
        """Wrapper for assess_service_quality that accepts string input"""
        if hasattr(self, 'features_df'):
            # Clean up column names - remove any extra quotes from each column
            columns = [col.strip().strip("'\"") for col in columns_input.split(',')]
            if len(columns) == 2 and all(col in self.features_df.columns for col in columns):
                return self._assess_service_quality(
                    self.features_df[columns[0]],
                    self.features_df[columns[1]]
                )
        return {"error": "Invalid columns input. Expected format: 'column1,column2'"}
    
    def _assess_service_quality(self, accessibility: pd.Series, service_availability: pd.Series) -> dict:
        """Assess service quality with dynamic patterns"""
        accessibility_stats = self._calculate_statistics(accessibility)
        service_stats = self._calculate_statistics(service_availability)
        accessibility_distribution = self._analyze_distribution(accessibility)
        service_distribution = self._analyze_distribution(service_availability)
        
        # Dynamic pattern analysis based on data characteristics
        acc_q1, acc_q3 = accessibility_stats["q1"], accessibility_stats["q3"]
        service_q1, service_q3 = service_stats["q1"], service_stats["q3"]
        
        if accessibility_stats["median"] > acc_q3 and service_stats["median"] > service_q3:
            quality = "excellent_service"
        elif accessibility_stats["median"] > acc_q1:
            quality = "good_service"
        else:
            quality = "basic_service"
            
        return {
            "service_quality": quality,
            "accessibility_statistics": accessibility_stats,
            "service_statistics": service_stats,
            "accessibility_distribution": accessibility_distribution,
            "service_distribution": service_distribution,
            "interpretation": f"Service quality: {quality.replace('_', ' ')} based on dynamic quartile analysis"
        }
    
    def _calculate_accessibility_index(self, accessibility_scores: pd.Series) -> dict:
        """Calculate accessibility index with dynamic weighting"""
        stats = self._calculate_statistics(accessibility_scores)
        distribution = self._analyze_distribution(accessibility_scores)
        
        return {
            "scores": accessibility_scores.tolist(),
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Accessibility scores range from {stats['min']:.2f} to {stats['max']:.2f}"
        }