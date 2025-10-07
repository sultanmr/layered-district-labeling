from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from langchain.tools import Tool
from ..react_base import ReactSegmentationStrategy
from .schemas import BusTramTag

# Set up logger
logger = logging.getLogger(__name__)

class BusTramStopsReactSegmenter(ReactSegmentationStrategy):
    """ReAct agent for bus/tram stops segmentation with reasoning capabilities"""
    
    def __init__(self):
        tools = self._create_all_tools()
        cols = ['stops_per_capita', 'coverage_score', 'service_quality_score', 'num_stops', 'neighborhoods_covered']
        # Initialize through parent class which handles LLM setup
        super().__init__(tools, cols, "gemini")

    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment districts using ReAct agent reasoning with feature storage for tools"""
        # Store features for tool access - make sure it's available before calling parent
        self.features_df = features.copy()
        
        try:
            # Create analysis prompt with the stored features
            analysis_prompt = self._create_analysis_prompt("bus_tram_stops", features, BusTramTag.get_all_tags())
            return self._get_results(analysis_prompt, features)

        except Exception as e:
            logger.error(f"ReAct agent error: {e}")

    def _create_tools(self):
        """Create bus/tram stops analysis specific tools using superclass helpers"""
        return [
            self._create_density_analysis_tool(
                "analyze_transport_density",
                "Analyze transport stop density and categorize levels using dynamic quartile-based thresholds. Input should be a string with the column name like 'stops_per_capita'.",
                lambda data: self._analyze_density(data, "transport density")
            ),
            self._create_quality_assessment_tool(
                "evaluate_service_coverage",
                "Evaluate service coverage and categorize levels using dynamic data-driven thresholds. Input should be a string with the column name like 'coverage_score'.",
                lambda data: self._evaluate_quality(data, "service coverage")
            ),
            self._create_quality_assessment_tool(
                "assess_service_quality",
                "Assess service quality with dynamic scoring. Input should be a string with the column name like 'service_quality_score'.",
                lambda data: self._evaluate_quality(data, "service quality")
            ),
            Tool(
                name="calculate_coverage_index",
                func=self._calculate_coverage_index_wrapper,
                description="Calculate coverage index with dynamic weighting based on data variability. Input should be a string with two column names separated by comma like 'num_stops,neighborhoods_covered'."
            )
        ]
    
    def _calculate_coverage_index_wrapper(self, columns_input: str) -> dict:
        """Wrapper for calculate_coverage_index that accepts string input"""
        if hasattr(self, 'features_df'):
            # Clean up column names - remove any extra quotes from each column
            columns = [col.strip().strip("'\"") for col in columns_input.split(',')]
            if len(columns) == 2 and all(col in self.features_df.columns for col in columns):
                return self._calculate_coverage_index(
                    self.features_df[columns[0]],
                    self.features_df[columns[1]]
                )
        return {"error": "Invalid columns input. Expected format: 'column1,column2'"}
    
    def _calculate_coverage_index(self, stops: pd.Series, neighborhoods: pd.Series) -> dict:
        """Calculate coverage index with dynamic weighting"""
        # Dynamic weighting based on data variability
        stops_std = stops.std()
        neighborhoods_std = neighborhoods.std()
        total_std = stops_std + neighborhoods_std
        
        if total_std > 0:
            stops_weight = stops_std / total_std
            neighborhoods_weight = neighborhoods_std / total_std
        else:
            stops_weight = 0.6
            neighborhoods_weight = 0.4
            
        coverage_index = (stops * stops_weight) + (neighborhoods * neighborhoods_weight)
        stats = self._calculate_statistics(coverage_index)
        distribution = self._analyze_distribution(coverage_index)
        
        return {
            "scores": coverage_index.tolist(),
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Coverage index range from {stats['min']:.2f} to {stats['max']:.2f} with dynamic weighting",
            "dynamic_weights": {
                "stops_weight": float(stops_weight),
                "neighborhoods_weight": float(neighborhoods_weight)
            }
        }
    
    def _calculate_coverage_index(self, stops: pd.Series, neighborhoods: pd.Series) -> dict:
        """Calculate coverage index with dynamic weighting"""
        # Dynamic weighting based on data variability
        stops_std = stops.std()
        neighborhoods_std = neighborhoods.std()
        total_std = stops_std + neighborhoods_std
        
        if total_std > 0:
            stops_weight = stops_std / total_std
            neighborhoods_weight = neighborhoods_std / total_std
        else:
            stops_weight = 0.6
            neighborhoods_weight = 0.4
            
        coverage_index = (stops * stops_weight) + (neighborhoods * neighborhoods_weight)
        stats = self._calculate_statistics(coverage_index)
        distribution = self._analyze_distribution(coverage_index)
        
        return {
            "scores": coverage_index.tolist(),
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Coverage index range from {stats['min']:.2f} to {stats['max']:.2f} with dynamic weighting",
            "dynamic_weights": {
                "stops_weight": float(stops_weight),
                "neighborhoods_weight": float(neighborhoods_weight)
            }
        }