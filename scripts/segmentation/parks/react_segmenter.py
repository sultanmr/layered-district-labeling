from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from langchain.tools import Tool
from ..react_base import ReactSegmentationStrategy
from .schemas import  GreenSpaceTag

# Set up logger
logger = logging.getLogger(__name__)

class ParksReactSegmenter(ReactSegmentationStrategy):
    """ReAct agent for green spaces segmentation with reasoning capabilities"""
    
    def __init__(self):
        tools = self._create_all_tools()
        cols = ['green_space_per_capita', 'maintenance_score', 'avg_park_size', 'num_green_spaces', 'park_area_per_10k_residents']
        # Initialize through parent class which handles LLM setup
        super().__init__(tools, cols, "gemini")


    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment districts using ReAct agent reasoning with feature storage for tools"""
        # Store features for tool access - make sure it's available before calling parent
        self.features_df = features.copy()
        
        try:
            # Create analysis prompt with the stored features
            analysis_prompt = self._create_analysis_prompt("parks", features, GreenSpaceTag.get_all_tags())
            return self._get_results(analysis_prompt, features)

        except Exception as e:
            logger.error(f"ReAct agent error: {e}")

    def _create_tools(self):
        """Create green space analysis specific tools"""
        return [
            Tool(
                name="analyze_green_density",
                func=self._analyze_green_density_wrapper,
                description="Analyze green space per capita and categorize density levels using dynamic quartile-based thresholds. Input should be a string with the column name like 'green_space_per_capita'."
            ),
            Tool(
                name="evaluate_maintenance_quality",
                func=self._evaluate_maintenance_quality_wrapper,
                description="Evaluate maintenance scores and categorize quality levels using dynamic data-driven thresholds. Input should be a string with the column name like 'maintenance_score'."
            ),
            Tool(
                name="assess_park_size_distribution",
                func=self._assess_park_size_distribution_wrapper,
                description="Analyze park size distribution and identify patterns using dynamic statistical analysis. Input should be a string with two column names separated by comma like 'avg_park_size,num_green_spaces'."
            ),
            Tool(
                name="calculate_accessibility_score",
                func=self._calculate_accessibility_score_wrapper,
                description="Calculate accessibility score with dynamic weighting based on data variability. Input should be a string with two column names separated by comma like 'green_space_per_capita,num_green_spaces'."
            )            
        ]

    

    def _analyze_green_density_wrapper(self, column_name: str) -> dict:
        """Wrapper for analyze_green_density that accepts string input"""
        # Clean up column name - remove any extra quotes
        clean_column_name = column_name.strip().strip("'\"")
        
        if hasattr(self, 'features_df') and clean_column_name in self.features_df.columns:
            return self._analyze_green_density(self.features_df[clean_column_name])
        return {"error": f"Column '{clean_column_name}' not found in features data. Available columns: {list(self.features_df.columns) if hasattr(self, 'features_df') else 'None'}"}
    
      
       

    def _evaluate_maintenance_quality_wrapper(self, column_name: str) -> dict:
        """Wrapper for evaluate_maintenance_quality that accepts string input"""
        # Clean up column name - remove any extra quotes
        clean_column_name = column_name.strip().strip("'\"")
        
        if hasattr(self, 'features_df') and clean_column_name in self.features_df.columns:
            return self._evaluate_maintenance_quality(self.features_df[clean_column_name])
        return {"error": f"Column '{clean_column_name}' not found in features data. Available columns: {list(self.features_df.columns) if hasattr(self, 'features_df') else 'None'}"}
    
    def _assess_park_size_distribution_wrapper(self, columns_input: str) -> dict:
        """Wrapper for assess_park_size_distribution that accepts string input"""
        if hasattr(self, 'features_df'):
            # Clean up column names - remove any extra quotes from each column
            columns = [col.strip().strip("'\"") for col in columns_input.split(',')]
            if len(columns) == 2 and all(col in self.features_df.columns for col in columns):
                return self._assess_park_size_distribution(
                    self.features_df[columns[0]],
                    self.features_df[columns[1]]
                )
        return {"error": "Invalid columns input. Expected format: 'column1,column2'"}
    
    def _calculate_accessibility_score_wrapper(self, columns_input: str) -> dict:
        """Wrapper for calculate_accessibility_score that accepts string input"""
        if hasattr(self, 'features_df'):
            # Clean up column names - remove any extra quotes from each column
            columns = [col.strip().strip("'\"") for col in columns_input.split(',')]
            if len(columns) == 2 and all(col in self.features_df.columns for col in columns):
                return self._calculate_accessibility_score(
                    self.features_df[columns[0]],
                    self.features_df[columns[1]]
                )
        return {"error": "Invalid columns input. Expected format: 'column1,column2'"}
    
    
    def _analyze_green_density(self, per_capita_data: pd.Series) -> dict:
        """Analyze green space density patterns with dynamic thresholds"""
        stats = self._calculate_statistics(per_capita_data)
        distribution = self._analyze_distribution(per_capita_data)
        
        # Dynamic threshold calculation based on data distribution
        q1, q3 = stats["q1"], stats["q3"]
        iqr = stats["iqr"]
        
        # Calculate density levels based on quartiles
        if stats["median"] > q3 + iqr * 0.5:
            density_level = "very_high"
        elif stats["median"] > q3:
            density_level = "high"
        elif stats["median"] > q1:
            density_level = "medium"
        else:
            density_level = "low"
            
        return {
            "density_level": density_level,
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Overall {density_level.replace('_', ' ')} green space density based on dynamic quartile analysis",
            "dynamic_thresholds": {
                "very_high_threshold": float(q3 + iqr * 0.5),
                "high_threshold": float(q3),
                "medium_threshold": float(q1),
                "low_threshold": float(q1 - iqr * 0.5)
            }
        }
    
    def _evaluate_maintenance_quality(self, scores: pd.Series) -> dict:
        """Evaluate maintenance quality with dynamic thresholds"""
        stats = self._calculate_statistics(scores)
        distribution = self._analyze_distribution(scores)
        
        # Dynamic quality thresholds based on data distribution
        q1, q3 = stats["q1"], stats["q3"]
        
        if stats["median"] > q3 + (q3 - q1) * 0.5:
            quality = "excellent"
        elif stats["median"] > q3:
            quality = "good"
        elif stats["median"] > q1:
            quality = "fair"
        else:
            quality = "poor"
            
        return {
            "quality_level": quality,
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Overall {quality} maintenance quality based on dynamic quartile analysis",
            "dynamic_thresholds": {
                "excellent_threshold": float(q3 + (q3 - q1) * 0.5),
                "good_threshold": float(q3),
                "fair_threshold": float(q1),
                "poor_threshold": float(q1 - (q3 - q1) * 0.5)
            }
        }
    
    def _assess_park_size_distribution(self, sizes: pd.Series, counts: pd.Series) -> dict:
        """Assess park size distribution with dynamic patterns"""
        size_stats = self._calculate_statistics(sizes)
        count_stats = self._calculate_statistics(counts)
        size_distribution = self._analyze_distribution(sizes)
        count_distribution = self._analyze_distribution(counts)
        
        # Dynamic pattern analysis based on data characteristics
        size_q1, size_q3 = size_stats["q1"], size_stats["q3"]
        count_q1, count_q3 = count_stats["q1"], count_stats["q3"]
        
        if size_stats["median"] > size_q3 and count_stats["median"] > count_q3:
            pattern = "large_parks_good_distribution"
        elif size_stats["median"] > size_q1:
            pattern = "medium_parks"
        else:
            pattern = "small_parks_limited"
            
        return {
            "distribution_pattern": pattern,
            "size_statistics": size_stats,
            "count_statistics": count_stats,
            "size_distribution": size_distribution,
            "count_distribution": count_distribution,
            "interpretation": f"Park distribution pattern: {pattern.replace('_', ' ')} based on dynamic quartile analysis"
        }
    
    def _calculate_accessibility_score(self, per_capita: pd.Series, counts: pd.Series) -> dict:
        """Calculate accessibility scores with dynamic weighting"""
        # Dynamic weighting based on data variability
        per_capita_std = per_capita.std()
        counts_std = counts.std()
        total_std = per_capita_std + counts_std
        
        if total_std > 0:
            per_capita_weight = per_capita_std / total_std
            counts_weight = counts_std / total_std
        else:
            per_capita_weight = 0.6
            counts_weight = 0.4
            
        accessibility_scores = (per_capita * per_capita_weight) + (np.log1p(counts) * counts_weight)
        stats = self._calculate_statistics(accessibility_scores)
        distribution = self._analyze_distribution(accessibility_scores)
        
        return {
            "scores": accessibility_scores.tolist(),
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Accessibility scores range from {stats['min']:.2f} to {stats['max']:.2f} with dynamic weighting",
            "dynamic_weights": {
                "per_capita_weight": float(per_capita_weight),
                "counts_weight": float(counts_weight)
            }
        }
    
    
    
   