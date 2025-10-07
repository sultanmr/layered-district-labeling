from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from langchain.tools import Tool
from ..react_base import ReactSegmentationStrategy
from .schemas import MilieuschutzProtectionZonesTag

logger = logging.getLogger(__name__)

class MilieuschutzProtectionZonesReactSegmenter(ReactSegmentationStrategy):
    def __init__(self):
        tools = self._create_all_tools()
        cols = ['milieuschutz_protection_zones_per_capita', 'num_milieuschutz_protection_zones']
        super().__init__(tools, cols, "openai")

    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        self.features_df = features.copy()
        
        try:
            analysis_prompt = self._create_analysis_prompt("milieuschutz_protection_zones", features, MilieuschutzProtectionZonesTag.get_all_tags())
            return self._get_results(analysis_prompt, features)
        except Exception as e:
            logger.error(f"ReAct agent error: {e}")

    def _create_tools(self):
        """Create milieuschutz_protection_zones analysis specific tools using superclass helpers"""
        return [
            self._create_density_analysis_tool(
                "analyze_density",
                "Analyze milieuschutz_protection_zones density using dynamic thresholds.",
                lambda data: self._analyze_density(data, "milieuschutz_protection_zones density")
            )
        ]
        
