from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from langchain.tools import Tool
from ..react_base import ReactSegmentationStrategy
from .schemas import LongTermListingsTag

logger = logging.getLogger(__name__)

class LongTermListingsReactSegmenter(ReactSegmentationStrategy):
    def __init__(self):
        tools = self._create_all_tools()
        cols = ['long_term_listings_per_capita', 'num_long_term_listings']
        super().__init__(tools, cols, "openai")

    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        self.features_df = features.copy()
        
        try:
            analysis_prompt = self._create_analysis_prompt("long_term_listings", features, LongTermListingsTag.get_all_tags())
            return self._get_results(analysis_prompt, features)
        except Exception as e:
            logger.error(f"ReAct agent error: {e}")

    def _create_tools(self):
        """Create long_term_listings analysis specific tools using superclass helpers"""
        return [
            self._create_density_analysis_tool(
                "analyze_density",
                "Analyze long_term_listings density using dynamic thresholds.",
                lambda data: self._analyze_density(data, "long_term_listings density")
            )
        ]
        
