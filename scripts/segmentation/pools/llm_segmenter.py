import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class PoolsLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'pools_per_capita': row['pools_per_capita'],
                'num_pools': row['num_pools']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these pools statistics for a district:

        District Profile:
        - Pools per capita: {district_summary['pools_per_capita']:.4f}
        - Number of pools: {district_summary['num_pools']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
