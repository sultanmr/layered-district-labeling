import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class RentStatsPerNeighborhoodLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'rent_stats_per_neighborhood_per_capita': row['rent_stats_per_neighborhood_per_capita'],
                'num_rent_stats_per_neighborhood': row['num_rent_stats_per_neighborhood']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these rent stats per neighborhood statistics for a district:

        District Profile:
        - Rent Stats Per Neighborhood per capita: {district_summary['rent_stats_per_neighborhood_per_capita']:.4f}
        - Number of rent stats per neighborhood: {district_summary['num_rent_stats_per_neighborhood']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
