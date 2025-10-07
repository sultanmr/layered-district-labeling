import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class LandPricesLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'land_prices_per_capita': row['land_prices_per_capita'],
                'num_land_prices': row['num_land_prices']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these land prices statistics for a district:

        District Profile:
        - Land Prices per capita: {district_summary['land_prices_per_capita']:.4f}
        - Number of land prices: {district_summary['num_land_prices']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
