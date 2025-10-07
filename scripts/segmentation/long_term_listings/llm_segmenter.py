import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class LongTermListingsLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'long_term_listings_per_capita': row['long_term_listings_per_capita'],
                'num_long_term_listings': row['num_long_term_listings']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these long term listings statistics for a district:

        District Profile:
        - Long Term Listings per capita: {district_summary['long_term_listings_per_capita']:.4f}
        - Number of long term listings: {district_summary['num_long_term_listings']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
