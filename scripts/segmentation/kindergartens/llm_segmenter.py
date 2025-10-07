import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class KindergartensLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'kindergartens_per_capita': row['kindergartens_per_capita'],
                'num_kindergartens': row['num_kindergartens']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these kindergartens statistics for a district:

        District Profile:
        - Kindergartens per capita: {district_summary['kindergartens_per_capita']:.4f}
        - Number of kindergartens: {district_summary['num_kindergartens']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
