import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class DistrictsPopStatLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'districts_pop_stat_per_capita': row['districts_pop_stat_per_capita'],
                'num_districts_pop_stat': row['num_districts_pop_stat']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these districts pop stat statistics for a district:

        District Profile:
        - Districts Pop Stat per capita: {district_summary['districts_pop_stat_per_capita']:.4f}
        - Number of districts pop stat: {district_summary['num_districts_pop_stat']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
