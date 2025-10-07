import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class CrimeStatisticsLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'crime_statistics_per_capita': row['crime_statistics_per_capita'],
                'num_crime_statistics': row['num_crime_statistics']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these crime statistics statistics for a district:

        District Profile:
        - Crime Statistics per capita: {district_summary['crime_statistics_per_capita']:.4f}
        - Number of crime statistics: {district_summary['num_crime_statistics']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
