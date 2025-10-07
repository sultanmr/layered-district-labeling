import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class UniversitiesLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'universities_per_capita': row['universities_per_capita'],
                'num_universities': row['num_universities']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these universities statistics for a district:

        District Profile:
        - Universities per capita: {district_summary['universities_per_capita']:.4f}
        - Number of universities: {district_summary['num_universities']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
