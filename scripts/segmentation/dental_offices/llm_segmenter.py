import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class DentalOfficesLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'dental_offices_per_capita': row['dental_offices_per_capita'],
                'num_dental_offices': row['num_dental_offices']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these dental offices statistics for a district:

        District Profile:
        - Dental Offices per capita: {district_summary['dental_offices_per_capita']:.4f}
        - Number of dental offices: {district_summary['num_dental_offices']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
