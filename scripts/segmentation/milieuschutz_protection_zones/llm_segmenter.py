import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class MilieuschutzProtectionZonesLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        district_tags = {}
        
        for _, row in features.iterrows():
            district = row['district']
            summary = {
                'milieuschutz_protection_zones_per_capita': row['milieuschutz_protection_zones_per_capita'],
                'num_milieuschutz_protection_zones': row['num_milieuschutz_protection_zones']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        return f"""
        Analyze these milieuschutz protection zones statistics for a district:

        District Profile:
        - Milieuschutz Protection Zones per capita: {district_summary['milieuschutz_protection_zones_per_capita']:.4f}
        - Number of milieuschutz protection zones: {district_summary['num_milieuschutz_protection_zones']}

        Suggest 3-6 hyphenated tags that reflect these characteristics.

        Return ONLY comma-separated tags, nothing else:
        """
