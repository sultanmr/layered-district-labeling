import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class BanksLlmSegmenter(LLMSegmentationStrategy):
    def __init__(self):
        super().__init__()

    def segment(self, features: pd.DataFrame):
        # Use the actual feature values for each district instead of clustering
        district_tags = {}
        
        # Generate Gemini-powered tags for each district based on its actual data
        for _, row in features.iterrows():
            district = row['district']
            # Extract the numeric features for this district
            summary = {
                'banks_per_capita': row['banks_per_capita'],
                'atms_per_capita': row['atms_per_capita'],
                'accessibility_score': row['accessibility_score'],
                'service_availability_score': row['service_availability_score'],
                'num_banks': row['num_banks'],
                'num_atms': row['num_atms']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        """Generate a prompt for Gemini with district statistics"""
        return f"""
        Analyze these banking statistics for a district:

        District Profile:
        - Banks per capita: {district_summary['banks_per_capita']:.4f}
        - ATMs per capita: {district_summary['atms_per_capita']:.4f}
        - Accessibility score: {district_summary['accessibility_score']:.2f}
        - Service availability score: {district_summary['service_availability_score']:.2f}
        - Number of banks: {district_summary['num_banks']}
        - Number of ATMs: {district_summary['num_atms']}

        Suggest 3-6 hyphenated tags (e.g., 'well-banked') that:
        1. Reflect the banking service characteristics
        2. Are descriptive and meaningful
        3. Are specific to this district's banking profile

        Return ONLY comma-separated tags, nothing else:
        """