import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class ParksLlmSegmenter(LLMSegmentationStrategy):
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
                'green_space_per_capita': row['green_space_per_capita'],
                'maintenance_score': row['maintenance_score'],
                'avg_park_size': row['avg_park_size'],
                'num_green_spaces': row['num_green_spaces']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        """Generate a prompt for Gemini with district statistics"""
        return f"""
        Analyze these green space statistics for a district:

        District Profile:
        - Green space per capita: {district_summary['green_space_per_capita']:.2f}
        - Maintenance score: {district_summary['maintenance_score']:.2f}
        - Average park size: {district_summary['avg_park_size']:.2f}
        - Number of green spaces: {district_summary['num_green_spaces']:.2f}

        Suggest 3-6 hyphenated tags (e.g., 'well-maintained') that:
        1. Reflect the quantitative characteristics
        2. Are descriptive and meaningful
        3. Are specific to this district's green space profile

        Return ONLY comma-separated tags, nothing else:
        """

