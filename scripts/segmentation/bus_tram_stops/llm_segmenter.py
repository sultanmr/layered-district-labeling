import pandas as pd
from segmentation.base import LLMSegmentationStrategy

class BusTramStopsLlmSegmenter(LLMSegmentationStrategy):
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
                'stops_per_capita': row['stops_per_capita'],
                'coverage_score': row['coverage_score'],
                'service_quality_score': row['service_quality_score'],
                'num_stops': row['num_stops'],
                'neighborhoods_covered': row['neighborhoods_covered']
            }
            
            prompt = self._create_prompt(summary)
            llm_tags = self._get_tags(prompt)
            district_tags[district] = llm_tags

        return district_tags

    def _create_prompt(self, district_summary: dict) -> str:
        """Generate a prompt for Gemini with district statistics"""
        return f"""
        Analyze these public transportation statistics for a district:

        District Profile:
        - Stops per capita: {district_summary['stops_per_capita']:.4f}
        - Coverage score: {district_summary['coverage_score']:.2f}
        - Service quality score: {district_summary['service_quality_score']:.2f}
        - Number of stops: {district_summary['num_stops']}
        - Neighborhoods covered: {district_summary['neighborhoods_covered']}

        Suggest 3-6 hyphenated tags (e.g., 'well-served') that:
        1. Reflect the public transportation characteristics
        2. Are descriptive and meaningful
        3. Are specific to this district's transportation profile

        Return ONLY comma-separated tags, nothing else:
        """