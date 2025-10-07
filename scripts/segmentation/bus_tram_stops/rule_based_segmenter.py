from typing import Dict, List
import pandas as pd
from ..base import SegmentationStrategy

class BusTramStopsRuleBasedSegmenter(SegmentationStrategy):
    """Rule-based segmentation for bus and tram stops"""
    
    def __init__(self, threshold_multiplier: float = 1.0):
        self.threshold_multiplier = threshold_multiplier
        
    def segment(self, features: pd.DataFrame):
        
        median_stops_per_capita = features['stops_per_capita'].median()
        median_coverage = features['coverage_score'].median()
        median_service_quality = features['service_quality_score'].median()

        features['#well-served'] = features['stops_per_capita'] > median_stops_per_capita
        features['#transport-desert'] = features['stops_per_capita'] < median_stops_per_capita * 0.5
        features['#good-coverage'] = features['coverage_score'] > median_coverage
        features['#limited-coverage'] = features['coverage_score'] < median_coverage * 0.5
        features['#high-quality-service'] = features['service_quality_score'] > median_service_quality

        result_dict = {}
        for _, row in features.iterrows():
            tags = []
            if row['#well-served']:
                tags.append('#well-served')
            if row['#transport-desert']:
                tags.append('#transport-desert')
            if row['#good-coverage']:
                tags.append('#good-coverage')
            if row['#limited-coverage']:
                tags.append('#limited-coverage')
            if row['#high-quality-service']:
                tags.append('#high-quality-service')
            result_dict[row['district']] = tags
        
        return result_dict