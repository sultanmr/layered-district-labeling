from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class BusTramStopsFeatureProcessor(FeatureProcessor):
    """Processes public transportation features"""
    
    def process_features(self, raw_data):
        features = raw_data.copy()

        # 1. Calculate core transportation metrics
        features['stops_per_capita'] = (
            features['num_stops'] / 
            features['population'].replace(0, 1)
        )
        
        # 2. Coverage scoring
        features['coverage_score'] = (
            features['neighborhoods_covered'] / 
            features['num_stops'].replace(0, 1)
        )
        
        # 3. Service quality scoring
        features['service_quality_score'] = (
            features['location_accuracy_score'] * 0.4 + 
            (features['stops_with_address'] / features['num_stops'].replace(0, 1)) * 0.3 +
            (features['unique_stop_names'] / features['num_stops'].replace(0, 1)) * 0.3
        )

        # Fill NaN values
        features['stops_per_capita'] = features['stops_per_capita'].fillna(0)
        features['coverage_score'] = features['coverage_score'].fillna(0)
        features['service_quality_score'] = features['service_quality_score'].fillna(0)
        
        return features