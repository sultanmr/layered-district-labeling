from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class HospitalsFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['hospitals_per_capita'] = (
            features['num_hospitals'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['hospitals_per_capita'] = features['hospitals_per_capita'].fillna(0)
        
        return features
