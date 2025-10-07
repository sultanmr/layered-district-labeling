from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class UniversitiesFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['universities_per_capita'] = (
            features['num_universities'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['universities_per_capita'] = features['universities_per_capita'].fillna(0)
        
        return features
