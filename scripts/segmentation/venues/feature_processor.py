from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class VenuesFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['venues_per_capita'] = (
            features['num_venues'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['venues_per_capita'] = features['venues_per_capita'].fillna(0)
        
        return features
