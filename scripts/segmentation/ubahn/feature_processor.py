from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class UbahnFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['ubahn_per_capita'] = (
            features['num_ubahn'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['ubahn_per_capita'] = features['ubahn_per_capita'].fillna(0)
        
        return features
