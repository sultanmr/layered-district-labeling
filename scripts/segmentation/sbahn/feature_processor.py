from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class SbahnFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['sbahn_per_capita'] = (
            features['num_sbahn'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['sbahn_per_capita'] = features['sbahn_per_capita'].fillna(0)
        
        return features
