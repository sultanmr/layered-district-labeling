from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class KindergartensFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['kindergartens_per_capita'] = (
            features['num_kindergartens'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['kindergartens_per_capita'] = features['kindergartens_per_capita'].fillna(0)
        
        return features
