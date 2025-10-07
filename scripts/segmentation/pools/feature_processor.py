from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class PoolsFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['pools_per_capita'] = (
            features['num_pools'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['pools_per_capita'] = features['pools_per_capita'].fillna(0)
        
        return features
