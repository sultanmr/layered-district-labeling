from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class DistrictsPopStatFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['districts_pop_stat_per_capita'] = (
            features['num_districts_pop_stat'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['districts_pop_stat_per_capita'] = features['districts_pop_stat_per_capita'].fillna(0)
        
        return features
