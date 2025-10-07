from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class CrimeStatisticsFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['crime_statistics_per_capita'] = (
            features['num_crime_statistics'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['crime_statistics_per_capita'] = features['crime_statistics_per_capita'].fillna(0)
        
        return features
