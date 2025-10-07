from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class RentStatsPerNeighborhoodFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['rent_stats_per_neighborhood_per_capita'] = (
            features['num_rent_stats_per_neighborhood'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['rent_stats_per_neighborhood_per_capita'] = features['rent_stats_per_neighborhood_per_capita'].fillna(0)
        
        return features
