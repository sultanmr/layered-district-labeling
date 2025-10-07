from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class LongTermListingsFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['long_term_listings_per_capita'] = (
            features['num_long_term_listings'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['long_term_listings_per_capita'] = features['long_term_listings_per_capita'].fillna(0)
        
        return features
