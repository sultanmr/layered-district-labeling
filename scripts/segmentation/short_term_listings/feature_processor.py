from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class ShortTermListingsFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['short_term_listings_per_capita'] = (
            features['num_short_term_listings'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['short_term_listings_per_capita'] = features['short_term_listings_per_capita'].fillna(0)
        
        return features
