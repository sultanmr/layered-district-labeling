from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class LandPricesFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['land_prices_per_capita'] = (
            features['num_land_prices'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['land_prices_per_capita'] = features['land_prices_per_capita'].fillna(0)
        
        return features
