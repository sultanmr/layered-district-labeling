from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class DentalOfficesFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['dental_offices_per_capita'] = (
            features['num_dental_offices'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['dental_offices_per_capita'] = features['dental_offices_per_capita'].fillna(0)
        
        return features
