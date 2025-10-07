from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class PlaygroundsFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['playgrounds_per_capita'] = (
            features['num_playgrounds'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['playgrounds_per_capita'] = features['playgrounds_per_capita'].fillna(0)
        
        return features
