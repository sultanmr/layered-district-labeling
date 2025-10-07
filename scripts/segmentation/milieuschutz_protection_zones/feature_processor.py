from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class MilieuschutzProtectionZonesFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data):
        features = raw_data.copy()

        # Calculate core metrics
        features['milieuschutz_protection_zones_per_capita'] = (
            features['num_milieuschutz_protection_zones'] / 
            features['population'].replace(0, 1)
        )

        # Fill NaN values
        features['milieuschutz_protection_zones_per_capita'] = features['milieuschutz_protection_zones_per_capita'].fillna(0)
        
        return features
