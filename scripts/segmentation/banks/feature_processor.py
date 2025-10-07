from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class BanksFeatureProcessor(FeatureProcessor):
    """Processes banking features"""
    
    def process_features(self, raw_data):
        features = raw_data.copy()

        # 1. Calculate core banking metrics
        features['banks_per_capita'] = (
            features['num_banks'] / 
            features['population'].replace(0, 1)
        )
        
        features['atms_per_capita'] = (
            features['num_atms'] / 
            features['population'].replace(0, 1)
        )
        
        # 2. Accessibility scoring
        features['accessibility_score'] = (
            features['num_wheelchair_accessible'] / 
            features['num_banks'].replace(0, 1)
        )
        
        # 3. Service availability scoring
        features['service_availability_score'] = (
            features['avg_opening_hours_availability'] * 0.6 + 
            (features['num_atms'] / features['num_banks'].replace(0, 1)) * 0.4
        )

        # Fill NaN values
        features['banks_per_capita'] = features['banks_per_capita'].fillna(0)
        features['atms_per_capita'] = features['atms_per_capita'].fillna(0)
        features['accessibility_score'] = features['accessibility_score'].fillna(0)
        features['service_availability_score'] = features['service_availability_score'].fillna(0)
        
        return features