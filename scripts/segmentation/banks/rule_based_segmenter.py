from typing import Dict, List
import pandas as pd
from ..base import SegmentationStrategy

class BanksRuleBasedSegmenter(SegmentationStrategy):
    """Rule-based segmentation for banking services"""
    
    def __init__(self, threshold_multiplier: float = 1.0):
        self.threshold_multiplier = threshold_multiplier
        
    def segment(self, features: pd.DataFrame):
        
        median_banks_per_capita = features['banks_per_capita'].median()
        median_atms_per_capita = features['atms_per_capita'].median()
        median_accessibility = features['accessibility_score'].median()
        median_service_availability = features['service_availability_score'].median()

        features['#well-banked'] = features['banks_per_capita'] > median_banks_per_capita
        features['#banking-desert'] = features['banks_per_capita'] < median_banks_per_capita * 0.5
        features['#good-service'] = features['service_availability_score'] > median_service_availability
        features['#accessible-banking'] = features['accessibility_score'] > median_accessibility
        features['#atm-rich'] = features['atms_per_capita'] > median_atms_per_capita
        features['#atm-poor'] = features['atms_per_capita'] < median_atms_per_capita * 0.5

        result_dict = {}
        for _, row in features.iterrows():
            tags = []
            if row['#well-banked']:
                tags.append('#well-banked')
            if row['#banking-desert']:
                tags.append('#banking-desert')
            if row['#good-service']:
                tags.append('#good-service')
            if row['#accessible-banking']:
                tags.append('#accessible-banking')
            if row['#atm-rich']:
                tags.append('#atm-rich')
            if row['#atm-poor']:
                tags.append('#atm-poor')
            result_dict[row['district']] = tags
        
        return result_dict