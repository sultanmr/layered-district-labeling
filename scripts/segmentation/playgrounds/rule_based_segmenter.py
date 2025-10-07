from typing import Dict, List
import pandas as pd
from ..base import SegmentationStrategy

class PlaygroundsRuleBasedSegmenter(SegmentationStrategy):
    def __init__(self, threshold_multiplier: float = 1.0):
        self.threshold_multiplier = threshold_multiplier
        
    def segment(self, features: pd.DataFrame):
        
        median_per_capita = features['playgrounds_per_capita'].median()

        features['#well-served'] = features['playgrounds_per_capita'] > median_per_capita
        features['#underserved'] = features['playgrounds_per_capita'] < median_per_capita * 0.5

        result_dict = {}
        for _, row in features.iterrows():
            tags = []
            if row['#well-served']:
                tags.append('#well-served')
            if row['#underserved']:
                tags.append('#underserved')
            result_dict[row['district']] = tags
        
        return result_dict
