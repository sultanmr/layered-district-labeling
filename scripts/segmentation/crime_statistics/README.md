# Crime Statistics Segmentation Module

## 🏷️ Crime Statistics Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Crime Statistics characteristics:

- **#well-served**: Above median number of crime_statistics per capita (crime_statistics_per_capita > median_crime_statistics_per_capita)
- **#underserved**: Below 50% of median crime_statistics per capita (crime_statistics_per_capita < 0.5 * median_crime_statistics_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.crime_statistics` joined with regional statistics
2. Calculates key metrics:
      - crime_statistics_per_10k_residents

### Segmentation Approaches
#### Rule-Based
- Uses fixed thresholds based on median values
- Configurable threshold multiplier for adjusting sensitivity
- Generates consistent tags based on simple comparisons

#### LLM-Powered (Gemini)
- Uses actual district data values with LLM analysis
- Generates descriptive tags based on individual district characteristics
- Provides district-specific insights without clustering

#### ReAct Agent
- Uses reasoning and tool-based analysis with dynamic thresholds
- Incorporates multiple analytical tools for comprehensive assessment:
  
- Provides detailed reasoning for tag assignments

## 📊 Usage
```python
# For data loading
from segmentation.Crime Statistics.data_loader import CrimeStatisticsDataLoader
loader = CrimeStatisticsDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Crime Statistics.rule_based_segmenter import CrimeStatisticsRuleBasedSegmenter
rule_segmenter = CrimeStatisticsRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Crime Statistics.llm_segmenter import CrimeStatisticsLlmSegmenter
llm_segmenter = CrimeStatisticsLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Crime Statistics.react_segmenter import CrimeStatisticsReactSegmenter
react_segmenter = CrimeStatisticsReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Crime Statistics (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
