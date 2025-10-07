# Rent Stats Per Neighborhood Segmentation Module

## 🏷️ Rent Stats Per Neighborhood Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Rent Stats Per Neighborhood characteristics:

- **#well-served**: Above median number of rent_stats_per_neighborhood per capita (rent_stats_per_neighborhood_per_capita > median_rent_stats_per_neighborhood_per_capita)
- **#underserved**: Below 50% of median rent_stats_per_neighborhood per capita (rent_stats_per_neighborhood_per_capita < 0.5 * median_rent_stats_per_neighborhood_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.rent_stats_per_neighborhood` joined with regional statistics
2. Calculates key metrics:
      - rent_stats_per_neighborhood_per_10k_residents

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
from segmentation.Rent Stats Per Neighborhood.data_loader import RentStatsPerNeighborhoodDataLoader
loader = RentStatsPerNeighborhoodDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Rent Stats Per Neighborhood.rule_based_segmenter import RentStatsPerNeighborhoodRuleBasedSegmenter
rule_segmenter = RentStatsPerNeighborhoodRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Rent Stats Per Neighborhood.llm_segmenter import RentStatsPerNeighborhoodLlmSegmenter
llm_segmenter = RentStatsPerNeighborhoodLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Rent Stats Per Neighborhood.react_segmenter import RentStatsPerNeighborhoodReactSegmenter
react_segmenter = RentStatsPerNeighborhoodReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Rent Stats Per Neighborhood (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
