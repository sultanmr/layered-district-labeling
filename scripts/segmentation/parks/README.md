# Parks Segmentation Module

## 🏷️ Parks Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Parks characteristics:

- **#well-maintained**: Above median maintenance score (maintenance_score > median_maintenance)
- **#needs-attention**: Below 70% of median maintenance score (maintenance_score < 0.7 * median_maintenance)
- **#large-park**: Above median average park size (avg_park_size > median_park_size)
- **#spacious**: Above median green space per capita (green_space_per_capita > median_per_capita)
- **#crowded**: Below 50% of median green space per capita (green_space_per_capita < 0.5 * median_per_capita)
- **#many-parks**: Above median number of green spaces (num_green_spaces > median_num_spaces)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.parks` joined with regional statistics
2. Calculates key metrics:
      - park_area_per_10k_residents

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
from segmentation.Parks.data_loader import ParksDataLoader
loader = ParksDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Parks.rule_based_segmenter import ParksRuleBasedSegmenter
rule_segmenter = ParksRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Parks.llm_segmenter import ParksLlmSegmenter
llm_segmenter = ParksLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Parks.react_segmenter import ParksReactSegmenter
react_segmenter = ParksReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Parks (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
