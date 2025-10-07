# Districts Pop Stat Segmentation Module

## 🏷️ Districts Pop Stat Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Districts Pop Stat characteristics:

- **#well-served**: Above median number of districts_pop_stat per capita (districts_pop_stat_per_capita > median_districts_pop_stat_per_capita)
- **#underserved**: Below 50% of median districts_pop_stat per capita (districts_pop_stat_per_capita < 0.5 * median_districts_pop_stat_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.districts_pop_stat` joined with regional statistics
2. Calculates key metrics:
      - districts_pop_stat_per_10k_residents

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
from segmentation.Districts Pop Stat.data_loader import DistrictsPopStatDataLoader
loader = DistrictsPopStatDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Districts Pop Stat.rule_based_segmenter import DistrictsPopStatRuleBasedSegmenter
rule_segmenter = DistrictsPopStatRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Districts Pop Stat.llm_segmenter import DistrictsPopStatLlmSegmenter
llm_segmenter = DistrictsPopStatLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Districts Pop Stat.react_segmenter import DistrictsPopStatReactSegmenter
react_segmenter = DistrictsPopStatReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Districts Pop Stat (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
