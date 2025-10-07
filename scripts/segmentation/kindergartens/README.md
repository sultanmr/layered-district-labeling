# Kindergartens Segmentation Module

## 🏷️ Kindergartens Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Kindergartens characteristics:

- **#well-served**: Above median number of kindergartens per capita (kindergartens_per_capita > median_kindergartens_per_capita)
- **#underserved**: Below 50% of median kindergartens per capita (kindergartens_per_capita < 0.5 * median_kindergartens_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.kindergartens` joined with regional statistics
2. Calculates key metrics:
      - kindergartens_per_10k_residents

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
from segmentation.Kindergartens.data_loader import KindergartensDataLoader
loader = KindergartensDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Kindergartens.rule_based_segmenter import KindergartensRuleBasedSegmenter
rule_segmenter = KindergartensRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Kindergartens.llm_segmenter import KindergartensLlmSegmenter
llm_segmenter = KindergartensLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Kindergartens.react_segmenter import KindergartensReactSegmenter
react_segmenter = KindergartensReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Kindergartens (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
