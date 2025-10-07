# Schools Segmentation Module

## 🏷️ Schools Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Schools characteristics:

- **#well-served**: Above median number of schools per capita (schools_per_capita > median_schools_per_capita)
- **#underserved**: Below 50% of median schools per capita (schools_per_capita < 0.5 * median_schools_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.schools` joined with regional statistics
2. Calculates key metrics:
      - schools_per_10k_residents

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
from segmentation.Schools.data_loader import SchoolsDataLoader
loader = SchoolsDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Schools.rule_based_segmenter import SchoolsRuleBasedSegmenter
rule_segmenter = SchoolsRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Schools.llm_segmenter import SchoolsLlmSegmenter
llm_segmenter = SchoolsLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Schools.react_segmenter import SchoolsReactSegmenter
react_segmenter = SchoolsReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Schools (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
