# Sbahn Segmentation Module

## 🏷️ Sbahn Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Sbahn characteristics:

- **#well-served**: Above median number of sbahn per capita (sbahn_per_capita > median_sbahn_per_capita)
- **#underserved**: Below 50% of median sbahn per capita (sbahn_per_capita < 0.5 * median_sbahn_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.sbahn` joined with regional statistics
2. Calculates key metrics:
      - sbahn_per_10k_residents

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
from segmentation.Sbahn.data_loader import SbahnDataLoader
loader = SbahnDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Sbahn.rule_based_segmenter import SbahnRuleBasedSegmenter
rule_segmenter = SbahnRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Sbahn.llm_segmenter import SbahnLlmSegmenter
llm_segmenter = SbahnLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Sbahn.react_segmenter import SbahnReactSegmenter
react_segmenter = SbahnReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Sbahn (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
