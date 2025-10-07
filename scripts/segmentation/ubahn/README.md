# Ubahn Segmentation Module

## 🏷️ Ubahn Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Ubahn characteristics:

- **#well-served**: Above median number of ubahn per capita (ubahn_per_capita > median_ubahn_per_capita)
- **#underserved**: Below 50% of median ubahn per capita (ubahn_per_capita < 0.5 * median_ubahn_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.ubahn` joined with regional statistics
2. Calculates key metrics:
      - ubahn_per_10k_residents

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
from segmentation.Ubahn.data_loader import UbahnDataLoader
loader = UbahnDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Ubahn.rule_based_segmenter import UbahnRuleBasedSegmenter
rule_segmenter = UbahnRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Ubahn.llm_segmenter import UbahnLlmSegmenter
llm_segmenter = UbahnLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Ubahn.react_segmenter import UbahnReactSegmenter
react_segmenter = UbahnReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Ubahn (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
