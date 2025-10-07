# Universities Segmentation Module

## 🏷️ Universities Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Universities characteristics:

- **#well-served**: Above median number of universities per capita (universities_per_capita > median_universities_per_capita)
- **#underserved**: Below 50% of median universities per capita (universities_per_capita < 0.5 * median_universities_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.universities` joined with regional statistics
2. Calculates key metrics:
      - universities_per_10k_residents

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
from segmentation.Universities.data_loader import UniversitiesDataLoader
loader = UniversitiesDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Universities.rule_based_segmenter import UniversitiesRuleBasedSegmenter
rule_segmenter = UniversitiesRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Universities.llm_segmenter import UniversitiesLlmSegmenter
llm_segmenter = UniversitiesLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Universities.react_segmenter import UniversitiesReactSegmenter
react_segmenter = UniversitiesReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Universities (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
