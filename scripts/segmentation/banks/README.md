# Banks Segmentation Module

## 🏷️ Banks Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Banks characteristics:

- **#well-banked**: Above median number of banks per capita (banks_per_capita > median_banks_per_capita)
- **#banking-desert**: Below 50% of median banks per capita (banks_per_capita < 0.5 * median_banks_per_capita)
- **#good-service**: Above median service availability score (service_availability_score > median_service_availability)
- **#accessible-banking**: Above median accessibility score (accessibility_score > median_accessibility)
- **#atm-rich**: Above median ATMs per capita (atms_per_capita > median_atms_per_capita)
- **#atm-poor**: Below 50% of median ATMs per capita (atms_per_capita < 0.5 * median_atms_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.banks` joined with regional statistics
2. Calculates key metrics:
      - banks_per_10k_residents
   - atms_per_10k_residents

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
from segmentation.Banks.data_loader import BanksDataLoader
loader = BanksDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Banks.rule_based_segmenter import BanksRuleBasedSegmenter
rule_segmenter = BanksRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Banks.llm_segmenter import BanksLlmSegmenter
llm_segmenter = BanksLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Banks.react_segmenter import BanksReactSegmenter
react_segmenter = BanksReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Banks (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
