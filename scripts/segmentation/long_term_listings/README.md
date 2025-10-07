# Long Term Listings Segmentation Module

## 🏷️ Long Term Listings Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Long Term Listings characteristics:

- **#well-served**: Above median number of long_term_listings per capita (long_term_listings_per_capita > median_long_term_listings_per_capita)
- **#underserved**: Below 50% of median long_term_listings per capita (long_term_listings_per_capita < 0.5 * median_long_term_listings_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.long_term_listings` joined with regional statistics
2. Calculates key metrics:
      - long_term_listings_per_10k_residents

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
from segmentation.Long Term Listings.data_loader import LongTermListingsDataLoader
loader = LongTermListingsDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Long Term Listings.rule_based_segmenter import LongTermListingsRuleBasedSegmenter
rule_segmenter = LongTermListingsRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Long Term Listings.llm_segmenter import LongTermListingsLlmSegmenter
llm_segmenter = LongTermListingsLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Long Term Listings.react_segmenter import LongTermListingsReactSegmenter
react_segmenter = LongTermListingsReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Long Term Listings (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
