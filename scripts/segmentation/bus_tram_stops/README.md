# Bus Tram Stops Segmentation Module

## 🏷️ Bus Tram Stops Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Bus Tram Stops characteristics:

- **#well-served**: Above median number of stops per capita (stops_per_capita > median_stops_per_capita)
- **#transport-desert**: Below 50% of median stops per capita (stops_per_capita < 0.5 * median_stops_per_capita)
- **#good-coverage**: Above median coverage score (coverage_score > median_coverage)
- **#limited-coverage**: Below 50% of median coverage score (coverage_score < 0.5 * median_coverage)
- **#high-quality-service**: Above median service quality score (service_quality_score > median_service_quality)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.bus_tram_stops` joined with regional statistics
2. Calculates key metrics:
      - stops_per_10k_residents

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
from segmentation.Bus Tram Stops.data_loader import BusTramStopsDataLoader
loader = BusTramStopsDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Bus Tram Stops.rule_based_segmenter import BusTramStopsRuleBasedSegmenter
rule_segmenter = BusTramStopsRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Bus Tram Stops.llm_segmenter import BusTramStopsLlmSegmenter
llm_segmenter = BusTramStopsLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Bus Tram Stops.react_segmenter import BusTramStopsReactSegmenter
react_segmenter = BusTramStopsReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Bus Tram Stops (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
