# Milieuschutz Protection Zones Segmentation Module

## 🏷️ Milieuschutz Protection Zones Labels
### 🔍 Label Categories
The module generates the following tags for districts based on their Milieuschutz Protection Zones characteristics:

- **#well-served**: Above median number of milieuschutz_protection_zones per capita (milieuschutz_protection_zones_per_capita > median_milieuschutz_protection_zones_per_capita)
- **#underserved**: Below 50% of median milieuschutz_protection_zones per capita (milieuschutz_protection_zones_per_capita < 0.5 * median_milieuschutz_protection_zones_per_capita)

## 🛠 Implementation Details
### Data Processing
1. Pulls from `berlin_source_data.milieuschutz_protection_zones` joined with regional statistics
2. Calculates key metrics:
      - milieuschutz_protection_zones_per_10k_residents

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
from segmentation.Milieuschutz Protection Zones.data_loader import MilieuschutzProtectionZonesDataLoader
loader = MilieuschutzProtectionZonesDataLoader()
features_df = loader.load_data()

# For rule-based segmentation
from segmentation.Milieuschutz Protection Zones.rule_based_segmenter import MilieuschutzProtectionZonesRuleBasedSegmenter
rule_segmenter = MilieuschutzProtectionZonesRuleBasedSegmenter(threshold_multiplier=1.0)  # Adjust sensitivity
rule_tags = rule_segmenter.segment(features_df)

# For LLM-based segmentation
from segmentation.Milieuschutz Protection Zones.llm_segmenter import MilieuschutzProtectionZonesLlmSegmenter
llm_segmenter = MilieuschutzProtectionZonesLlmSegmenter()
llm_tags = llm_segmenter.segment(features_df)

# For ReAct agent segmentation
from segmentation.Milieuschutz Protection Zones.react_segmenter import MilieuschutzProtectionZonesReactSegmenter
react_segmenter = MilieuschutzProtectionZonesReactSegmenter()
react_tags = react_segmenter.segment(features_df)
```

## ⚠️ Edge Cases
- Handles districts with no Milieuschutz Protection Zones (returns empty tags)
- Adjusts for varying population sizes in per-capita calculations
- Uses COALESCE to avoid division by zero in population calculations
- Gracefully handles missing data in feature calculations
