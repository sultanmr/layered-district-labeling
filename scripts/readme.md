# Berlin Urban Analytics Project

## Project Structure
```
.
├── segmentation/          # Neighborhood analysis modules
│   ├── banks/            # Banking service analysis
│   ├── bus_tram_stops/   # Public transport accessibility
│   ├── crime_statistics/ # Crime pattern analysis
│   ├── dental_offices/   # Dental care service analysis
│   ├── districts_pop_stat/ # Population statistics analysis
│   ├── hospitals/        # Healthcare facility analysis
│   ├── kindergartens/    # Early education facility analysis
│   ├── land_prices/      # Real estate market analysis
│   ├── long_term_listings/ # Long-term rental market analysis
│   ├── milieuschutz_protection_zones/ # Heritage protection analysis
│   ├── parks/            # Green space analysis
│   ├── playgrounds/      # Recreational space analysis
│   ├── pools/            # Swimming facility analysis
│   ├── rent_stats_per_neighborhood/ # Rental market analysis
│   ├── sbahn/            # S-Bahn transport analysis
│   ├── schools/          # Educational facility analysis
│   ├── short_term_listings/ # Short-term rental analysis
│   ├── ubahn/            # U-Bahn transport analysis
│   ├── universities/     # Higher education analysis
│   ├── venues/           # Entertainment venue analysis
│   ├── README.md         # Architecture documentation
│   └── orchestrator.py   # Coordination logic

├── README.md              # This file
├── requirements.txt       # Dependencies
├── schema.txt            # Database schema definitions
└── viz/                  # Visualization outputs
```

## Key Features
- Modular neighborhood segmentation with standardized interfaces
- Multiple analysis approaches: rule-based, LLM-powered, and ReAct agents
- Comprehensive documentation with detailed schemas for each module
- Standardized SQL queries and data loading patterns
- Unit and integration test coverage
- Caching mechanisms for efficient data processing

## Standardized Module Structure
Each segmentation module follows the same consistent pattern:

### Core Components:
- **Data Loaders**: Fetch and preprocess data from `berlin_source_data` tables
- **Feature Processors**: Calculate key metrics (per-capita calculations, quality scores)
- **Segmenters**: Three approaches for neighborhood categorization:
  - Rule-based segmentation with configurable thresholds
  - LLM-powered analysis using Gemini
  - ReAct agent reasoning with dynamic tool-based analysis

### Common Features:
- Per-capita calculations (per 10k residents)
- Median-based thresholding for consistent categorization
- Graceful handling of missing data and edge cases
- COALESCE usage to avoid division by zero
- Population-adjusted metrics

## Available Analysis Modules

### Transportation & Infrastructure
- **bus_tram_stops**: Public transport accessibility and coverage analysis
- **sbahn**: S-Bahn network accessibility analysis
- **ubahn**: U-Bahn network accessibility analysis

### Healthcare & Social Services
- **hospitals**: Healthcare facility availability and quality
- **dental_offices**: Dental care service accessibility
- **kindergartens**: Early childhood education facilities

### Education & Culture
- **schools**: Primary and secondary education facilities
- **universities**: Higher education institution analysis
- **venues**: Entertainment and cultural venue analysis

### Real Estate & Housing
- **land_prices**: Property value and real estate market analysis
- **long_term_listings**: Long-term rental market analysis
- **short_term_listings**: Short-term rental market analysis
- **rent_stats_per_neighborhood**: Comprehensive rental statistics

### Public Amenities & Recreation
- **parks**: Green space and park accessibility
- **playgrounds**: Children's recreational facilities
- **pools**: Swimming and aquatic facilities

### Safety & Community
- **crime_statistics**: Crime pattern and safety assessment
- **milieuschutz_protection_zones**: Heritage and neighborhood protection
- **districts_pop_stat**: Population demographic analysis

### Financial Services
- **banks**: Banking service accessibility and ATM availability

## Label Categories Overview

Each module generates descriptive hashtags for districts based on their characteristics:

### Common Tags Across Modules:
- **#well-served**: Above median service availability
- **#underserved**: Below 50% of median service availability
- **#transport-desert**: Limited public transport access
- **#good-coverage**: Above median service coverage
- **#high-quality-service**: Above median service quality

### Module-Specific Tags:
- **banks**: #well-banked, #banking-desert, #good-service, #accessible-banking, #atm-rich, #atm-poor
- **bus_tram_stops**: #limited-coverage, #high-quality-service

## Usage Examples

### Individual Module Usage
```python
# For any module (example with banks)
from segmentation.banks.data_loader import BanksDataLoader
from segmentation.banks.rule_based_segmenter import BanksRuleBasedSegmenter
from segmentation.banks.llm_segmenter import BanksLlmSegmenter
from segmentation.banks.react_segmenter import BanksReactSegmenter

loader = BanksDataLoader()
rule_segmenter = BanksRuleBasedSegmenter(threshold_multiplier=1.0)
llm_segmenter = BanksLlmSegmenter()
react_segmenter = BanksReactSegmenter()

features_df = loader.load_data()
rule_tags = rule_segmenter.segment(features_df)
llm_tags = llm_segmenter.segment(features_df)
react_tags = react_segmenter.segment(features_df)
```

### Orchestrated Analysis
```python
from segmentation.orchestrator import analyze_all

# Analyze all available urban data modules
results = analyze_all(engine)
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Configure database connection with environment variables
3. Set up API keys for LLM services (Gemini, OpenAI, DeepSeek)
4. Run analyses through orchestrator or individual modules

## Documentation
Each module contains detailed documentation:
- `README.md` - Implementation details, usage examples, and analysis methodology
- `schema.md` - Data schema, derived features, and statistical analysis approaches
- `schemas.py` - Pydantic models for tag definitions and validation rules
- `sql_queries.py` - Standardized SQL queries for data access

See individual module directories for comprehensive implementation details and API documentation.

## Edge Case Handling
All modules gracefully handle:
- Districts with no relevant facilities (returns empty tags)
- Varying population sizes in per-capita calculations
- Division by zero using COALESCE in SQL queries
- Missing data in feature calculations