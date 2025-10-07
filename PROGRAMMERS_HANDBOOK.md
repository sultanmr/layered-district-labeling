# 🏗️ Segmentation System Programmer's Handbook

## 📋 Table of Contents
- [🏗️ Segmentation System Programmer's Handbook](#️-segmentation-system-programmers-handbook)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [🏛️ System Architecture](#️-system-architecture)
    - [Architecture Diagram](#architecture-diagram)
    - [Component Relationships](#component-relationships)
  - [🔧 7-Step Implementation Process](#-7-step-implementation-process)
    - [Step 1: Create Directory Structure](#step-1-create-directory-structure)
    - [Step 2: Define Database Schema](#step-2-define-database-schema)
    - [Step 3: Implement Data Loader](#step-3-implement-data-loader)
    - [Step 4: Create Feature Processor](#step-4-create-feature-processor)
    - [Step 5: Implement Segmentation Strategies](#step-5-implement-segmentation-strategies)
    - [Step 6: Configure Module Exports](#step-6-configure-module-exports)
    - [Step 7: Update Configuration](#step-7-update-configuration)
  - [📊 Example: Crime Statistics Implementation](#-example-crime-statistics-implementation)
    - [Directory Structure](#directory-structure)
    - [crime_statistics/\_\_init\_\_.py](#crime_statistics__init__py)
    - [crime_statistics/data_loader.py](#crime_statisticsdata_loaderpy)
    - [crime_statistics/feature_processor.py](#crime_statisticsfeature_processorpy)
    - [crime_statistics/rule_based_segmenter.py](#crime_statisticsrule_based_segmenterpy)
    - [crime_statistics/schema.md](#crime_statisticsschemamd)
  - [⚙️ Configuration Management](#️-configuration-management)
    - [segment_config.json Structure](#segment_configjson-structure)
    - [Environment Variables](#environment-variables)
  - [🧪 Testing Guidelines](#-testing-guidelines)
    - [Unit Testing](#unit-testing)
    - [Integration Testing](#integration-testing)
  - [🚀 Deployment Pipeline](#-deployment-pipeline)
    - [Pipeline Execution Flow](#pipeline-execution-flow)
  - [🔍 Troubleshooting](#-troubleshooting)
    - [Common Issues](#common-issues)
    - [Debugging Tips](#debugging-tips)
  - [📈 Best Practices](#-best-practices)
    - [Code Organization](#code-organization)
    - [Performance Optimization](#performance-optimization)
    - [Error Handling](#error-handling)
  - [🎨 Visualization Integration](#-visualization-integration)
    - [Heatmap Generation](#heatmap-generation)
    - [Network Graphs](#network-graphs)
  - [🔮 Future Enhancements](#-future-enhancements)

## 🎯 Overview

This handbook provides a comprehensive guide for adding new segmentation components to the neighborhood analysis system. The system follows a modular, extensible architecture inspired by the `green_spaces` implementation pattern.

## 🏛️ System Architecture

### Architecture Diagram

```mermaid
graph TB
    subgraph "Input Layer"
        DB[(Database)]
        Config[Configuration]
        Env[Environment Variables]
    end

    subgraph "Core Processing"
        Orchestrator[Orchestrator]
        
        subgraph "Component Modules"
            DataLoader[Data Loader]
            FeatureProcessor[Feature Processor]
            
            subgraph "Segmentation Strategies"
                RuleBased[Rule-Based]
                ML[Machine Learning]
                RAG[RAG]
                LLM[LLM-Powered]
                ReAct[ReAct Agent]
            end
        end
    end

    subgraph "Output Layer"
        Results[Results Aggregator]
        Visualizations[Visualizations]
        DB_Output[(Output Database)]
        Cache[(Cache Storage)]
    end

    DB --> DataLoader
    Config --> Orchestrator
    Env --> Orchestrator
    Orchestrator --> DataLoader
    DataLoader --> FeatureProcessor
    FeatureProcessor --> RuleBased
    FeatureProcessor --> ML
    FeatureProcessor --> RAG
    FeatureProcessor --> LLM
    FeatureProcessor --> ReAct
    RuleBased --> Results
    ML --> Results
    RAG --> Results
    LLM --> Results
    ReAct --> Results
    Results --> Visualizations
    Results --> DB_Output
    DataLoader -.-> Cache
    Cache -.-> DataLoader
```

### Component Relationships

| Component | Responsibility | Interface | Key Features |
|-----------|----------------|-----------|-------------|
| **DataLoader** | Loads raw data from database with intelligent caching | [`base.DataLoader`](scripts/segmentation/base.py:16) | Disk caching, timeout management, query optimization |
| **FeatureProcessor** | Processes raw data into features | [`base.FeatureProcessor`](scripts/segmentation/base.py:154) | Data transformation, normalization, feature engineering |
| **SegmentationStrategy** | Base interface for segmentation approaches | [`base.SegmentationStrategy`](scripts/segmentation/base.py:162) | Abstract base class for all strategies |
| **LLMSegmentationStrategy** | Base for LLM-powered segmentation | [`base.LLMSegmentationStrategy`](scripts/segmentation/base.py:257) | Multi-provider support (Gemini, OpenAI, DeepSeek) |
| **ReActSegmentationStrategy** | Base for ReAct agent segmentation | [`react_base.ReactSegmentationStrategy`](scripts/segmentation/react_base.py:18) | Tool-based reasoning, dynamic analysis |
| **Orchestrator** | Coordinates pipeline execution | [`orchestrator.SegmentationOrchestrator`](scripts/segmentation/orchestrator.py:13) | Configuration loading, result aggregation, visualization |
| **ResultAggregator** | Combines results from strategies | [`base.ResultAggregator`](scripts/segmentation/base.py:186) | Multi-strategy result merging, DataFrame conversion |

## 🔧 8-Step Implementation Process

### Step 1: Create Directory Structure with Enhanced Patterns

```bash
segmentation/
└── your_component/          # snake_case component name
    ├── __init__.py          # Module exports (required)
    ├── data_loader.py       # Data loading with caching (required)
    ├── feature_processor.py # Feature processing (required)
    ├── rule_based_segmenter.py # Rule-based segmentation (recommended)
    ├── ml_segmenter.py      # Machine learning segmentation (optional)
    ├── llm_segmenter.py     # LLM-powered segmentation (optional)
    ├── react_segmenter.py   # ReAct agent segmentation (optional)
    ├── rag_segmenter.py     # RAG-based segmentation (optional)
    ├── sql_queries.py       # SQL queries (required)
    ├── schemas.py           # Pydantic tag schemas (recommended)
    ├── schema.md           # Database schema documentation (required)
    ├── README.md           # Component documentation (required)
    └── rag_docs.json       # RAG knowledge base (optional)
```

### Step 2: Define Database Schema and Tag Schemas

Create [`schema.md`](scripts/segmentation/green_spaces/schema.md:1) with database documentation, and [`schemas.py`](scripts/segmentation/green_spaces/schemas.py:1) with Pydantic tag definitions:

```markdown
# Your Component Schema Documentation

## Database Tables
### `your_table`
- `id`: Primary key
- `neighborhood_id`: Foreign key to neighborhoods
- `metric_1`: Description with units
- `metric_2`: Description with units
- `year`: Data collection year
- `source`: Data source information
```

## Data Relationships
```mermaid
erDiagram
    YOUR_TABLE ||--o{ NEIGHBORHOODS : "located_in"
    YOUR_TABLE }|--|| DATA_SOURCES : "sourced_from"
```

## Field Descriptions
| Field | Type | Description | Example | Constraints |
|-------|------|-------------|---------|-------------|
| metric_1 | numeric | Description with units | 42.5 | NOT NULL, >= 0 |
| metric_2 | numeric | Description with units | 15.2 | NOT NULL, >= 0 |
```

Create [`schemas.py`](scripts/segmentation/green_spaces/schemas.py:1) with Pydantic tag definitions:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class TagCategory(str, Enum):
    """Categories for tag organization"""
    QUALITY = "quality"
    QUANTITY = "quantity"
    TREND = "trend"
    COMPARISON = "comparison"

class YourComponentTag(BaseModel):
    """Pydantic model for component tags with validation rules"""
    tag: str = Field(..., description="The hashtag name")
    category: TagCategory = Field(..., description="Category of the tag")
    description: str = Field(..., description="Description of what the tag represents")
    rule: str = Field(..., description="The rule that triggers this tag")
    
    @classmethod
    def get_all_tags(cls):
        """Return all predefined tags with their rules"""
        return [
            YourComponentTag(
                tag="#high-metric",
                category=TagCategory.QUALITY,
                description="Above median metric value",
                rule="metric_1 > median_metric"
            ),
            # Add more tags...
        ]
```


### Step 3: Implement Data Loader with Caching

Create [`data_loader.py`](scripts/segmentation/green_spaces/data_loader.py:1) with intelligent caching:

```python
from ..base import DataLoader
from .sql_queries import YOUR_COMPONENT_QUERY

class YourComponentDataLoader(DataLoader):
    """Data loader for your component with intelligent caching"""
    
    @property
    def query(self) -> str:
        return YOUR_COMPONENT_QUERY
    
    def load_data(self, engine) -> pd.DataFrame:
        """Load data with caching - automatically uses parent class caching"""
        # The base DataLoader handles caching automatically
        # Cache timeout is configurable via TIME_OUT environment variable
        # Default timeout: 12 hours (43200 seconds)
        return super().load_data(engine)
```

Create [`sql_queries.py`](scripts/segmentation/green_spaces/sql_queries.py:1) with comprehensive queries:

```python
"""SQL queries for your component segmentation with proper joins and filtering"""

YOUR_COMPONENT_QUERY = """
SELECT
    t.neighborhood,
    COUNT(t.id) AS total_items,
    AVG(t.metric_1) AS avg_metric,
    SUM(t.metric_2) AS total_metric,
    -- Include additional calculated metrics
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.metric_1) AS median_metric,
    STDDEV(t.metric_1) AS metric_stddev,
    -- Join with neighborhoods for additional context if needed
    n.population,
    n.area_sq_km
FROM your_database.your_table t
LEFT JOIN test_berlin_data.neighborhoods n ON t.neighborhood = n.name
WHERE t.year = EXTRACT(YEAR FROM CURRENT_DATE) - 1  -- Most recent complete year
  AND t.metric_1 IS NOT NULL  -- Exclude null values
  AND t.metric_2 > 0  -- Positive values only
GROUP BY t.neighborhood, n.population, n.area_sq_km
ORDER BY t.neighborhood
"""

# Additional queries for specific use cases
YOUR_COMPONENT_DETAILED_QUERY = """
SELECT * FROM your_database.your_table
WHERE neighborhood = %s
ORDER BY year DESC, metric_1 DESC
"""
```

### Step 4: Create Feature Processor with Enhanced Processing

Create [`feature_processor.py`](scripts/segmentation/green_spaces/feature_processor.py:1) with comprehensive feature engineering:

```python
from typing import Dict, List
import pandas as pd
import numpy as np
from ..base import FeatureProcessor

class YourComponentFeatureProcessor(FeatureProcessor):
    """Processes your component features with comprehensive transformations"""
    
    def process_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        features = raw_data.copy()
        
        # Add your feature calculations with robust error handling
        features['normalized_metric'] = (
            features['total_metric'] /
            features['total_items'].replace(0, 1)
        )
        
        # Additional feature engineering
        features['log_metric'] = np.log1p(features['total_metric'])
        features['metric_per_capita'] = (
            features['total_metric'] /
            (features['population'] / 1000).replace(0, 1)
        )
        
        # Handle missing data with appropriate strategies
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        # Remove outliers using IQR method
        features = self._remove_outliers(features, ['normalized_metric', 'metric_per_capita'])
        
        return features
    
    def _remove_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove outliers using IQR method for specified columns"""
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean
```

### Step 5: Implement Enhanced Segmentation Strategies

#### Rule-Based Segmenter ([`rule_based_segmenter.py`](scripts/segmentation/green_spaces/rule_based_segmenter.py:1))

```python
from typing import Dict, List
import pandas as pd
from ..base import SegmentationStrategy

class YourComponentRuleBasedSegmenter(SegmentationStrategy):
    """Rule-based segmentation for your component"""
    
    def __init__(self, threshold_multiplier: float = 1.0):
        self.threshold_multiplier = threshold_multiplier
        
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        # Calculate medians
        median_metric = features['normalized_metric'].median()
        
        # Apply rules
        features['#high-metric'] = features['normalized_metric'] > median_metric
        features['#low-metric'] = features['normalized_metric'] < median_metric * 0.5
        
        # Convert to result dictionary
        result_dict = {}
        for _, row in features.iterrows():
            tags = []
            if row['#high-metric']:
                tags.append('#high-metric')
            if row['#low-metric']:
                tags.append('#low-metric')
            result_dict[row['neighborhood']] = tags
            
        return result_dict
```

#### ML Segmenter ([`ml_segmenter.py`](scripts/segmentation/green_spaces/ml_segmenter.py:1))

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ..base import SegmentationStrategy

class YourComponentMlSegmenter(SegmentationStrategy):
    """ML-based segmentation for your component"""
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        # Select and scale features
        numeric_features = features[['normalized_metric', 'total_items']]
        scaled_features = self.scaler.fit_transform(numeric_features)
        
        # Cluster and generate tags
        clusters = self.model.fit_predict(scaled_features)
        return self._generate_tags(features, clusters)
    
    def _generate_tags(self, features, clusters):
        # Implement your tag generation logic
        neighborhood_tags = {}
        # ... tag generation code
        return neighborhood_tags
```

#### LLM-Powered Segmenter ([`llm_segmenter.py`](scripts/segmentation/green_spaces/llm_segmenter.py:1))

```python
import pandas as pd
from ..base import LLMSegmentationStrategy

class YourComponentLlmSegmenter(LLMSegmentationStrategy):
    """LLM-powered segmentation for your component with multi-provider support"""
    
    def __init__(self, llm_provider="gemini"):
        super().__init__(init_gemini=False)  # Don't auto-init, we'll handle it
        self.llm_provider = llm_provider
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM based on provider selection"""
        if self.llm_provider == "gemini":
            self.llm = self._initialize_gemini_llm()
        elif self.llm_provider == "openai":
            self.llm = self._create_langchain_openai()
        elif self.llm_provider == "deepseek":
            self.llm = self._create_langchain_deepseek()
    
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment using actual neighborhood data values with LLM analysis"""
        neighborhood_tags = {}
        
        # Generate LLM-powered tags for each neighborhood based on its actual data
        for _, row in features.iterrows():
            neighborhood = row['neighborhood']
            # Extract the numeric features for this neighborhood
            summary = {
                'normalized_metric': row.get('normalized_metric', 0),
                'total_items': row.get('total_items', 0),
                'metric_per_capita': row.get('metric_per_capita', 0)
            }
            
            prompt = self._create_prompt(summary, neighborhood)
            llm_tags = self._get_tags(prompt)
            neighborhood_tags[neighborhood] = llm_tags

        return neighborhood_tags

    def _create_prompt(self, neighborhood_summary: dict, neighborhood: str) -> str:
        """Generate a prompt for LLM with neighborhood statistics"""
        return f"""
        Analyze these statistics for neighborhood '{neighborhood}':

        Neighborhood Profile:
        - Normalized metric: {neighborhood_summary['normalized_metric']:.2f}
        - Total items: {neighborhood_summary['total_items']:.2f}
        - Metric per capita: {neighborhood_summary['metric_per_capita']:.2f}

        Suggest 3-6 hyphenated tags (e.g., 'high-metric', 'well-served') that:
        1. Reflect the quantitative characteristics
        2. Are descriptive and meaningful
        3. Are specific to this neighborhood's profile

        Return ONLY comma-separated tags, nothing else:
        """
```

#### RAG Segmenter ([`rag_segmenter.py`](segmentation/green_spaces/rag_segmenter.py:1))

```python
import pandas as pd
from ..base import SegmentationStrategy
from ..retriever import VectorRetriever
import json
import os

class YourComponentRagSegmenter(SegmentationStrategy):
    """RAG-based segmentation for your component"""
    
    def __init__(self):
        self.retriever = VectorRetriever()
        rag_file = os.path.join(os.path.dirname(__file__), 'rag_docs.json')
        if os.path.isfile(rag_file):
            with open(rag_file, 'r') as f:
                rag_docs = json.load(f)
                self.retriever.add_documents(rag_docs)
                
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        results = {}
        for _, row in features.iterrows():
            query = f"""
            Component characteristics:
            - Normalized metric: {row['normalized_metric']:.2f}
            - Total items: {row['total_items']:.2f}
            """
            context_docs = self.retriever.retrieve(query)
            tags = set()
            for doc in context_docs:
                tags.update(doc.get('tags', []))
            results[row['neighborhood']] = list(tags)
            
        return results
```

#### ReAct Agent Segmenter ([`react_segmenter.py`](scripts/segmentation/green_spaces/react_segmenter.py:1))

```python
from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from langchain.tools import Tool
from ..react_base import ReactSegmentationStrategy
from .schemas import YourComponentTag

class YourComponentReactSegmenter(ReactSegmentationStrategy):
    """ReAct agent for your component segmentation with reasoning capabilities"""
    
    def __init__(self, llm_provider="gemini"):
        tools = self._create_all_tools()
        cols = ['normalized_metric', 'total_items', 'metric_per_capita']
        super().__init__(tools, cols, llm_provider)
    
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment neighborhoods using ReAct agent reasoning"""
        self.features_df = features.copy()
        
        try:
            analysis_prompt = self._create_analysis_prompt(
                "your component",
                features,
                YourComponentTag.get_all_tags()
            )
            return self._get_results(analysis_prompt, features)
        except Exception as e:
            logging.error(f"ReAct agent error: {e}")
            return self._get_fallback_tags(features)
    
    def _create_tools(self):
        """Create component-specific analysis tools"""
        return [
            Tool(
                name="analyze_metric_patterns",
                func=self._analyze_metric_patterns_wrapper,
                description="Analyze metric patterns and categorize levels using dynamic thresholds"
            ),
            # Add more component-specific tools...
        ]
    
    def _analyze_metric_patterns_wrapper(self, column_name: str) -> dict:
        """Wrapper for metric pattern analysis"""
        clean_column_name = column_name.strip().strip("'\"")
        if hasattr(self, 'features_df') and clean_column_name in self.features_df.columns:
            return self._analyze_metric_patterns(self.features_df[clean_column_name])
        return {"error": f"Column '{clean_column_name}' not found"}
    
    def _analyze_metric_patterns(self, data: pd.Series) -> dict:
        """Analyze metric patterns with dynamic thresholds"""
        stats = self._calculate_statistics(data)
        distribution = self._analyze_distribution(data)
        
        # Dynamic threshold calculation
        q1, q3 = stats["q1"], stats["q3"]
        iqr = stats["iqr"]
        
        if stats["median"] > q3 + iqr * 0.5:
            level = "very_high"
        elif stats["median"] > q3:
            level = "high"
        elif stats["median"] > q1:
            level = "medium"
        else:
            level = "low"
            
        return {
            "level": level,
            "statistics": stats,
            "distribution_analysis": distribution
        }
```

### Step 6: Configure Module Exports

Create [`__init__.py`](segmentation/green_spaces/__init__.py:1):

```python
from .data_loader import YourComponentDataLoader
from .feature_processor import YourComponentFeatureProcessor
from .rule_based_segmenter import YourComponentRuleBasedSegmenter
from .ml_segmenter import YourComponentMlSegmenter
from .gemini_segmenter import YourComponentGeminiSegmenter
from .rag_segmenter import YourComponentRagSegmenter

__all__ = [
    'YourComponentDataLoader',
    'YourComponentFeatureProcessor',
    'YourComponentRuleBasedSegmenter',
    'YourComponentMlSegmenter', 
    'YourComponentGeminiSegmenter',
    'YourComponentRagSegmenter'
]
```

### Step 7: Update Configuration and Environment

Update [`config/segment_config.json`](scripts/config/segment_config.json:1) with the new format:

```json
{
  "tables": {
    "green_spaces": ["rule_based", "llm", "react"],
    "rent_stats": ["rule_based", "react"],
    "land_prices": ["rule_based", "react"],
    "short_time_listings": ["rule_based"],
    "your_component": ["rule_based", "llm", "react", "rag"]
  }
}
```

### Step 8: Configure Environment Variables

Update your environment variables with the new requirements:

```bash
# Required Environment Variables
DB_URL=postgresql://user:password@host:port/database

# LLM API Keys (choose one or multiple)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional Configuration
SEGMENT_CONFIG=config/segment_config.json
VISUALIZATION_PATH=viz/
CACHE_PATH=cache/data_loader_cache.json
TIME_OUT=43200  # Cache timeout in seconds (12 hours)

# Logging Configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## 📊 Example Implementations

### Crime Statistics Implementation

#### Directory Structure
```
segmentation/crime_statistics/
├── __init__.py
├── data_loader.py
├── feature_processor.py
├── rule_based_segmenter.py
├── ml_segmenter.py
├── llm_segmenter.py
├── react_segmenter.py
├── rag_segmenter.py
├── sql_queries.py
├── schemas.py
├── schema.md
├── README.md
└── rag_docs.json
```

### crime_statistics/__init__.py
```python
from .data_loader import CrimeStatisticsDataLoader
from .feature_processor import CrimeStatisticsFeatureProcessor
from .rule_based_segmenter import CrimeStatisticsRuleBasedSegmenter
from .ml_segmenter import CrimeStatisticsMlSegmenter
from .llm_segmenter import CrimeStatisticsLlmSegmenter
from .react_segmenter import CrimeStatisticsReactSegmenter
from .rag_segmenter import CrimeStatisticsRagSegmenter

__all__ = [
    'CrimeStatisticsDataLoader',
    'CrimeStatisticsFeatureProcessor',
    'CrimeStatisticsRuleBasedSegmenter',
    'CrimeStatisticsMlSegmenter',
    'CrimeStatisticsLlmSegmenter',
    'CrimeStatisticsReactSegmenter',
    'CrimeStatisticsRagSegmenter'
]
```

### crime_statistics/data_loader.py
```python
from ..base import DataLoader
from .sql_queries import CRIME_STATISTICS_QUERY

class CrimeStatisticsDataLoader(DataLoader):
    @property
    def query(self) -> str:
        return CRIME_STATISTICS_QUERY
```

### crime_statistics/feature_processor.py
```python
from typing import Dict, List
import pandas as pd
from ..base import FeatureProcessor

class CrimeStatisticsFeatureProcessor(FeatureProcessor):
    def process_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        features = raw_data.copy()
        features['crime_rate_per_1000'] = (
            features['total_crimes'] / 
            (features['population'] / 1000).replace(0, 1)
        )
        return features.fillna(0)
```

### crime_statistics/rule_based_segmenter.py
```python
from typing import Dict, List
import pandas as pd
from ..base import SegmentationStrategy

class CrimeStatisticsRuleBasedSegmenter(SegmentationStrategy):
    def __init__(self, threshold_multiplier: float = 1.0):
        self.threshold_multiplier = threshold_multiplier
        
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        median_rate = features['crime_rate_per_1000'].median()
        
        result_dict = {}
        for _, row in features.iterrows():
            tags = []
            if row['crime_rate_per_1000'] > median_rate * 1.5:
                tags.append('#high-crime')
            elif row['crime_rate_per_1000'] < median_rate * 0.5:
                tags.append('#low-crime')
            result_dict[row['neighborhood']] = tags
            
        return result_dict
```

### crime_statistics/schema.md
```markdown
# Crime Statistics Schema

## Database Tables
### `crime_statistics`
- `id`: Primary key
- `neighborhood`: Neighborhood name
- `crime_type`: Type of crime
- `incident_count`: Number of incidents
- `year`: Year of data

## Data Relationships
```mermaid
erDiagram
    CRIME_STATISTICS ||--o{ NEIGHBORHOODS : "occurred_in"
```
```

### Rent Statistics Implementation

Rent statistics module provides comprehensive rental market analysis with multiple segmentation strategies.

#### Key Features
- **Dynamic Thresholds**: Automatic calculation of price tiers based on city averages
- **Market Stability Analysis**: Volatility and trend detection
- **Affordability Scoring**: Income-adjusted rent analysis
- **Multi-Strategy Support**: Rule-based, LLM-powered, and ReAct agent segmentation

#### Available Tags
- `#luxury_rent`, `#mid_range_rent`, `#affordable_rent` - Price tier classification
- `#rising_rents`, `#stable_rents`, `#declining_rents` - Market trend analysis
- `#high_affordability`, `#low_affordability` - Affordability assessment
- `#volatile_market`, `#stable_market` - Market stability indicators

#### Usage Example
```python
from segmentation.rent_stats import RentStatsDataLoader, RentStatsFeatureProcessor
from segmentation.rent_stats import RentStatsReactSegmenter, RentStatsLlmSegmenter

# Load and process data
loader = RentStatsDataLoader()
processor = RentStatsFeatureProcessor()
segmenter = RentStatsReactSegmenter()  # or RentStatsLlmSegmenter()

raw_data = loader.load_data(engine)
features = processor.process_features(raw_data)
results = segmenter.segment(features)
```

### Land Prices Implementation

Land prices module analyzes property values and development potential with advanced segmentation.

#### Key Features
- **Price Premium Analysis**: Comparison to city-wide averages
- **Development Potential Scoring**: Floor space ratio and intensity metrics
- **Investment Scoring**: Dynamic weighting of market factors
- **Multi-Provider LLM Support**: Gemini, OpenAI, and DeepSeek integration

#### Available Tags
- `#luxury`, `#mid_range`, `#affordable` - Price tier classification
- `#rising`, `#stable`, `#declining` - Market trend analysis
- `#high_density`, `#development_potential` - Development characteristics
- `#volatile_market`, `#established_market` - Market maturity indicators

#### Usage Example
```python
from segmentation.land_prices import LandPricesDataLoader, LandPricesFeatureProcessor
from segmentation.land_prices import LandPricesReactSegmenter

# Load and process data
loader = LandPricesDataLoader()
processor = LandPricesFeatureProcessor()
segmenter = LandPricesReactSegmenter()

raw_data = loader.load_data(engine)
features = processor.process_features(raw_data)
results = segmenter.segment(features)
```

### Short-term Listings Implementation

Short-term rental market analysis with occupancy and pricing insights.

#### Key Features
- **Occupancy Rate Analysis**: Booking patterns and availability
- **Pricing Strategy Insights**: Dynamic pricing optimization
- **Seasonality Detection**: Peak and off-peak period identification
- **Market Saturation Assessment**: Supply-demand balance analysis

#### Available Tags
- `#high_occupancy`, `#low_occupancy` - Booking rate classification
- `#premium_pricing`, `#competitive_pricing` - Price positioning
- `#seasonal_peak`, `#year_round` - Seasonality patterns
- `#saturated_market`, `#emerging_market` - Market maturity

## ⚙️ Configuration Management

### segment_config.json Structure
```json
{
  "tables": {
    "green_spaces": ["rule_based", "llm", "react"],
    "rent_stats": ["rule_based", "react"],
    "land_prices": ["rule_based", "react"],
    "short_time_listings": ["rule_based"],
    "crime_statistics": ["rule_based", "llm", "rag"],
    "your_component": ["rule_based", "llm", "react", "rag"]
  }
}
```

### Environment Variables
```bash
# Required
DB_URL=postgresql://user:password@host:port/database
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
SEGMENT_CONFIG=config/segment_config.json
VISUALIZATION_PATH=viz/
```

## 🧪 Enhanced Testing Guidelines

### Unit Testing with Mocking
```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from your_component.feature_processor import YourComponentFeatureProcessor
from your_component.llm_segmenter import YourComponentLlmSegmenter

def test_feature_processor_with_outliers():
    """Test feature processing with outlier removal"""
    processor = YourComponentFeatureProcessor()
    test_data = pd.DataFrame({
        'total_metric': [100, 200, 1000],  # 1000 is an outlier
        'total_items': [10, 20, 5],
        'population': [1000, 2000, 500]
    })
    
    result = processor.process_features(test_data)
    assert 'normalized_metric' in result.columns
    assert 'metric_per_capita' in result.columns
    assert len(result) <= 2  # Outlier should be removed

def test_llm_segmenter_with_mock():
    """Test LLM segmenter with mocked API calls"""
    with patch('your_component.llm_segmenter.YourComponentLlmSegmenter._get_tags') as mock_get_tags:
        mock_get_tags.return_value = ['#high-metric', '#well-served']
        
        segmenter = YourComponentLlmSegmenter(llm_provider="gemini")
        test_features = pd.DataFrame({
            'neighborhood': ['Test Neighborhood'],
            'normalized_metric': [15.5],
            'total_items': [25]
        })
        
        result = segmenter.segment(test_features)
        assert 'Test Neighborhood' in result
        assert len(result['Test Neighborhood']) == 2

def test_react_agent_tool_integration():
    """Test ReAct agent tool integration"""
    from your_component.react_segmenter import YourComponentReactSegmenter
    
    # Test that tools are properly created
    segmenter = YourComponentReactSegmenter()
    tools = segmenter._create_tools()
    assert len(tools) > 0
    assert any(tool.name == "analyze_metric_patterns" for tool in tools)
```

### Integration Testing with Multiple Strategies
```python
def test_full_pipeline_with_multiple_strategies():
    """Test complete pipeline with multiple segmentation strategies"""
    orchestrator = SegmentationOrchestrator("sqlite:///:memory:")
    orchestrator.add("your_component", ["rule_based", "llm", "react"])
    
    # Mock data or use test database
    results = orchestrator.run_pipeline()
    assert not results.empty
    assert 'your_component' in results['table_name'].values
    
    # Verify multiple strategies produced results
    strategy_counts = results['table_name'].value_counts()
    assert len(strategy_counts) >= 1

def test_caching_integration():
    """Test data loader caching functionality"""
    from your_component.data_loader import YourComponentDataLoader
    
    loader = YourComponentDataLoader()
    mock_engine = Mock()
    
    # First call should execute query
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame({'neighborhood': ['Test'], 'metric': [100]})
        result1 = loader.load_data(mock_engine)
        
    # Second call should use cache
    with patch('pandas.read_sql') as mock_read_sql:
        result2 = loader.load_data(mock_engine)
        mock_read_sql.assert_not_called()  # Should not call SQL again
    
    assert result1.equals(result2)
```

### Performance Testing
```python
def test_segmentation_performance():
    """Test segmentation performance with large datasets"""
    import time
    
    # Create large test dataset
    large_features = pd.DataFrame({
        'neighborhood': [f'Neighborhood_{i}' for i in range(1000)],
        'normalized_metric': np.random.normal(100, 25, 1000),
        'total_items': np.random.randint(1, 50, 1000)
    })
    
    segmenter = YourComponentRuleBasedSegmenter()
    
    start_time = time.time()
    result = segmenter.segment(large_features)
    end_time = time.time()
    
    assert len(result) == 1000
    assert (end_time - start_time) < 5.0  # Should complete in under 5 seconds
```

## 🚀 Deployment Pipeline

### Pipeline Execution Flow
```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Orchestrator
    participant DB
    participant Strategies
    
    User->>Pipeline: Execute run_analysis_pipeline.py
    Pipeline->>Orchestrator: Initialize with DB_URL
    Orchestrator->>DB: Load configuration
    Orchestrator->>DB: Execute data loaders
    DB-->>Orchestrator: Return raw data
    Orchestrator->>Strategies: Process features & segment
    Strategies-->>Orchestrator: Return segmentation results
    Orchestrator->>Pipeline: Aggregate results
    Pipeline->>User: Return final results + visualizations
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **ClassNotFoundError** | Check `__init__.py` exports and naming conventions |
| **Database Connection** | Verify DB_URL environment variable |
| **Gemini API Errors** | Check GEMINI_API_KEY and internet connection |
| **Empty Results** | Verify SQL query returns data and feature calculations |
| **Memory Issues** | Use smaller batch sizes or optimize queries |

### Debugging Tips
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug SQL queries
print(your_data_loader.query)

# Inspect intermediate results
print(features.head())
print("Medians:", features.median())
```

## 📈 Enhanced Best Practices

### Code Organization & Architecture
- ✅ **Consistent Naming**: Use `YourComponentDataLoader`, `your_component` pattern
- ✅ **Single Responsibility**: Each class should have one clear purpose
- ✅ **Comprehensive Documentation**: Docstrings for all public methods and classes
- ✅ **Type Hints**: Use type hints throughout for better IDE support and code clarity
- ✅ **Modular Design**: Keep components independent and reusable
- ✅ **Configuration-Driven**: Use JSON configs for easy customization

### Performance Optimization
```python
# Use vectorized operations instead of loops
# Good: Vectorized operations (fast)
features['normalized'] = features['total'] / features['count'].replace(0, 1)

# Better: Use numpy for mathematical operations
import numpy as np
features['log_metric'] = np.log1p(features['total_metric'])

# Best: Use built-in pandas methods for aggregation
summary_stats = features.agg(['mean', 'median', 'std'])

# Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(neighborhood_data):
    """Cache results of expensive calculations"""
    # Complex computation here
    return result
```

### Advanced Error Handling & Resilience
```python
def robust_segment(self, features: pd.DataFrame):
    """Enhanced error handling with multiple fallback strategies"""
    try:
        # Primary segmentation attempt
        return self.segment(features)
        
    except Exception as e:
        logging.error(f"Primary segmentation failed: {e}")
        
        # Fallback 1: Try simplified rule-based approach
        try:
            return self._fallback_rule_based(features)
        except Exception as fallback_error:
            logging.warning(f"Rule-based fallback also failed: {fallback_error}")
            
            # Fallback 2: Use statistical baseline
            try:
                return self._statistical_baseline(features)
            except Exception as baseline_error:
                logging.error(f"All fallbacks failed: {baseline_error}")
                
                # Ultimate fallback: Return minimal tags
                return self._minimal_fallback(features)

def _fallback_rule_based(self, features):
    """Simplified rule-based fallback"""
    # Basic median-based rules
    result = {}
    for _, row in features.iterrows():
        tags = []
        if row.get('normalized_metric', 0) > features['normalized_metric'].median():
            tags.append('#above-average')
        result[row['neighborhood']] = tags
    return result

def _minimal_fallback(self, features):
    """Minimal fallback that always works"""
    return {row['neighborhood']: ['#basic-analysis'] for _, row in features.iterrows()}
```

### LLM Integration Best Practices
```python
# LLM Provider Configuration
def configure_llm_provider(self, provider="gemini"):
    """Dynamically configure LLM provider with fallbacks"""
    providers = {
        "gemini": self._create_langchain_gemini,
        "openai": self._create_langchain_openai,
        "deepseek": self._create_langchain_deepseek
    }
    
    try:
        return providers[provider]()
    except Exception as e:
        logging.warning(f"Primary LLM provider {provider} failed: {e}")
        # Try fallback providers in order
        for fallback_provider in [p for p in providers if p != provider]:
            try:
                return providers[fallback_provider]()
            except Exception:
                continue
        raise RuntimeError("All LLM providers failed")

# Prompt Engineering
def create_robust_prompt(self, data_summary: dict) -> str:
    """Create robust prompts with clear instructions and examples"""
    return f"""
    Analyze these neighborhood statistics and suggest descriptive tags:

    Data Summary:
    {json.dumps(data_summary, indent=2)}

    Instructions:
    1. Generate 3-6 hyphenated tags (e.g., 'high-density', 'well-maintained')
    2. Focus on quantitative characteristics
    3. Be specific and meaningful
    4. Return ONLY comma-separated tags

    Example Output: high-density,well-maintained,good-access
    """
```

### Caching & Performance
```python
# Intelligent caching strategy
def smart_cache_decorator(timeout=3600):
    """Decorator for smart caching with timeout"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = hashlib.md5(
                str(args + tuple(sorted(kwargs.items()))).encode()
            ).hexdigest()
            
            # Check cache validity
            current_time = time.time()
            if (cache_key in cache and
                current_time - cache[cache_key]['timestamp'] < timeout):
                return cache[cache_key]['result']
            
            # Execute and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = {
                'result': result,
                'timestamp': current_time
            }
            
            return result
        return wrapper
    return decorator
```

### Testing & Quality Assurance
```python
# Comprehensive test patterns
def test_strategy_comparison():
    """Test that different strategies produce consistent results"""
    features = create_test_features()
    
    # Test multiple strategies
    strategies = {
        'rule_based': RuleBasedSegmenter(),
        'llm': LlmSegmenter(),
        'react': ReactSegmenter()
    }
    
    results = {}
    for name, segmenter in strategies.items():
        try:
            results[name] = segmenter.segment(features)
        except Exception as e:
            logging.warning(f"Strategy {name} failed: {e}")
            results[name] = None
    
    # Verify at least one strategy worked
    assert any(results.values()), "All strategies failed"
    
    # Check consistency between working strategies
    working_results = {k: v for k, v in results.items() if v is not None}
    if len(working_results) > 1:
        # Compare results for consistency
        pass
```

## 🎨 Enhanced Visualization Integration

### Advanced Heatmap Generation
The system automatically generates comprehensive heatmaps showing tag distribution across neighborhoods with improved styling:

```python
# Enhanced heatmap with better styling and component-specific colors
def generate_enhanced_heatmap(results, component_name, save_path=None):
    """Generate enhanced heatmap with component-specific styling"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Filter results for specific component
    component_results = results[results['table_name'] == component_name]
    tags_expanded = component_results.explode('hashtags')
    
    # Create heatmap data with improved styling
    heatmap_data = pd.crosstab(
        tags_expanded['neighborhood'],
        tags_expanded['hashtags']
    )
    
    # Component-specific color schemes
    color_schemes = {
        'green_spaces': ['#f0f0f0', '#4caf50'],  # Green theme
        'rent_stats': ['#f0f0f0', '#ff6b6b'],    # Red theme
        'land_prices': ['#f0f0f0', '#ffa726'],   # Orange theme
        'crime_statistics': ['#f0f0f0', '#5c6bc0']  # Blue theme
    }
    
    colors = color_schemes.get(component_name, ['#f0f0f0', '#4c72b0'])
    
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        heatmap_data,
        cmap=colors,
        linewidths=0.5,
        annot=False,
        fmt="",
        cbar_kws={'label': 'Tag Presence'}
    )
    
    plt.title(f"{component_name.replace('_', ' ').title()} - Neighborhood Tag Distribution",
              fontsize=16, pad=20)
    plt.xlabel("Tags", fontsize=12)
    plt.ylabel("Neighborhoods", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### Interactive Network Graphs
Enhanced network visualizations with interactive capabilities and component-specific styling:

```python
# Interactive network graphs with improved layout and tooltips
def generate_interactive_network(results, component_name, save_path=None):
    """Generate interactive network graph with enhanced features"""
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    
    # Filter and prepare data
    component_results = results[results['table_name'] == component_name]
    tags_expanded = component_results.explode('hashtags')
    
    G = nx.Graph()
    
    # Add nodes with enhanced attributes
    for _, row in tags_expanded.iterrows():
        # Neighborhood nodes
        G.add_node(row['neighborhood'],
                   type='neighborhood',
                   size=800,
                   color=to_rgba('skyblue', 0.9))
        
        # Tag nodes
        G.add_node(row['hashtags'],
                   type='tag',
                   size=600,
                   color=to_rgba('salmon', 0.9))
        
        G.add_edge(row['neighborhood'], row['hashtags'],
                   weight=1.0, alpha=0.3)

    # Improved layout algorithm
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
    # Enhanced styling
    plt.figure(figsize=(20, 15))
    
    # Draw nodes with better styling
    neighborhoods = [n for n in G.nodes if G.nodes[n]['type'] == 'neighborhood']
    tags = [n for n in G.nodes if G.nodes[n]['type'] == 'tag']
    
    nx.draw_networkx_nodes(G, pos, nodelist=neighborhoods,
                          node_color=[G.nodes[n]['color'] for n in neighborhoods],
                          node_size=1000, alpha=0.9, edgecolors='black', linewidths=2)
    
    nx.draw_networkx_nodes(G, pos, nodelist=tags,
                          node_color=[G.nodes[n]['color'] for n in tags],
                          node_size=800, alpha=0.9, edgecolors='black', linewidths=2)
    
    # Enhanced edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.4,
                          edge_color='gray', style='dashed')
    
    # Improved labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif',
                           bbox=dict(facecolor='white', alpha=0.85,
                                   edgecolor='none', pad=2, boxstyle='round,pad=0.3'))
    
    plt.title(f"{component_name.replace('_', ' ').title()} - Neighborhood-Tag Network",
              fontsize=18, pad=30)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### Component-Specific Visualization Examples
The system now includes component-specific visualization examples in the `scripts/viz/` directory:

- **Green Spaces**: Network graphs showing park distribution patterns
- **Rent Statistics**: Heatmaps of rental price tiers across neighborhoods
- **Land Prices**: Investment potential visualizations with price gradients
- **Crime Statistics**: Safety assessment maps with crime density overlays

### Automated Visualization Pipeline
```python
# Automated visualization generation for all components
def generate_all_visualizations(results, output_dir="viz"):
    """Generate all visualizations for pipeline results"""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate visualizations for each component
    for component_name in results['table_name'].unique():
        heatmap_path = os.path.join(output_dir, f"{component_name}_heatmap.png")
        network_path = os.path.join(output_dir, f"{component_name}_network.png")
        
        generate_enhanced_heatmap(results, component_name, heatmap_path)
        generate_interactive_network(results, component_name, network_path)
        
        print(f"Generated visualizations for {component_name}")
```

## 🔮 Future Enhancements & Roadmap

### 🚀 Planned Features

#### Immediate Priorities (Q4 2025)
- **Real-time Data Streaming**: Integration with Kafka/Redis for live data updates
- **Advanced ML Integration**: Deep learning models for predictive segmentation
- **Custom Tagging Engine**: User-defined tag rules and templates
- **RESTful API Endpoints**: JSON API for programmatic access to segmentation results
- **Interactive Dashboard**: Web-based visualization dashboard with filtering and drill-down

#### Medium-term Goals (Q1 2026)
- **Multi-city Support**: Extend beyond Berlin to support multiple cities
- **Temporal Analysis**: Time-series segmentation and trend detection
- **Cross-component Correlation**: Identify relationships between different data domains
- **Automated Documentation**: AI-generated documentation for new components
- **Performance Monitoring**: Real-time performance metrics and alerting

#### Long-term Vision (2026+)
- **Federated Learning**: Privacy-preserving distributed model training
- **Natural Language Interface**: Conversational AI for data exploration
- **Automated Data Discovery**: AI-driven identification of new data sources
- **Predictive Analytics**: Forecast future neighborhood trends
- **Integration Marketplace**: Third-party component ecosystem

### 🛠️ Technical Extension Points

#### Custom Segmentation Strategies
```python
from scripts.segmentation.base import SegmentationStrategy
from typing import Dict, List
import pandas as pd

class CustomSegmentationStrategy(SegmentationStrategy):
    """Template for creating custom segmentation strategies"""
    
    def __init__(self, custom_config: dict = None):
        self.config = custom_config or {}
        # Initialize any custom components here
        
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Implement your custom segmentation logic
        
        Args:
            features: Processed feature DataFrame with neighborhood data
            
        Returns:
            Dictionary mapping neighborhood names to list of tags
        """
        results = {}
        
        # Your custom segmentation algorithm here
        for _, row in features.iterrows():
            neighborhood = row['neighborhood']
            tags = self._analyze_neighborhood(row)
            results[neighborhood] = tags
            
        return results
    
    def _analyze_neighborhood(self, row: pd.Series) -> List[str]:
        """Custom analysis logic for a single neighborhood"""
        tags = []
        
        # Example custom logic
        if row.get('custom_metric', 0) > self.config.get('threshold', 100):
            tags.append('#custom-high')
            
        return tags
```

#### Custom Data Loaders with External APIs
```python
from scripts.segmentation.base import DataLoader
import pandas as pd
import requests
import json

class ApiDataLoader(DataLoader):
    """Data loader that integrates with external APIs"""
    
    def __init__(self, api_url: str, api_key: str = None):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        
    def load_data(self, engine) -> pd.DataFrame:
        """Load data from external API with intelligent caching"""
        cache_key = f"api_{self.api_url}"
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Fetch from API
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        try:
            response = requests.get(self.api_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = self._process_api_response(data)
            
            # Cache the result
            self._save_to_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logging.error(f"API data loading failed: {e}")
            # Fallback to cached data or empty DataFrame
            return pd.DataFrame()
    
    def _process_api_response(self, data: dict) -> pd.DataFrame:
        """Transform API response into standardized DataFrame format"""
        # Your custom processing logic here
        return pd.DataFrame(data.get('results', []))
```

#### Plugin System Architecture
```python
# Example plugin registration system
class PluginRegistry:
    """Registry for custom segmentation plugins"""
    
    _plugins = {}
    
    @classmethod
    def register_plugin(cls, name: str, plugin_class):
        """Register a new segmentation plugin"""
        cls._plugins[name] = plugin_class
        
    @classmethod
    def get_plugin(cls, name: str):
        """Get a registered plugin by name"""
        return cls._plugins.get(name)
    
    @classmethod
    def list_plugins(cls):
        """List all registered plugins"""
        return list(cls._plugins.keys())

# Usage example
PluginRegistry.register_plugin('sentiment_analysis', CustomSentimentSegmenter)
```

### 🔧 Integration Patterns

#### Database Extension
```python
# Extending the system with new database types
class NoSqlDataLoader(DataLoader):
    """Data loader for NoSQL databases (MongoDB, Elasticsearch, etc.)"""
    
    def __init__(self, connection_string: str, collection: str):
        self.connection_string = connection_string
        self.collection = collection
        
    def load_data(self, engine) -> pd.DataFrame:
        """Load data from NoSQL database"""
        # Implement NoSQL-specific data loading
        # Convert to pandas DataFrame format
        pass
```

#### External Service Integration
```python
# Integration with external geospatial services
class GeospatialEnricher:
    """Enrich neighborhood data with geospatial information"""
    
    def __init__(self, mapbox_api_key: str):
        self.mapbox_api_key = mapbox_api_key
        
    def enrich_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add geospatial features to existing data"""
        enriched = features.copy()
        
        # Add distance to city center
        enriched['distance_to_center'] = self._calculate_distances(
            enriched['latitude'], enriched['longitude']
        )
        
        # Add proximity to public transport
        enriched['transit_score'] = self._calculate_transit_scores(
            enriched['latitude'], enriched['longitude']
        )
        
        return enriched
```

### 📊 Advanced Analytics Extensions

#### Time Series Analysis
```python
class TemporalSegmenter(SegmentationStrategy):
    """Segmentation based on temporal patterns and trends"""
    
    def __init__(self, lookback_period: int = 12):
        self.lookback_period = lookback_period  # months
        
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze temporal patterns for segmentation"""
        results = {}
        
        for neighborhood, data in features.groupby('neighborhood'):
            trends = self._analyze_trends(data)
            tags = self._generate_trend_tags(trends)
            results[neighborhood] = tags
            
        return results
    
    def _analyze_trends(self, data: pd.DataFrame) -> dict:
        """Analyze temporal trends in neighborhood data"""
        # Implement trend analysis (moving averages, seasonality, etc.)
        pass
```

#### Cross-Domain Correlation
```python
class CrossDomainSegmenter(SegmentationStrategy):
    """Segmentation that correlates multiple data domains"""
    
    def __init__(self, component_weights: dict = None):
        self.component_weights = component_weights or {
            'green_spaces': 0.3,
            'rent_stats': 0.4,
            'crime_statistics': 0.3
        }
        
    def segment(self, all_features: dict) -> Dict[str, List[str]]:
        """Segment based on correlations across multiple domains"""
        # Combine features from different components
        combined_features = self._combine_features(all_features)
        
        # Perform cross-domain analysis
        correlations = self._analyze_correlations(combined_features)
        
        return self._generate_cross_domain_tags(correlations)
```

### 🎯 Contribution Guidelines

#### Adding New Components
1. **Follow Established Patterns**: Use existing components as templates
2. **Comprehensive Testing**: Include unit, integration, and performance tests
3. **Documentation**: Update README.md and schema documentation
4. **Configuration**: Add to segment_config.json with appropriate strategies
5. **Visualization**: Implement component-specific visualization templates

#### Extending Core Functionality
1. **Backward Compatibility**: Ensure changes don't break existing functionality
2. **Configuration-Driven**: Use environment variables and JSON configs
3. **Modular Design**: Keep extensions independent and optional
4. **Performance Considerations**: Optimize for large datasets
5. **Error Handling**: Implement robust fallback mechanisms

### 🔮 Emerging Technologies Integration

#### Blockchain Integration
```python
class BlockchainVerifier:
    """Verify data integrity using blockchain technology"""
    
    def __init__(self, blockchain_url: str):
        self.blockchain_url = blockchain_url
        
    def verify_data_integrity(self, data_hash: str) -> bool:
        """Verify that data hasn't been tampered with"""
        # Implement blockchain verification logic
        pass
```

#### Federated Learning
```python
class FederatedSegmenter(SegmentationStrategy):
    """Privacy-preserving segmentation using federated learning"""
    
    def __init__(self, federated_nodes: list):
        self.nodes = federated_nodes
        
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Perform segmentation without centralized data collection"""
        # Implement federated learning logic
        pass
```

---

## 📞 Enhanced Support & Resources

### 📚 Comprehensive Documentation
- **Component Documentation**: Individual [`README.md`](scripts/segmentation/green_spaces/README.md:1) files for each module
- **Architecture Overview**: This handbook provides system-wide architecture and patterns
- **API Documentation**: Code docstrings and type hints throughout the codebase
- **Configuration Guide**: Detailed [`segment_config.json`](scripts/config/segment_config.json:1) documentation
- **Database Schema**: Component-specific [`schema.md`](scripts/segmentation/green_spaces/schema.md:1) files

### 🎯 Reference Implementations
- **Green Spaces**: [`segmentation/green_spaces/`](scripts/segmentation/green_spaces/) - Complete reference implementation
- **Rent Statistics**: [`segmentation/rent_stats/`](scripts/segmentation/rent_stats/) - Advanced ReAct agent example
- **Land Prices**: [`segmentation/land_prices/`](scripts/segmentation/land_prices/) - Multi-provider LLM integration
- **Crime Statistics**: [`segmentation/crime_statistics/`](scripts/segmentation/crime_statistics/) - RAG-based segmentation

### 🛠️ Development Resources
- **Base Classes**: Core abstractions in [`scripts/segmentation/base.py`](scripts/segmentation/base.py:1)
- **ReAct Framework**: Agent-based segmentation in [`scripts/segmentation/react_base.py`](scripts/segmentation/react_base.py:1)
- **Configuration Management**: [`scripts/config/`](scripts/config/) directory
- **Visualization Templates**: [`scripts/viz/`](scripts/viz/) visualization components
- **Testing Utilities**: Comprehensive test patterns and mock utilities

### 🐛 Issue Reporting & Support
- **GitHub Issues**: Use the project's issue tracker for:
  - Bug reports with reproduction steps
  - Feature requests with use cases
  - Documentation improvements
  - Performance issues

- **Troubleshooting Guide**: Refer to the [Troubleshooting](#-troubleshooting) section
- **Common Issues**: See the [Common Issues](#common-issues) table for quick solutions

### 🤝 Contribution Guidelines
- **Code Style**: Follow established patterns and naming conventions
- **Testing Requirements**: Include comprehensive unit and integration tests
- **Documentation**: Update all relevant documentation files
- **Backward Compatibility**: Ensure no breaking changes to existing functionality
- **Performance Considerations**: Optimize for large datasets and scalability

### 📖 Learning Resources
- **Pandas Documentation**: Essential for data manipulation ([pandas.pydata.org](https://pandas.pydata.org))
- **LangChain Guides**: LLM integration patterns ([langchain.com](https://langchain.com))
- **SQL Best Practices**: Database query optimization
- **Machine Learning**: Scikit-learn for ML-based segmentation
- **Visualization**: Matplotlib/Seaborn for data visualization

### 🔧 Development Tools
- **Python 3.8+**: Required runtime environment
- **Jupyter Notebooks**: For exploratory data analysis
- **VS Code/PyCharm**: Recommended IDEs with Python support
- **Docker**: Containerization for consistent environments
- **Git**: Version control and collaboration

### 🚀 Quick Help Commands
```bash
# Run the complete pipeline
python scripts/run_analysis_pipeline.py

# Test a specific component
python -m pytest scripts/segmentation/your_component/tests/ -v

# Generate visualizations
python scripts/viz/generate_visualizations.py

# Check code quality
python -m flake8 scripts/segmentation/your_component/
python -m mypy scripts/segmentation/your_component/
```

### 📋 Support Matrix
| Component | Supported Strategies | LLM Providers | Special Features |
|-----------|---------------------|---------------|------------------|
| **Green Spaces** | Rule-based, LLM, ReAct | Gemini, OpenAI, DeepSeek | Advanced caching, RAG integration |
| **Rent Statistics** | Rule-based, ReAct | Gemini, OpenAI | Dynamic thresholds, market analysis |
| **Land Prices** | Rule-based, ReAct | Gemini, OpenAI, DeepSeek | Investment scoring, development potential |
| **Crime Statistics** | Rule-based, LLM, RAG | Gemini | Safety assessment, trend analysis |

### 🆘 Emergency Recovery
```python
# Minimal fallback implementation
from scripts.segmentation.base import SegmentationStrategy
import pandas as pd

class EmergencySegmenter(SegmentationStrategy):
    """Minimal segmenter for emergency recovery"""
    
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        return {row['neighborhood']: ['#basic-analysis'] for _, row in features.iterrows()}
```

### 📞 Community & Support
- **Slack/Discord**: Join the development community for real-time support
- **Weekly Office Hours**: Scheduled Q&A sessions with core maintainers
- **Code Reviews**: Submit pull requests for expert feedback
- **Pair Programming**: Collaborative development sessions available

**Remember**: Always check existing implementations before creating new components. The `green_spaces` module serves as the canonical reference for all patterns and best practices.

## 🎯 Enhanced Quick Start Checklist

### 🏗️ Foundation Setup
- [ ] **Create Component Directory Structure** with enhanced patterns
  ```bash
  segmentation/your_component/
  ├── __init__.py          # Module exports
  ├── data_loader.py       # Data loading with caching
  ├── feature_processor.py # Feature processing
  ├── rule_based_segmenter.py # Rule-based segmentation
  ├── llm_segmenter.py     # LLM-powered segmentation (optional)
  ├── react_segmenter.py   # ReAct agent segmentation (optional)
  ├── sql_queries.py       # SQL queries
  ├── schemas.py           # Pydantic tag schemas
  ├── schema.md           # Database documentation
  ├── README.md           # Component documentation
  └── rag_docs.json       # RAG knowledge base (optional)
  ```

- [ ] **Define Database Schema** in [`schema.md`](scripts/segmentation/green_spaces/schema.md:1)
- [ ] **Create Tag Schemas** in [`schemas.py`](scripts/segmentation/green_spaces/schemas.py:1) with Pydantic models

### 🔧 Core Implementation
- [ ] **Implement DataLoader** with intelligent caching
  - Extend [`base.DataLoader`](scripts/segmentation/base.py:16)
  - Include SQL query in [`sql_queries.py`](scripts/segmentation/green_spaces/sql_queries.py:1)
  - Enable caching with timeout management

- [ ] **Create FeatureProcessor** with comprehensive transformations
  - Extend [`base.FeatureProcessor`](scripts/segmentation/base.py:154)
  - Implement feature engineering and normalization
  - Include outlier removal and data validation

- [ ] **Implement Segmentation Strategies** (choose one or more):
  - **Rule-Based**: Extend [`base.SegmentationStrategy`](scripts/segmentation/base.py:162)
  - **LLM-Powered**: Extend [`base.LLMSegmentationStrategy`](scripts/segmentation/base.py:257)
  - **ReAct Agent**: Extend [`react_base.ReactSegmentationStrategy`](scripts/segmentation/react_base.py:18)
  - **RAG-Based**: Extend [`base.SegmentationStrategy`](scripts/segmentation/base.py:162) with vector retrieval

### ⚙️ Configuration & Integration
- [ ] **Configure Module Exports** in [`__init__.py`](scripts/segmentation/green_spaces/__init__.py:1)
- [ ] **Update Configuration** in [`config/segment_config.json`](scripts/config/segment_config.json:1)
- [ ] **Set Environment Variables**:
  ```bash
  DB_URL=postgresql://user:password@host:port/database
  GEMINI_API_KEY=your_gemini_api_key_here  # For LLM strategies
  OPENAI_API_KEY=your_openai_api_key_here  # Alternative LLM provider
  DEEPSEEK_API_KEY=your_deepseek_api_key_here  # Alternative LLM provider
  SEGMENT_CONFIG=config/segment_config.json
  VISUALIZATION_PATH=viz/
  CACHE_PATH=cache/data_loader_cache.json
  TIME_OUT=43200  # 12-hour cache timeout
  LOG_LEVEL=INFO
  ```

### 🧪 Testing & Validation
- [ ] **Unit Testing** with comprehensive test coverage
  - Test DataLoader caching functionality
  - Test FeatureProcessor transformations
  - Test segmentation strategies with mock data
  - Include error handling and edge cases

- [ ] **Integration Testing** with multiple strategies
  - Test pipeline execution with orchestrator
  - Verify result aggregation and consistency
  - Test caching integration and performance

- [ ] **Performance Testing** with large datasets
  - Verify scalability with 1000+ neighborhoods
  - Test memory usage and execution time
  - Validate caching effectiveness

### 📊 Visualization & Documentation
- [ ] **Implement Visualization Integration**
  - Test heatmap generation with component-specific styling
  - Verify network graph creation
  - Ensure proper file output and formatting

- [ ] **Create Comprehensive Documentation**
  - Update [`README.md`](scripts/segmentation/green_spaces/README.md:1) with usage examples
  - Document database schema and relationships
  - Include configuration examples and best practices
  - Add troubleshooting guide and common issues

- [ ] **Verify End-to-End Pipeline**
  - Test complete pipeline execution
  - Validate visualization output
  - Check result quality and consistency

### 🚀 Advanced Features (Optional)
- [ ] **Multi-Provider LLM Support** (Gemini, OpenAI, DeepSeek)
- [ ] **ReAct Agent Tools** with component-specific analysis
- [ ] **RAG Integration** with knowledge base documents
- [ ] **Enhanced Caching** with intelligent timeout management
- [ ] **Component-Specific Visualization** with custom styling
- [ ] **Error Recovery** with multiple fallback strategies

### ✅ Final Validation
- [ ] **Code Quality Check**
  - Follow naming conventions (`YourComponentDataLoader`, `your_component`)
  - Use type hints and comprehensive docstrings
  - Implement proper error handling and logging
  - Follow single responsibility principle

- [ ] **Performance Benchmarking**
  - Test with production-scale data
  - Verify memory efficiency
  - Ensure acceptable execution times

- [ ] **Documentation Review**
  - Verify all public methods are documented
  - Check configuration examples are accurate
  - Ensure troubleshooting guide is comprehensive

- [ ] **Integration Testing**
  - Test with existing components
  - Verify no breaking changes
  - Ensure backward compatibility

### 🎯 Getting Started Template
```bash
# Quick setup command sequence
cd scripts/segmentation
mkdir your_component
cd your_component

# Create basic structure
touch __init__.py data_loader.py feature_processor.py rule_based_segmenter.py
touch sql_queries.py schemas.py schema.md README.md

# Copy and adapt from existing components
cp ../green_spaces/schema.md ./
cp ../green_spaces/schemas.py ./
cp ../green_spaces/__init__.py ./
```

**Pro Tip**: Start with the rule-based segmenter first, then add LLM/ReAct strategies incrementally. Test each component independently before integrating into the full pipeline.

---

*This handbook is maintained and updated regularly. Last updated: 2025-08-28*