# Berlin Urban Analytics Project
This project represents a sophisticated **urban analytics system** that leverages cutting-edge Large Language Models (LLMs) and agentic reasoning to provide intelligent neighborhood segmentation and urban feature analysis for Berlin districts.

## Project Structure
```
.
├── segmentation/          # AI-powered neighborhood analysis modules
│   ├── orchestrator.py   # AI orchestration and coordination logic
│   ├── react_base.py     # ReAct agent framework with reasoning capabilities
│   ├── retriever.py      # RAG implementation for context retrieval
│   ├── base.py           # Core AI infrastructure (LLM, caching, agents)
│   ├── banks/            # Banking service analysis with AI reasoning
│   ├── bus_tram_stops/   # Public transport accessibility with AI
│   ├── crime_statistics/ # Crime pattern analysis using LLMs
│   ├── dental_offices/   # Dental care service analysis with AI
│   ├── districts_pop_stat/ # Population statistics with AI insights
│   ├── hospitals/        # Healthcare facility analysis using ReAct agents
│   ├── kindergartens/    # Early education facility analysis with AI
│   ├── land_prices/      # Real estate market analysis with LLM reasoning
│   ├── long_term_listings/ # Long-term rental market AI analysis
│   ├── milieuschutz_protection_zones/ # Heritage protection AI analysis
│   ├── parks/            # Green space analysis with ReAct agents
│   ├── playgrounds/      # Recreational space analysis using AI
│   ├── pools/            # Swimming facility analysis with LLM reasoning
│   ├── rent_stats_per_neighborhood/ # Rental market AI analysis
│   ├── sbahn/            # S-Bahn transport analysis with AI
│   ├── schools/          # Educational facility analysis using ReAct
│   ├── short_term_listings/ # Short-term rental AI analysis
│   ├── ubahn/            # U-Bahn transport analysis with LLM reasoning
│   ├── universities/     # Higher education analysis using AI
│   ├── venues/           # Entertainment venue analysis with ReAct agents
│   └── README.md         # AI architecture documentation

├── README.md              # This file
├── requirements.txt       # Dependencies
├── schema.txt            # Database schema definitions
└── viz/                  # AI-generated visualization outputs
```

## 🤖 Features

### Multi-Modal Architecture
- **LLM-Powered Analysis**: Direct integration with Gemini, OpenAI, and DeepSeek models
- **ReAct Agent Framework**: Reasoning and Action agents with dynamic tool usage
- **RAG Implementation**: Vector-based retrieval for contextual neighborhood analysis
- **Multi-LLM Support**: Configurable AI model selection (Gemini, OpenAI, DeepSeek)

### Intelligent Segmentation Approaches
- **Rule-Based**: Traditional statistical thresholding with dynamic boundaries
- **LLM-Powered**: Direct LLM analysis with structured prompting and JSON output
- **ReAct Agents**: Advanced reasoning with tool-based analysis and iterative thinking

### Infrastructure Components
- **Smart Caching**: Intelligent query caching with automatic invalidation
- **Dynamic Thresholding**: AI-driven statistical boundary calculation
- **Agent Communication**: Message bus for multi-agent coordination
- **Error Recovery**: Graceful fallback mechanisms for AI failures

## Key Features
- **Advanced AI Integration**: Multi-LLM support with ReAct reasoning agents
- **Intelligent Neighborhood Segmentation**: Three-tier AI analysis approach
- **Dynamic Statistical Analysis**: AI-powered threshold calculation and pattern detection
- **Comprehensive Documentation**: Detailed AI methodology and implementation guides
- **Standardized AI Interfaces**: Consistent patterns across all analysis modules
- **Smart Caching**: Efficient data processing with intelligent cache management
- **Multi-Model Support**: Configurable AI backend selection

## 🏗️ Module Architecture

Each segmentation module implements a sophisticated **three-tier AI analysis system** with standardized interfaces:

### Core Components:
- **Data Loaders**: Smart data fetching with intelligent caching and preprocessing
- **Feature Processors**: AI-enhanced metric calculation with dynamic statistical analysis
- **Segmenters**: Three-tier AI analysis approach:
  - **Rule-based**: Traditional statistical segmentation with AI-optimized thresholds
  - **LLM-powered**: Direct LLM analysis with structured prompting and JSON parsing
  - **ReAct agents**: Advanced reasoning with dynamic tool usage and iterative analysis

### Features:
- **Dynamic Threshold Calculation**: AI-driven statistical boundary optimization
- **Multi-LLM Integration**: Configurable model selection (Gemini, OpenAI, DeepSeek)
- **Intelligent Pattern Detection**: Statistical clustering with AI-enhanced insights
- **Agentic Reasoning**: ReAct framework with tool-based analysis capabilities
- **Contextual Understanding**: RAG implementation for neighborhood context retrieval
- **Adaptive Error Handling**: Graceful fallback mechanisms for AI failures

## 🎯 Analysis Modules

### Transportation & Infrastructure
- **bus_tram_stops**: AI-powered public transport accessibility with dynamic coverage analysis
- **sbahn**: S-Bahn network analysis using ReAct agents for accessibility scoring
- **ubahn**: U-Bahn network analysis with LLM-powered route optimization insights

### Healthcare & Social Services
- **hospitals**: Healthcare facility analysis with AI-driven quality assessment
- **dental_offices**: Dental care service accessibility using intelligent pattern recognition
- **kindergartens**: Early childhood education facilities with AI-enhanced demand analysis

### Education & Culture
- **schools**: Primary and secondary education analysis with ReAct agent reasoning
- **universities**: Higher education institution analysis using LLM-powered insights
- **venues**: Entertainment and cultural venue analysis with AI-driven popularity scoring

### Real Estate & Housing
- **land_prices**: Property value analysis with AI-powered market trend detection
- **long_term_listings**: Long-term rental market analysis using intelligent clustering
- **short_term_listings**: Short-term rental market analysis with AI-driven demand patterns
- **rent_stats_per_neighborhood**: Comprehensive rental statistics with predictive analytics

### Public Amenities & Recreation
- **parks**: Green space analysis with ReAct agents for maintenance and accessibility
- **playgrounds**: Children's recreational facilities with AI-driven safety assessment
- **pools**: Swimming facility analysis using intelligent capacity and quality metrics

### Safety & Community
- **crime_statistics**: Crime pattern analysis with AI-powered risk assessment
- **milieuschutz_protection_zones**: Heritage protection analysis using contextual AI
- **districts_pop_stat**: Population demographic analysis with predictive modeling

### Financial Services
- **banks**: Banking service accessibility with AI-driven ATM network optimization

## 🏷️ Generated Label Categories

Each module leverages **advanced AI reasoning** to generate descriptive hashtags for districts based on comprehensive analysis:

### Common Tags:
- **#well-served**: AI-calculated above-median service availability
- **#underserved**: AI-identified below 50% of median service availability
- **#transport-desert**: LLM-identified limited public transport access
- **#good-coverage**: ReAct agent-determined above-median service coverage
- **#high-quality-service**: AI-evaluated above-median service quality

### Enhanced Module-Specific Tags:
- **banks**: #well-banked, #banking-desert, #good-service, #accessible-banking, #atm-rich, #atm-poor
- **bus_tram_stops**: #limited-coverage, #high-quality-service
- **parks**: #high-green-density, #well-maintained, #large-parks, #accessible-parks
- **hospitals**: #well-served-healthcare, #medical-desert, #high-quality-care

## 🚀 Usage Examples

### Individual Module with AI Analysis
```python
# For any module (example with banks) - AI-enhanced analysis
from segmentation.banks.data_loader import BanksDataLoader
from segmentation.banks.rule_based_segmenter import BanksRuleBasedSegmenter
from segmentation.banks.llm_segmenter import BanksLlmSegmenter
from segmentation.banks.react_segmenter import BanksReactSegmenter

loader = BanksDataLoader()
rule_segmenter = BanksRuleBasedSegmenter(threshold_multiplier=1.0)
llm_segmenter = BanksLlmSegmenter()  # Direct LLM analysis
react_segmenter = BanksReactSegmenter()  # ReAct agent reasoning

features_df = loader.load_data()
rule_tags = rule_segmenter.segment(features_df)  # Traditional stats
llm_tags = llm_segmenter.segment(features_df)    # LLM-powered insights
react_tags = react_segmenter.segment(features_df) # Advanced reasoning
```

### Orchestrated Analysis
```python
from segmentation.orchestrator import SegmentationOrchestrator

# AI-powered analysis of all urban data modules
orchestrator = SegmentationOrchestrator(db_url="your_database_url")
orchestrator.add("parks", segment_types=["rule_based", "llm", "react"])
orchestrator.add("hospitals", segment_types=["llm", "react"])
results = orchestrator.run_pipeline()

# Generate visualizations
orchestrator.visualize_results(results, "ai_analysis_output/")
```

### AI Configuration
```python
# Multi-LLM setup with custom configuration
from segmentation.parks.react_segmenter import ParksReactSegmenter

# Configure with different AI models
segmenter_gemini = ParksReactSegmenter()  # Default: Gemini
# segmenter_openai = ParksReactSegmenter(llm_name="openai")  # OpenAI
# segmenter_deepseek = ParksReactSegmenter(llm_name="deepseek")  # DeepSeek

# Run ReAct agent analysis
ai_tags = segmenter_gemini.segment(features_df)
```

## 🛠️ Setup & Configuration

### Getting Started
1. **Install AI Dependencies**: `pip install -r requirements.txt`
2. **Configure Database**: Set up environment variables for database connection
3. **Setup AI APIs**: Configure API keys for LLM services:
   - `GEMINI_API_KEY` for Google Gemini
   - `OPENAI_API_KEY` for OpenAI models  
   - `DEEPSEEK_API_KEY` for DeepSeek models
4. **Run Analysis**: Use orchestrator or individual AI modules

### Model Configuration
- **Gemini**: Default model (`gemini-2.5-flash-lite`) with structured output
- **OpenAI**: GPT-4o-mini with enhanced reasoning capabilities
- **DeepSeek**: Cost-effective alternative with strong performance

## 📚 Documentation

Each module contains comprehensive AI documentation:
- `README.md` - AI methodology, reasoning patterns, and implementation details
- `schema.md` - Data schema with AI-derived features and statistical approaches
- `schemas.py` - Pydantic models for AI-generated tag definitions and validation
- `sql_queries.py` - Standardized queries optimized for AI analysis
- `react_segmenter.py` - ReAct agent implementation with custom tools
- `llm_segmenter.py` - Direct LLM integration with structured prompting

## 🔧 Edge Case Handling

All AI modules implement sophisticated error handling:
- **Failure Recovery**: Graceful fallback to rule-based analysis
- **Missing Data**: Intelligent imputation and contextual understanding
- **Statistical Outliers**: AI-driven outlier detection and handling
- **API Rate Limits**: Smart retry mechanisms with exponential backoff
- **JSON Parsing**: Robust extraction from LLM responses with multiple pattern matching

## 🎯 Performance & Optimization

### Intelligent Caching System
- **Query Result Caching**: 12-hour cache duration with automatic invalidation
- **Statistical Analysis Caching**: Pre-computed statistical distributions
- **LLM Response Caching**: Structured output caching for repeated analyses

### Dynamic Analysis Tools
- **Statistical Analysis**: Descriptive statistics, distribution analysis, pattern detection
- **Cluster Identification**: Natural cluster detection with dynamic boundaries
- **Correlation Analysis**: Cross-feature correlation with strength assessment
- **Neighborhood Comparison**: Dynamic outlier detection and performance ranking

### ReAct Agent Framework
- **Thought-Action-Observation**: Structured reasoning process
- **Dynamic Tool Usage**: Context-aware tool selection and parameterization
- **Iterative Analysis**: Multi-step reasoning with progressive refinement
- **JSON Output Parsing**: Robust extraction from agent responses

## 🚀 Advanced Capabilities

### Multi-Agent Coordination
- **Orchestrator Pattern**: Centralized coordination of multiple AI agents
- **Message Bus**: Inter-agent communication for complex analysis
- **Result Aggregation**: Intelligent combination of multiple AI approaches

### Contextual Intelligence
- **RAG Implementation**: Vector-based retrieval for neighborhood context
- **Dynamic Prompting**: Context-aware prompt generation for LLMs
- **Domain-Specific Tools**: Custom analysis tools for each urban domain

### Production-Ready
- **Error Resilience**: Comprehensive error handling and fallback strategies
- **Performance Optimization**: Efficient LLM usage with smart caching
- **Scalable Architecture**: Modular design supporting multiple AI models
- **Monitoring & Logging**: Comprehensive logging for AI operations

This project demonstrates cutting-edge AI application in urban analytics, combining traditional statistical methods with advanced LLM reasoning and agentic systems to provide comprehensive neighborhood intelligence.
