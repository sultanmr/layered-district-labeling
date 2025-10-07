from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import PromptTemplate
import traceback

from .base import LLMSegmentationStrategy

# Set up logger
logger = logging.getLogger(__name__)


class ReactSegmentationStrategy(LLMSegmentationStrategy):
    """Base class for ReAct agent-based segmentation strategies"""
    
    def __init__(self, tools, cols, llm_name="gemini"):
        # Initialize Gemini LLM through parent class
        super().__init__(False) # Don't auto-init Gemini here
        self.tools = tools
        self.cols = cols
        
        # Create LangChain compatible LLM
        #self.langchain_llm = self._create_langchain_openai()
        if llm_name == "gemini":
            self.langchain_llm = self._create_langchain_gemini()
        elif llm_name == "openai":            
            self.langchain_llm = self._create_langchain_openai()
        elif llm_name == "deepseek": 
            self.langchain_llm = self._create_langchain_deepseek()
        
        self.agent_executor = self._create_agent_executor()
        
    
    def _identify_clusters_wrapper(self, input_str: str) -> dict:
        """Wrapper for identify_clusters that accepts string input"""
        # Clean up input - remove any extra quotes and whitespace
        clean_input = input_str.strip().strip("'\"").lower()
        
        if hasattr(self, 'features_df') and clean_input == 'all':
            return self._identify_clusters(self.features_df, self.cols)
        return {"error": "Input should be 'all' to analyze all numeric columns"}
    
    def _compare_neighborhoods_wrapper(self, column_name: str) -> dict:
        """Wrapper for compare_neighborhoods that accepts string input"""
        # Clean up column name - remove any extra quotes
        clean_column_name = column_name.strip().strip("'\"")
        
        if hasattr(self, 'features_df') and clean_column_name in self.features_df.columns:
            return self._compare_neighborhoods(self.features_df, self.cols)
        return {"error": f"Column '{clean_column_name}' not found in features data. Available columns: {list(self.features_df.columns) if hasattr(self, 'features_df') else 'None'}"}
    
    def _generate_dynamic_tags_wrapper(self, input_str: str) -> dict:
        """Wrapper for generate_dynamic_tags that accepts string input"""
        # Clean up input - remove any extra quotes and whitespace
        clean_input = input_str.strip().strip("'\"").lower()
        
        if hasattr(self, 'features_df') and clean_input == 'all':
            return self._generate_dynamic_tags(self.features_df, self.cols)
        return {"error": "Input should be 'all' to analyze all numeric columns"}
    
    def _analyze_spatial_correlations_wrapper(self, input_str: str) -> dict:
        """Wrapper for analyze_spatial_correlations that accepts string input"""
        # Clean up input - remove any extra quotes and whitespace
        clean_input = input_str.strip().strip("'\"").lower()
        
        if hasattr(self, 'features_df') and clean_input == 'all':
            return self._analyze_correlations(self.features_df, self.cols)
        return {"error": "Input should be 'all' to analyze all correlations"}
    
    # Original tool implementations
    def _calculate_statistics(self, data: pd.Series) -> dict:
        """Calculate descriptive statistics for numerical data - uses base class method"""
        return self._calculate_statistics(data)
    
    def _identify_clusters(self, features_df: pd.DataFrame, numeric_cols) -> dict:
        """Identify natural clusters using statistical patterns with enhanced analysis"""        
        features = features_df[numeric_cols]
        
        # Enhanced cluster analysis with dynamic boundaries
        clusters = {}
        patterns = {}
        
        for col in numeric_cols:
            col_data = features[col].dropna()
            if len(col_data) > 0:
                stats = self._calculate_statistics(col_data)
                distribution = self._analyze_distribution(col_data)
                pattern_analysis = self._identify_patterns(col_data)
                
                q1, q3 = stats["q1"], stats["q3"]
                iqr = stats["iqr"]
                
                clusters[col] = {
                    "low": float(q1 - iqr * 0.25),
                    "medium_low": float(q1),
                    "medium_high": float(q3),
                    "high": float(q3 + iqr * 0.25),
                    "distribution": stats,
                    "pattern_analysis": pattern_analysis
                }
                
                patterns[col] = pattern_analysis
        
        # Cross-feature correlation analysis
        correlations = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                try:
                    corr = features[col1].corr(features[col2])
                    correlations[f"{col1}_{col2}"] = {
                        "correlation": float(corr),
                        "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak",
                        "direction": "positive" if corr > 0 else "negative"
                    }
                except:
                    pass
        
        return {
            "feature_clusters": clusters,
            "cross_feature_patterns": patterns,
            "correlations": correlations,
            "interpretation": "Enhanced cluster analysis with dynamic boundaries and pattern detection"
        }
    def _create_agent_executor(self):
        """Create ReAct agent executor with custom prompt and increased limits"""
        prompt = self._get_agent_prompt()
        
        agent = create_react_agent(self.langchain_llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,  # Increased from default 10
            max_execution_time=120,  # Increased from default 60 seconds
            early_stopping_method="force"  # Force completion when hitting limits
        )
    
    def _get_agent_prompt(self) -> PromptTemplate:
        """Get customized ReAct prompt for segmentation tasks"""
        # Use a custom prompt optimized for JSON output and reduced tool usage
        # This prompt is designed to work with the child class's analysis prompt
        return PromptTemplate.from_template("""
You are an expert urban analyst specializing in neighborhood segmentation.
Your task is to analyze neighborhood data and generate descriptive hashtags.

Available tools: {tools}

CRITICAL INSTRUCTIONS:
1. You MUST follow the ReAct format EXACTLY: Thought -> Action -> Action Input -> Observation
2. DO NOT output JSON directly without going through the Thought->Action->Observation process first
3. You MUST use the tools to analyze the data before providing your final answer
4. Only provide JSON in the Final Answer section, after you have completed your analysis
5. If you see JSON format instructions in the question, ignore them until the Final Answer

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (MUST be valid JSON only)

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
    
    def _get_results (self, analysis_prompt, features):
        """Helper to run the agent and handle exceptions, returning tags or fallback"""
        try:
            logger.info(f"Invoking agent with prompt length: {len(analysis_prompt)}")
            result = self.agent_executor.invoke({
                "input": analysis_prompt,
                "chat_history": []
            })
                
            logger.info("Agent execution completed successfully")
            # Parse agent output to extract tags
            output_text = result.get("output", "")
            output_json = self._extract_json_from_string(output_text)
            neighborhoods = output_json.get("neighborhoods", {})
            if len(neighborhoods) == 0:
                raise ValueError("No neighborhoods found in agent output JSON")
                
            return neighborhoods

            
        except Exception as e:
            logger.error(f"ReAct agent error: {e}")            
            logger.error(traceback.format_exc())         
        finally:
            # Clean up stored features
            if hasattr(self, 'features_df'):
                delattr(self, 'features_df')


    
    
    def _create_analysis_prompt(self, title, features: pd.DataFrame, predefined_tags) -> str:
        """Create analysis prompt for segmentation with JSON output format"""
        
        tag_rules = "\n".join([f"- {tag.tag}: {tag.rule} ({tag.description})" for tag in predefined_tags])
        
        #format_instructions = self.tag_parser.get_format_instructions()
        
        # Get actual district names from the data
        neighborhood_names = features['district'].tolist() if 'district' in features.columns else ["Unknown districts"]
        
        return f"""
Analyze {title} data for {len(features)} neighborhoods. You have access to tools that can:

1. Calculate descriptive statistics (mean, median, standard deviation, min, max, quartiles)
2. Analyze {title} density patterns using dynamic quartile-based thresholds
3. Evaluate maintenance quality levels using data-driven thresholds
4. Assess park size distribution with dynamic pattern analysis
5. Calculate accessibility scores with dynamic weighting
6. Identify natural clusters using enhanced statistical patterns
7. Compare neighborhoods with dynamic outlier detection
8. Generate dynamic tags based on statistical patterns
9. Analyze spatial correlations between {title} and other urban features

Available neighborhood names: {neighborhood_names}
Available data columns: {list(features.columns)}

PREDEFINED TAG RULES (use these exact tags when conditions are met):
{tag_rules}

Task: Use the available tools to analyze this neighborhood data and assign appropriate tags based on the predefined rules.

Think step by step and use the tools to:
1. Calculate statistical distributions and dynamic thresholds for each metric
2. Apply the predefined tag rules to each neighborhood using data-driven comparisons
3. Generate additional dynamic tags based on statistical patterns
4. Create JSON output with tags, its format should similar to this {{ 'neighborhoods': {{ 'Neighborhood A': ['#high-green-density', '#well-maintained', '#large-parks'] }}, }}

IMPORTANT: You MUST follow the ReAct format EXACTLY:
- Thought: Think about what to do next
- Action: Use one of the available tools
- Action Input: Provide input to the tool
- Observation: Get the result from the tool
- Repeat until you have enough information
- Final Answer: Provide the structured JSON output

DO NOT output JSON directly without going through the Thought->Action->Observation process first.
You MUST use the tools to analyze the data before providing your final answer.

When you have completed your analysis and are ready to provide the final answer, use the JSON format specified in the base prompt.

Remember to use the tools strategically and follow the ReAct format for your responses.
"""
    
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment neighborhoods using ReAct agent reasoning"""
        analysis_prompt = self._create_analysis_prompt(features)
        
        try:
            logger.info(f"Invoking agent with prompt length: {len(analysis_prompt)}")
            result = self.agent_executor.invoke({
                "input": analysis_prompt,
                "chat_history": []
            })
            
            logger.info("Agent execution completed successfully")
            # Parse agent output to extract tags
            return self._parse_agent_output(result["output"], features)
            
        except Exception as e:
            logger.error(f"ReAct agent error: {e}")
            import traceback
            logger.error(traceback.format_exc())

            return self._get_fallback_tags(features)
    
    
    
    # Common statistical tools that can be reused across modules
    def _calculate_statistics(self, data: pd.Series) -> dict:
        """Calculate descriptive statistics for numerical data (reusable)"""
        stats = {
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "count": int(len(data)),
            "q1": float(data.quantile(0.25)),
            "q3": float(data.quantile(0.75)),
            "iqr": float(data.quantile(0.75) - data.quantile(0.25))
        }
        
        # Add skewness and kurtosis if there are enough data points
        if len(data) > 2:
            stats["skewness"] = float(data.skew())
            stats["kurtosis"] = float(data.kurtosis())
        else:
            # For small datasets, set to 0 (approximately normal)
            stats["skewness"] = 0.0
            stats["kurtosis"] = 0.0
            
        return stats
    
    def _analyze_distribution(self, data: pd.Series) -> dict:
        """Analyze data distribution patterns (reusable)"""
        stats = self._calculate_statistics(data)
        
        # Dynamic threshold calculation
        mean = stats["mean"]
        std = stats["std"]
        q1 = stats["q1"]
        q3 = stats["q3"]
        
        return {
            "statistics": stats,
            "outlier_threshold_low": float(q1 - 1.5 * (q3 - q1)),
            "outlier_threshold_high": float(q3 + 1.5 * (q3 - q1)),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
            "distribution_type": self._classify_distribution(data)
        }
    
    def _classify_distribution(self, data: pd.Series) -> str:
        """Classify data distribution type (reusable)"""
        skewness = data.skew()
        if abs(skewness) > 1:
            return "highly_skewed"
        elif abs(skewness) > 0.5:
            return "moderately_skewed"
        else:
            return "approximately_normal"
    
    def _compare_to_reference(self, data: pd.Series, reference_data: pd.Series = None) -> dict:
        """Compare data to reference distribution (reusable)"""
        if reference_data is None:
            reference_data = data
            
        stats = self._calculate_statistics(data)
        ref_stats = self._calculate_statistics(reference_data)
        
        return {
            "current_stats": stats,
            "reference_stats": ref_stats,
            "comparison": {
                "mean_difference": float(stats["mean"] - ref_stats["mean"]),
                "mean_ratio": float(stats["mean"] / ref_stats["mean"]) if ref_stats["mean"] != 0 else float('inf'),
                "std_ratio": float(stats["std"] / ref_stats["std"]) if ref_stats["std"] != 0 else float('inf'),
                "effect_size": float((stats["mean"] - ref_stats["mean"]) / ref_stats["std"]) if ref_stats["std"] != 0 else float('inf')
            }
        }
    
    def _identify_patterns(self, data: pd.Series) -> dict:
        """Identify patterns and trends in data (reusable)"""
        stats = self._calculate_statistics(data)
        
        # Dynamic pattern detection
        patterns = []
        if stats["std"] / stats["mean"] < 0.1:
            patterns.append("low_variability")
        if stats["skewness"] > 1:
            patterns.append("right_skewed")
        elif stats["skewness"] < -1:
            patterns.append("left_skewed")
        if stats["kurtosis"] > 3:
            patterns.append("leptokurtic")
        elif stats["kurtosis"] < 3:
            patterns.append("platykurtic")
            
        return {
            "statistics": stats,
            "patterns": patterns,
            "variability_level": "high" if stats["std"] / stats["mean"] > 0.5 else "medium" if stats["std"] / stats["mean"] > 0.2 else "low"
        }

    def _extract_json_from_string(self, text: str) -> dict:
        """Extract JSON from text, handling markdown code blocks and malformed JSON"""
        import json
        import re
        
        # Remove markdown code block markers if present
        cleaned_text = re.sub(r'```json\n|\n```|```', '', text)
        
        # Try to find JSON content using multiple patterns
        json_patterns = [
            r'\{.*\}',  # Basic JSON object pattern
            r'Final Answer:\s*(\{.*\})',  # Final Answer: {json}
            r'Final Answer\s*(\{.*\})',  # Final Answer {json}
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_text, re.DOTALL)
            for match in matches:
                try:
                    if isinstance(match, str):
                        return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # If no pattern matched, try to parse the entire cleaned text
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty dict
            logger.warning(f"Failed to extract JSON from text: {cleaned_text[:200]}...")
            return {}

    def _generate_dynamic_tags(self, features_df: pd.DataFrame, numeric_columns: List[str] = None) -> dict:
        """Generate dynamic tags based on statistical patterns for any numeric columns"""
        dynamic_tags = {}
        
        # Use provided numeric columns or auto-detect
        if numeric_columns is None:
            numeric_columns = features_df.select_dtypes(include=['number']).columns.tolist()
        
        for col in numeric_columns:
            if col in features_df.columns:
                col_data = features_df[col].dropna()
                if len(col_data) > 0:
                    stats = self._calculate_statistics(col_data)
                    patterns = self._identify_patterns(col_data)
                    
                    # Generate dynamic tags based on statistical characteristics
                    tags = []
                    
                    # Variability tags
                    variability = patterns["variability_level"]
                    tags.append(f"#{variability}-variability")
                    
                    # Distribution shape tags
                    if "right_skewed" in patterns["patterns"]:
                        tags.append("#right-skewed-distribution")
                    elif "left_skewed" in patterns["patterns"]:
                        tags.append("#left-skewed-distribution")
                    
                    # Outlier detection
                    q1, q3 = stats["q1"], stats["q3"]
                    iqr = stats["iqr"]
                    outlier_low = q1 - 1.5 * iqr
                    outlier_high = q3 + 1.5 * iqr
                    
                    outlier_count = len(col_data[(col_data < outlier_low) | (col_data > outlier_high)])
                    if outlier_count > 0:
                        tags.append(f"#has-outliers-{outlier_count}")
                    
                    # Value range tags
                    if stats["max"] - stats["min"] > 2 * stats["std"]:
                        tags.append("#wide-value-range")
                    
                    dynamic_tags[col] = {
                        "statistics": stats,
                        "patterns": patterns["patterns"],
                        "generated_tags": tags
                    }
        
        return dynamic_tags

    def _analyze_correlations(self, features_df: pd.DataFrame, primary_columns: List[str], exclude_columns: List[str] = None) -> dict:
        """Analyze correlations between primary columns and other numeric features"""
        if exclude_columns is None:
            exclude_columns = ['district', 'id']
        
        other_cols = [col for col in features_df.columns
                     if col not in primary_columns + exclude_columns
                     and pd.api.types.is_numeric_dtype(features_df[col])]
        
        correlations = {}
        for primary_col in primary_columns:
            if primary_col in features_df.columns:
                primary_data = features_df[primary_col].dropna()
                if len(primary_data) > 0:
                    corr_results = {}
                    for other_col in other_cols:
                        if other_col in features_df.columns:
                            other_data = features_df[other_col].dropna()
                            if len(other_data) > 0 and len(primary_data) == len(other_data):
                                try:
                                    correlation = primary_data.corr(other_data)
                                    corr_results[other_col] = {
                                        "correlation": float(correlation),
                                        "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak",
                                        "direction": "positive" if correlation > 0 else "negative"
                                    }
                                except:
                                    corr_results[other_col] = {"error": "Could not calculate correlation"}
                    
                    correlations[primary_col] = corr_results
        
        return correlations
    

    def _compare_neighborhoods(self, features_df: pd.DataFrame, numeric_cols) -> dict:
        """Compare neighborhoods across multiple metrics with enhanced analysis"""
        comparisons = {}
        
        
        for col in numeric_cols:
            col_data = features_df[col].dropna()
            if len(col_data) > 0:
                stats = self._calculate_statistics(col_data)
                distribution = self._analyze_distribution(col_data)
                
                # Dynamic outlier detection
                q1, q3 = stats["q1"], stats["q3"]
                iqr = stats["iqr"]
                outlier_threshold_low = q1 - 1.5 * iqr
                outlier_threshold_high = q3 + 1.5 * iqr
                
                outliers_high = features_df[features_df[col] > outlier_threshold_high]
                outliers_low = features_df[features_df[col] < outlier_threshold_low]
                
                top_performers = features_df.nlargest(3, col)[['district', col]]
                bottom_performers = features_df.nsmallest(3, col)[['district', col]]
                
                comparisons[col] = {
                    "top_performers": top_performers.to_dict('records'),
                    "bottom_performers": bottom_performers.to_dict('records'),
                    "outliers_high": outliers_high[['district', col]].to_dict('records'),
                    "outliers_low": outliers_low[['district', col]].to_dict('records'),
                    "range": float(stats["max"] - stats["min"]),
                    "statistics": stats,
                    "distribution": distribution,
                    "outlier_thresholds": {
                        "low": float(outlier_threshold_low),
                        "high": float(outlier_threshold_high)
                    }
                }
        
        return {
            "neighborhood_comparisons": comparisons,
            "interpretation": "Enhanced comparison with dynamic outlier detection and statistical analysis"
        }
     # Wrapper methods for LangChain tool compatibility
    def _calculate_statistics_wrapper(self, column_name: str) -> dict:
        """Wrapper for calculate_statistics that accepts string input"""
        # Clean up column name - remove any extra quotes
        clean_column_name = column_name.strip().strip("'\"")
        
        if hasattr(self, 'features_df') and clean_column_name in self.features_df.columns:
            return self._calculate_statistics(self.features_df[clean_column_name])
        return {"error": f"Column '{clean_column_name}' not found in features data. Available columns: {list(self.features_df.columns) if hasattr(self, 'features_df') else 'None'}"}

    def _create_generic_tools(self):
        """Create generic statistical tools that can be used across different domains"""
        return [
            Tool(
                name="calculate_statistics",
                func=self._calculate_statistics_wrapper,
                description="Calculate descriptive statistics (mean, median, std, min, max, quartiles) for numerical data. Input should be a string with the column name."
            ),
            Tool(
                name="identify_clusters",
                func=self._identify_clusters_wrapper,
                description="Identify natural clusters using enhanced statistical patterns with dynamic boundaries and correlation analysis. Input should be 'all' to analyze all numeric columns."
            ),
            Tool(
                name="analyze_correlations",
                func=self._analyze_spatial_correlations_wrapper,
                description="Analyze correlations between features for cross-feature insights. Input should be 'all' to analyze all correlations."
            ),
            Tool(
                name="compare_neighborhoods",
                func=self._compare_neighborhoods_wrapper,
                description="Compare neighborhoods across multiple metrics with dynamic outlier detection and statistical analysis. Input should be a string with the column name like 'maintenance_score'."
            ),
            Tool(
                name="generate_dynamic_tags",
                func=self._generate_dynamic_tags_wrapper,
                description="Generate dynamic tags based on statistical patterns and data characteristics. Input should be 'all' to analyze all numeric columns."
            )
        ]
    

    @abstractmethod
    def _create_tools(self):
        """Create domain-specific analysis tools"""
        pass


    def _create_all_tools(self):
        """Create custom tools for analysis by merging generic and specific tools"""
        return self._create_generic_tools() + self._create_tools()
    
    def _create_density_analysis_tool(self, tool_name: str, description: str, analysis_method: callable) -> Tool:
        """Create a density analysis tool with standardized wrapper pattern"""
        def density_wrapper(column_name: str) -> dict:
            """Wrapper for density analysis that accepts string input"""
            # Clean up column name - remove any extra quotes
            clean_column_name = column_name.strip().strip("'\"")
            
            if hasattr(self, 'features_df') and clean_column_name in self.features_df.columns:
                return analysis_method(self.features_df[clean_column_name])
            return {"error": f"Column '{clean_column_name}' not found in features data. Available columns: {list(self.features_df.columns) if hasattr(self, 'features_df') else 'None'}"}
        
        return Tool(
            name=tool_name,
            func=density_wrapper,
            description=description
        )
    
    def _analyze_density(self, per_capita_data: pd.Series, density_type: str = "density") -> dict:
        """Generic density analysis with dynamic thresholds (reusable across all modules)"""
        stats = self._calculate_statistics(per_capita_data)
        distribution = self._analyze_distribution(per_capita_data)
        
        # Dynamic threshold calculation based on data distribution
        q1, q3 = stats["q1"], stats["q3"]
        iqr = stats["iqr"]
        
        # Calculate density levels based on quartiles
        if stats["median"] > q3 + iqr * 0.5:
            density_level = "very_high"
        elif stats["median"] > q3:
            density_level = "high"
        elif stats["median"] > q1:
            density_level = "medium"
        else:
            density_level = "low"
            
        return {
            "density_level": density_level,
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Overall {density_level.replace('_', ' ')} {density_type} based on dynamic quartile analysis",
            "dynamic_thresholds": {
                "very_high_threshold": float(q3 + iqr * 0.5),
                "high_threshold": float(q3),
                "medium_threshold": float(q1),
                "low_threshold": float(q1 - iqr * 0.5)
            }
        }
    
    def _create_quality_assessment_tool(self, tool_name: str, description: str, assessment_method: callable) -> Tool:
        """Create a quality assessment tool with standardized wrapper pattern"""
        def quality_wrapper(column_name: str) -> dict:
            """Wrapper for quality assessment that accepts string input"""
            # Clean up column name - remove any extra quotes
            clean_column_name = column_name.strip().strip("'\"")
            
            if hasattr(self, 'features_df') and clean_column_name in self.features_df.columns:
                return assessment_method(self.features_df[clean_column_name])
            return {"error": f"Column '{clean_column_name}' not found in features data. Available columns: {list(self.features_df.columns) if hasattr(self, 'features_df') else 'None'}"}
        
        return Tool(
            name=tool_name,
            func=quality_wrapper,
            description=description
        )
    
    def _evaluate_quality(self, quality_data: pd.Series, quality_type: str = "quality") -> dict:
        """Generic quality evaluation with dynamic thresholds (reusable across all modules)"""
        stats = self._calculate_statistics(quality_data)
        distribution = self._analyze_distribution(quality_data)
        
        # Dynamic quality thresholds based on data distribution
        q1, q3 = stats["q1"], stats["q3"]
        
        if stats["median"] > q3 + (q3 - q1) * 0.5:
            quality_level = "excellent"
        elif stats["median"] > q3:
            quality_level = "good"
        elif stats["median"] > q1:
            quality_level = "fair"
        else:
            quality_level = "poor"
            
        return {
            "quality_level": quality_level,
            "statistics": stats,
            "distribution_analysis": distribution,
            "interpretation": f"Overall {quality_level} {quality_type} based on dynamic quartile analysis",
            "dynamic_thresholds": {
                "excellent_threshold": float(q3 + (q3 - q1) * 0.5),
                "good_threshold": float(q3),
                "fair_threshold": float(q1),
                "poor_threshold": float(q1 - (q3 - q1) * 0.5)
            }
        }