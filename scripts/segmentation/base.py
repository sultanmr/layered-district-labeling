from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import google.generativeai as genai
import os
import re
import time
import hashlib
import json
from pathlib import Path
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class DataLoader(ABC):
    """Base class for loading data from different sources"""
    
    # Class-level cache for storing query results
    _cache: Dict[str, Dict] = {}
    
    def __init__(self):
        """Initialize DataLoader and load cache from disk"""
        self._load_cache_from_disk()
        self.eingine = None
    
    @property
    @abstractmethod
    def query(self) -> str:
        """Return the SQL query for this data source"""
        pass
    
    def _get_cache_path(self) -> Path:
        """Get the cache file path from environment variable"""
        #cache_path = os.getenv("CACHE_PATH", "cache/data_loader_cache.json")
        cache_path = Path(os.getenv("CACHE_PATH", "cache/data_loader_cache.json"))
        # Create the directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    

    def _get_timeout(self):
        default_timeout = 12 * 60 * 60
        time_out_str = os.getenv("TIME_OUT")
        if time_out_str:
            try:
                # Safely evaluate numeric expressions like '12 * 60 * 60'
                return int(eval(time_out_str, {"__builtins__": {}}))
            except Exception:
                return default_timeout
        return default_timeout

    
    def _load_cache_from_disk(self):
        """Load cache from disk file"""
        cache_path = self._get_cache_path()
        try:
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    self._cache = json.load(f)
                    # Convert string timestamps back to float
                    for key, entry in self._cache.items():
                        if 'timestamp' in entry:
                            self._cache[key]['timestamp'] = float(entry['timestamp'])
                logging.info(f"✓ Loaded cache from {cache_path}")
            else:
                self._cache = {}
                logging.error(f"✓ No cache file found at {cache_path}, starting with empty cache")
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"⚠️ Error loading cache from {cache_path}: {e}")
            self._cache = {}
    
    def _save_cache_to_disk(self):
        """Save cache to disk file"""
        cache_path = self._get_cache_path()
        try:
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to JSON-serializable format
            cache_to_save = {}
            for key, entry in self._cache.items():
                cache_to_save[key] = {
                    'timestamp': str(entry.get('timestamp', 0)),
                    'query': entry.get('query', ''),
                    'data': entry.get('data', {}).to_dict() if isinstance(entry.get('data'), pd.DataFrame) else {}
                }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_to_save, f, indent=2)
            logging.info(f"✓ Saved cache to {cache_path}")
        except (IOError, TypeError) as e:
            logging.error(f"⚠️ Error saving cache to {cache_path}: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a unique cache key for the query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cached data is still valid (within 12 hours)"""
        if self.eingine is None:
            return True
        
        current_time = time.time()
        cache_time = cache_entry.get('timestamp', 0)
        return (current_time - cache_time) < self._get_timeout() 
        
    def _get_from_cache(self, query: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available and valid"""
        cache_key = self._get_cache_key(query)
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                # Convert dict back to DataFrame if needed
                if isinstance(cache_entry.get('data'), dict):
                    cache_entry['data'] = pd.DataFrame(cache_entry['data'])
                return cache_entry['data'].copy()
        return None
        
    def _save_to_cache(self, query: str, data: pd.DataFrame):
        """Save query result to cache with timestamp and persist to disk"""
        cache_key = self._get_cache_key(query)
        self._cache[cache_key] = {
            'timestamp': time.time(),
            'data': data.copy(),
            'query': query
        }
        # Persist cache to disk
        self._save_cache_to_disk()
        
    def load_data(self, engine) -> pd.DataFrame:
        """Load and return raw data as DataFrame using provided engine
        
        First checks if data for this query is available in cache and still valid
        (within 12 hours). If cached data exists and is valid, returns it directly.
        Otherwise, executes the query and saves result to cache.
        """
        self.eingine = engine
        # Check cache first
        logging.info("Checking cache for data...")
        cached_data = self._get_from_cache(self.query)
        if cached_data is not None and not cached_data.empty:
            logging.info("✓ Retrieving from cache")
            return cached_data
            
        # If not in cache or cache expired, execute query
        result = pd.read_sql(self.query, engine)
        
        # Save to cache for future use
        self._save_to_cache(self.query, result)
        
        return result

class FeatureProcessor(ABC):
    """Base class for processing raw data into features"""
    
    @abstractmethod
    def process_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into features"""
        pass

class SegmentationStrategy(ABC):
    """Interface for segmentation approaches"""
    
    @abstractmethod
    def segment(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment neighborhoods based on features"""
        pass

class BaseClusterer(ABC):
    """Base class for clustering implementations"""
    
    @abstractmethod
    def process(self, features_df: pd.DataFrame, engine=None) -> pd.DataFrame:
        """Main processing method that returns cluster labels DataFrame"""
        pass

class BaseTagger(ABC):
    """Base class for tagging implementations"""
    
    @abstractmethod
    def calculate_tags(self, engine) -> pd.DataFrame:
        """Calculate and return tags DataFrame"""
        pass

class ResultAggregator:
    """Combines results from multiple segmentation approaches"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name, result: Dict[str, List[str]]):
        """Add segmentation result to aggregator"""
        if name not in self.results.keys():
            self.results[name] = {}
        self.results[name].update(result)
        #self.results.append(result)
    
    def aggregate(self) -> pd.DataFrame:
        """Combine all segmentation results into final output"""
        if not self.results:
            return pd.DataFrame(columns=['table_name', 'district', 'hashtags'])
            
        # Convert list of dicts to DataFrame
        combined = []
        for name, result in self.results.items():
            for district, segments in result.items():
                combined.append({
                    'table_name': name,
                    'district': district,
                    'hashtags': ','.join(segments)
                })
        
        return pd.DataFrame(combined)

class AgentMessage:
    """Base class for agent communication messages"""
    def __init__(self, sender: str, content: dict):
        self.sender = sender
        self.content = content

class AgentInterface(ABC):
    """Base interface for agentic components"""
    
    @abstractmethod
    def receive_message(self, message: AgentMessage):
        """Handle incoming message from another agent"""
        pass
    
    @abstractmethod
    def send_message(self, recipient: str, content: dict):
        """Send message to another agent"""
        pass

class MessageBus:
    """Central communication bus for agents"""
    def __init__(self):
        self.agents = {}
    
    def register_agent(self, agent_id: str, agent: AgentInterface):
        """Register an agent with the message bus"""
        self.agents[agent_id] = agent
    
    def route_message(self, sender: str, recipient: str, content: dict):
        """Route message between agents"""
        if recipient in self.agents:
            message = AgentMessage(sender, content)
            self.agents[recipient].receive_message(message)


# class TagsResponse(BaseModel):
#     """Pydantic model for Gemini report"""
#     neighborhood: str = Field(description="Name of the neighborhood")
#     tags: List[str] = Field(description="List of generated hashtags")
#     summary: Optional[str] = Field(default=None, description="Summary of the analysis")

class LLMSegmentationStrategy(SegmentationStrategy):
    """Base class for LLM-powered segmentation strategies"""
    
    def __init__(self, init_gemini: bool = True):
        
        if init_gemini:
            self.llm = self._initialize_gemini_llm()

    def _initialize_gemini_llm(self):
        """Initialize and return Gemini LLM instance"""
        api_key = self.get_api_key("GEMINI_API_KEY")
        # Configure Gemini
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def get_api_key(self, key_name):
        api_key = os.getenv(key_name)
        if not api_key:
            raise ValueError(
                f"{key_name} not found"
                f"Please set {key_name} environment variable."
            )
        return api_key
    

    def _create_langchain_gemini(self):
        """Create LangChain compatible Gemini LLM instance"""
        api_key = self.get_api_key("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.1
        )
    
    def _create_langchain_openai(self):
        """Create LangChain compatible OpenAI LLM instance"""
        api_key = self.get_api_key("OPENAI_API_KEY")
        return ChatOpenAI(
            model="gpt-4o-mini",  # or "gpt-4o", "gpt-4-turbo", etc.
            api_key=api_key,
            temperature=0.1
        )
    
    def _create_langchain_deepseek(self):
        """Create LangChain compatible DeepSeek LLM instance"""
        api_key = self.get_api_key("DEEPSEEK_API_KEY")
        
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.1
        )
    



    def _normalize_tag(self, tag: str) -> str:
        """Ensure tag has exactly one # and no starting '-'"""
        tag = tag.strip().lower()
        tag = re.sub(r'^-+', '', tag)
        tag = tag.replace(" ", "-")
        tag = re.sub(r'#+', '#', tag)
        if '#' not in tag:
            tag = '#' + tag
        return tag
    
    def _get_tags(self, prompt: str) -> list:
        """Get tags from Gemini"""
        try:
            response = self.llm.generate_content(prompt)
            
            # parser = PydanticOutputParser(pydantic_object=TagsResponse)
            # response = parser.parse(response.text)
            # if not response.tags:
            #     return ["#cluster-fallback"]
            # return response.tags

            tags = response.text.strip().lower().split(",")
            return [self._normalize_tag(tag) for tag in tags]
        except Exception as e:
            logging.error(f"error: {e}")
            return ["#cluster-fallback"]
