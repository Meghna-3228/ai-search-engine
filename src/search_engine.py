# search_engine.py

"""
Advanced Search Engine with Neural-Symbolic and Knowledge Graph Integration

This module provides the core search functionality, orchestrating query processing,
provider selection, caching, and result analysis.

Key Features:
- Abstract SearchProvider interface
- Concrete implementations for Cohere, OpenAI, Anthropic, Web API
- Integration with other managers (Config, API, Cache, Query, Analyzer)
- Neural-symbolic integration for logical reasoning
- Knowledge graph construction and augmentation
- Caching logic (get/set)
- Fallback provider logic
- Asynchronous search execution
- Ability to list available providers
- Proactive Exploration Suggestion Generation
"""

import os
import re
import json
import time
import random
import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from datetime import datetime
from functools import lru_cache, wraps
import httpx
import copy # +++ Added copy +++

# Import local modules - Use singleton accessors
from config_manager import get_config_manager, ConfigManager
from api_manager import get_api_manager, APIManager, COHERE_AVAILABLE, OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE
from query_processor import get_query_processor, QueryProcessor, get_context_manager, QueryContextManager, process_and_contextualize_query
from cache_manager import get_search_cache, SearchCache, CacheEntry

# Optional external dependencies
if COHERE_AVAILABLE:
    import cohere
if OPENAI_AVAILABLE:
    import openai
if ANTHROPIC_AVAILABLE:
    import anthropic

# Neural-Symbolic and Knowledge Graph dependencies
NEURO_SYMBOLIC_AVAILABLE = False
KNOWLEDGE_GRAPH_AVAILABLE = False
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
    from sympy import symbols, Eq, solve # type: ignore
    from transformers import AutoTokenizer, AutoModel
    NEURO_SYMBOLIC_AVAILABLE = True
    KNOWLEDGE_GRAPH_AVAILABLE = True # Assume KG uses networkx etc.
except ImportError:
    logging.warning("Neural-symbolic or KG dependencies missing (scikit-learn, spacy, sympy, transformers, networkx, torch, tensorflow). Some features disabled.")
    logging.warning("Run: python -m spacy download en_core_web_sm (or en_core_web_lg)")

# Configure logging
logger = logging.getLogger("search_engine")


class SearchProviderError(Exception):
    """Custom exception for search provider errors."""
    pass

# --- Neural-Symbolic Processor ---
class NeuroSymbolicProcessor:
    """Handles neuro-symbolic integration for search queries"""
    def __init__(self):
        logger.info("Initializing NeuroSymbolicProcessor...")
        if not NEURO_SYMBOLIC_AVAILABLE:
            raise ImportError("Required libraries for NeuroSymbolicProcessor not found.")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
            logger.info("NeuroSymbolicProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NeuroSymbolicProcessor: {e}", exc_info=True)
            raise

    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate BERT embeddings for text"""
        if not NEURO_SYMBOLIC_AVAILABLE: return np.array([])
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
        except Exception as e:
            logger.error(f"Error generating embedding for text: '{text[:50]}...': {e}", exc_info=False)
            return np.array([])


    def logical_reasoning(self, query: str, documents: List[Dict]) -> List[float]:
        """Apply symbolic reasoning and semantic similarity to score documents."""
        if not NEURO_SYMBOLIC_AVAILABLE: return [1.0] * len(documents)
        logger.info(f"Applying logical reasoning to query: '{query[:100]}...'")
        scores = []
        try:
            query_embedding = self.generate_embeddings(query)
            if query_embedding.size == 0: # Handle embedding failure
                 logger.warning("Query embedding failed, cannot calculate semantic similarity.")
                 return [0.0] * len(documents) # Or some default score?

            constraints = self._extract_constraints(query)
            logger.debug(f"Extracted constraints: {constraints}")

            for doc in documents:
                base_score = 1.0
                doc_text = doc.get('text', '') or doc.get('snippet', '')
                if not doc_text:
                    scores.append(0.0)
                    continue

                # Apply symbolic reasoning constraints
                if constraints:
                    for var_str, operation in constraints.items():
                        if var_str in doc_text: # Simple check for var name presence
                            # Placeholder for complex value extraction and evaluation logic
                            base_score *= 1.1 # Boost if related variable mentioned

                # Apply semantic similarity
                doc_embedding = self.generate_embeddings(doc_text)
                if doc_embedding.size == 0: # Handle doc embedding failure
                    semantic_similarity = 0.0
                else:
                     # Ensure embeddings are 2D for cosine_similarity
                     if doc_embedding.ndim == 1: doc_embedding = doc_embedding.reshape(1, -1)
                     query_embedding_2d = query_embedding.reshape(1, -1)
                     semantic_similarity = cosine_similarity(query_embedding_2d, doc_embedding)[0][0]

                final_score = base_score * float(max(0, semantic_similarity)) # Ensure non-negative
                logger.debug(f"Document score: {final_score:.4f} (base: {semantic_similarity:.4f}, modifier: {base_score:.2f})")
                scores.append(final_score)

        except Exception as e:
            logger.error(f"Error during logical reasoning: {e}", exc_info=True)
            return [0.0] * len(documents) # Return zero scores on error

        return scores

    def _extract_constraints(self, text: str) -> Dict:
        """Extract mathematical constraints from natural language (Simplified)."""
        if not NEURO_SYMBOLIC_AVAILABLE: return {}
        constraints = {}
        # Patterns need careful crafting and validation
        patterns = {
            'equation': r'(\b[a-zA-Z]+\b)\s*(?:is equal to|=)\s*([+-]?\d+(?:\.\d+)?)',
            'comparison': r'(\b[a-zA-Z]+\b)\s*(?:is greater than|>|is less than|<|>=|<=)\s*([+-]?\d+(?:\.\d+)?)'
        }
        try:
            # Equations
            for match in re.finditer(patterns['equation'], text, re.IGNORECASE):
                var, val_str = match.groups()
                try: constraints[var] = Eq(symbols(var), float(val_str))
                except (ValueError, TypeError) as parse_err: logger.debug(f"Constraint parse error: {parse_err}")
            # Comparisons
            for match in re.finditer(patterns['comparison'], text, re.IGNORECASE):
                 var, op_text, val_str = match.groups()
                 try:
                     var_sym = symbols(var)
                     val_float = float(val_str)
                     op_symbol = op_text.strip().lower()
                     if op_symbol in ['is greater than', '>']: constraints[var] = (var_sym > val_float)
                     elif op_symbol in ['is less than', '<']: constraints[var] = (var_sym < val_float)
                     elif op_symbol in ['>=']: constraints[var] = (var_sym >= val_float)
                     elif op_symbol in ['<=']: constraints[var] = (var_sym <= val_float)
                 except (ValueError, TypeError) as parse_err: logger.debug(f"Constraint parse error: {parse_err}")
        except Exception as e:
            logger.error(f"Error extracting constraints: {e}", exc_info=True)
        return constraints

# --- Knowledge Graph Builder (Placeholder/Temp Use) ---
class KnowledgeGraphBuilder:
    """Builds temporary knowledge graphs from search results."""
    def __init__(self):
        logger.info("Initializing KnowledgeGraphBuilder...")
        if not KNOWLEDGE_GRAPH_AVAILABLE:
            raise ImportError("NetworkX required for KnowledgeGraphBuilder")
        self.graph = nx.DiGraph()
        self.entity_cache = {} # For embeddings used in attention
        try: self.nlp = spacy.load("en_core_web_sm")
        except Exception: logger.error("Failed to load spaCy model for KG Builder"); raise
        logger.info("KnowledgeGraphBuilder initialized successfully")

    # ... (build_graph, graph_attention, _extract_entities, _add_to_graph - Implementations need review for robustness if used) ...
    # These methods likely duplicate logic now handled by the QueryProcessor's persistent KG
    # Keep the class definition for now if SearchEngine explicitly uses it, otherwise consider removing.

# --- Abstract Base Class for Providers ---
class SearchProvider:
    """Abstract base class for search providers."""
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.api_manager: APIManager = get_api_manager()
        logger.info(f"Initializing SearchProvider: {self.name}")

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement the async search() method.")

    def get_provider_info(self) -> Dict[str, Any]:
        available = True
        safe_config = {k: v for k, v in self.config.items() if 'key' not in k.lower() and 'token' not in k.lower()}
        return {"name": self.name, "available": available, "config": safe_config}


# --- Concrete Provider Implementations ---
# CohereSearchProvider, OpenAISearchProvider, AnthropicSearchProvider, WebApiSearchProvider
# (Keep existing implementations - Assume they return the correct success/error dictionary structure)
class CohereSearchProvider(SearchProvider):
    """Search provider using the Cohere API (Chat endpoint with web search connector)."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("cohere", config)
        self.client: Optional['cohere.Client'] = None
        self.is_available: bool = False
        if COHERE_AVAILABLE:
            try:
                self.client = self.api_manager.get_cohere_client()
                if self.client: self.is_available = True
                else: logger.warning("Cohere client init failed (API Manager).")
            except Exception as e: logger.error(f"Cohere client error: {e}.", exc_info=True)
        else: logger.warning("Cohere package not installed.")
        if self.is_available: logger.info("CohereSearchProvider initialized.")

    def get_provider_info(self) -> Dict[str, Any]:
        info = super().get_provider_info(); info['available'] = self.is_available; return info

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        if not self.is_available or not self.client:
            return {'provider': self.name, 'success': False, 'text': "Cohere provider unavailable.", 'documents': [], 'error': "Client not init.", 'raw_response': None, 'execution_time': 0}
        model = kwargs.get("model", self.config.get("model", "command-r"))
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.3))
        connector_id = kwargs.get("connector_id", self.config.get("connector_id", "web-search"))
        logger.info(f"Cohere search (Model: {model}, Conn: {connector_id}): '{query[:100]}...'")
        try:
            if not hasattr(self.client, 'chat') or not callable(self.client.chat): raise SearchProviderError("Client missing 'chat' method.")
            response = await asyncio.to_thread(self.client.chat, message=query, model=model, temperature=temperature, connectors=[{"id": connector_id}])
            if response is None: raise SearchProviderError("API returned null response.")
            exec_time = time.time() - start_time; logger.info(f"Cohere success ({exec_time:.2f}s).")
            docs = []
            if hasattr(response, 'documents') and response.documents:
                for doc in response.documents:
                    if isinstance(doc, dict): docs.append(doc)
                    elif hasattr(doc, 'dict'): docs.append(doc.dict())
                    else: logger.warning(f"Skipping non-dict Cohere doc: {type(doc)}")
            return {'provider': self.name, 'success': True, 'text': getattr(response, 'text', ''), 'documents': docs, 'error': None, 'raw_response': response, 'execution_time': exec_time}
        except Exception as e:
            exec_time = time.time() - start_time; logger.error(f"Cohere search failed ({exec_time:.2f}s): {e}", exc_info=True)
            err_type = type(e).__name__; err_msg = str(e)
            if COHERE_AVAILABLE and isinstance(e, cohere.errors.CohereAPIError): err_msg = f"Cohere API Error: {e.http_status} - {e.message}"
            return {'provider': self.name, 'success': False, 'text': f"Failed: {err_msg}", 'documents': [], 'error': f"{err_type}: {err_msg}", 'raw_response': None, 'execution_time': exec_time}

class OpenAISearchProvider(SearchProvider):
    """Search provider using OpenAI API (ChatCompletion)."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("openai", config)
        self.async_client: Optional['openai.AsyncOpenAI'] = None
        self.is_available: bool = False
        if OPENAI_AVAILABLE:
            try:
                api_key = self.api_manager.get_api_key("openai")
                if api_key: self.async_client = openai.AsyncOpenAI(api_key=api_key); self.is_available = True
                else: logger.warning("OpenAI API key not found.")
            except Exception as e: logger.error(f"OpenAI client error: {e}.", exc_info=True)
        else: logger.warning("OpenAI package not installed.")
        if self.is_available: logger.info("OpenAISearchProvider initialized (async).")

    def get_provider_info(self) -> Dict[str, Any]:
        info = super().get_provider_info(); info['available'] = self.is_available; return info

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        if not self.is_available or not self.async_client:
            return {'provider': self.name, 'success': False, 'text': "OpenAI provider unavailable.", 'documents': [], 'error': "Client not init.", 'raw_response': None, 'execution_time': 0}
        model = kwargs.get("model", self.config.get("model", "gpt-3.5-turbo"))
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.3))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 1500))
        sys_prompt = kwargs.get("system_prompt", "You are a helpful search assistant.")
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]
        logger.info(f"OpenAI search (Model: {model}): '{query[:100]}...'")
        try:
            response = await self.async_client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            exec_time = time.time() - start_time
            if response and response.choices:
                result_text = response.choices[0].message.content
                logger.info(f"OpenAI success ({exec_time:.2f}s).")
                return {'provider': self.name, 'success': True, 'text': result_text.strip() if result_text else "", 'documents': [], 'error': None, 'raw_response': response, 'execution_time': exec_time}
            else: raise SearchProviderError("API returned empty/invalid response.")
        except Exception as e:
            exec_time = time.time() - start_time; logger.error(f"OpenAI search failed ({exec_time:.2f}s): {e}", exc_info=True)
            err_type = type(e).__name__; err_msg = str(e)
            if OPENAI_AVAILABLE and isinstance(e, openai.APIError): err_msg = f"OpenAI API Error: {e}"
            return {'provider': self.name, 'success': False, 'text': f"Failed: {err_msg}", 'documents': [], 'error': f"{err_type}: {err_msg}", 'raw_response': None, 'execution_time': exec_time}

class AnthropicSearchProvider(SearchProvider):
    """Search provider using Anthropic API (Messages)."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("anthropic", config)
        self.async_client: Optional['anthropic.AsyncAnthropic'] = None
        self.is_available: bool = False
        if ANTHROPIC_AVAILABLE:
            try:
                api_key = self.api_manager.get_api_key("anthropic")
                if api_key: self.async_client = anthropic.AsyncAnthropic(api_key=api_key); self.is_available = True
                else: logger.warning("Anthropic API key not found.")
            except Exception as e: logger.error(f"Anthropic client error: {e}.", exc_info=True)
        else: logger.warning("Anthropic package not installed.")
        if self.is_available: logger.info("AnthropicSearchProvider initialized (async).")

    def get_provider_info(self) -> Dict[str, Any]:
        info = super().get_provider_info(); info['available'] = self.is_available; return info

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        if not self.is_available or not self.async_client:
            return {'provider': self.name, 'success': False, 'text': "Anthropic provider unavailable.", 'documents': [], 'error': "Client not init.", 'raw_response': None, 'execution_time': 0}
        model = kwargs.get("model", self.config.get("model", "claude-3-haiku-20240307"))
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.3))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 1500))
        messages = [{"role": "user", "content": query}]
        sys_prompt = kwargs.get("system_prompt", self.config.get("system_prompt", "You are a helpful search assistant."))
        logger.info(f"Anthropic search (Model: {model}): '{query[:100]}...'")
        try:
            response = await self.async_client.messages.create(model=model, system=sys_prompt, messages=messages, temperature=temperature, max_tokens=max_tokens)
            exec_time = time.time() - start_time
            if response and response.content and isinstance(response.content, list) and len(response.content) > 0:
                result_text = ""; block = response.content[0];
                if hasattr(block, 'text'): result_text = block.text
                logger.info(f"Anthropic success ({exec_time:.2f}s).")
                return {'provider': self.name, 'success': True, 'text': result_text.strip(), 'documents': [], 'error': None, 'raw_response': response, 'execution_time': exec_time}
            else: raise SearchProviderError("API returned empty/invalid response content.")
        except Exception as e:
            exec_time = time.time() - start_time; logger.error(f"Anthropic search failed ({exec_time:.2f}s): {e}", exc_info=True)
            err_type = type(e).__name__; err_msg = str(e)
            if ANTHROPIC_AVAILABLE and isinstance(e, anthropic.APIError): err_msg = f"Anthropic API Error: {e}"
            return {'provider': self.name, 'success': False, 'text': f"Failed: {err_msg}", 'documents': [], 'error': f"{err_type}: {err_msg}", 'raw_response': None, 'execution_time': exec_time}

class WebApiSearchProvider(SearchProvider):
    """Generic provider for calling an external web API."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("web_api", config)
        self.api_key: Optional[str] = None; self.api_url: Optional[str] = None
        self.api_key_header: str = self.config.get("api_key_header", "Authorization")
        self.timeout: int = self.config.get("timeout", 10); self.is_available: bool = False
        web_api_cfg = self.api_manager.get_web_api_config()
        if web_api_cfg and web_api_cfg.get("api_key"):
            self.api_key = web_api_cfg["api_key"]
            self.api_url = self.config.get("api_url") or web_api_cfg.get("api_url")
            if self.api_url: self.is_available = True; logger.info(f"WebApi initialized (URL: {self.api_url}).")
            else: logger.warning("WebApi disabled: URL missing.")
        else: logger.warning("WebApi disabled: Key missing.")

    def get_provider_info(self) -> Dict[str, Any]:
        info = super().get_provider_info(); info['available'] = self.is_available
        if self.api_url: info['config']['api_url'] = self.api_url
        return info

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        if not self.is_available or not self.api_url or not self.api_key:
            return {'provider': self.name, 'success': False, 'text': "Web API unavailable.", 'documents': [], 'error': "Client not configured", 'raw_response': None, 'execution_time': 0}
        headers = {"Content-Type": "application/json", self.api_key_header: f"Bearer {self.api_key}"}
        base_payload = self.config.get("payload", {}); request_payload = {**base_payload, "query": query, **kwargs}
        logger.info(f"Web API search (URL: {self.api_url}): '{query[:100]}...'")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, headers=headers, json=request_payload)
                response.raise_for_status()
            exec_time = time.time() - start_time; logger.info(f"Web API success ({exec_time:.2f}s). Status: {response.status_code}")
            try: data = response.json()
            except json.JSONDecodeError: raise SearchProviderError("API returned non-JSON.")
            text = data.get("answer", data.get("text", "No text provided."))
            docs_raw = data.get("sources", data.get("documents", [])); docs = []
            if isinstance(docs_raw, list):
                for doc in docs_raw:
                    if isinstance(doc, dict): docs.append({"title": doc.get("title"), "url": doc.get("url"), "snippet": doc.get("snippet", doc.get("text"))})
            return {'provider': self.name, 'success': True, 'text': text, 'documents': docs, 'error': None, 'raw_response': data, 'execution_time': exec_time}
        except httpx.RequestError as e:
            exec_time = time.time() - start_time; logger.error(f"Web API request failed ({exec_time:.2f}s): {e}", exc_info=True); err_type = type(e).__name__
            return {'provider': self.name, 'success': False, 'text': f"Request failed: {e}", 'documents': [], 'error': f"{err_type}: {e}", 'raw_response': None, 'execution_time': exec_time}
        except httpx.HTTPStatusError as e:
            exec_time = time.time() - start_time; resp_text = e.response.text[:200] if hasattr(e.response, 'text') else ''; logger.error(f"Web API error status ({exec_time:.2f}s): {e.response.status_code} - {resp_text}", exc_info=False); err_type = type(e).__name__
            return {'provider': self.name, 'success': False, 'text': f"API error: {e.response.status_code}", 'documents': [], 'error': f"{err_type}: Status {e.response.status_code}", 'raw_response': resp_text, 'execution_time': exec_time}
        except Exception as e:
            exec_time = time.time() - start_time; logger.error(f"Web API processing failed ({exec_time:.2f}s): {e}", exc_info=True); err_type = type(e).__name__
            return {'provider': self.name, 'success': False, 'text': f"Processing failed: {e}", 'documents': [], 'error': f"{err_type}: {e}", 'raw_response': None, 'execution_time': exec_time}

# --- Search Engine Orchestrator ---
class SearchEngine:
    """Orchestrates the search process using various components."""
    PROVIDER_CLASSES: Dict[str, Type[SearchProvider]] = {
        "cohere": CohereSearchProvider, "openai": OpenAISearchProvider,
        "anthropic": AnthropicSearchProvider, "web_api": WebApiSearchProvider,
    }

    def __init__(self):
        logger.info("Initializing SearchEngine...")
        self.config_manager: ConfigManager = get_config_manager()
        self.config = self.config_manager.get_active_config()
        self.api_manager: APIManager = get_api_manager()
        self.query_processor: QueryProcessor = get_query_processor()
        self.context_manager: QueryContextManager = get_context_manager()
        self.cache: SearchCache = get_search_cache()
        self.providers: Dict[str, SearchProvider] = self._initialize_providers()
        self.neuro_symbolic = None
        self.persistent_kg = self.query_processor.get_knowledge_graph() # Get ref from QueryProcessor

        if NEURO_SYMBOLIC_AVAILABLE:
            try: self.neuro_symbolic = NeuroSymbolicProcessor(); logger.info("Neuro-symbolic processor initialized")
            except Exception as e: logger.error(f"Failed to init neuro-symbolic processor: {e}", exc_info=True)
        else: logger.info("Neuro-symbolic processor disabled (dependencies missing).")

        logger.info(f"SearchEngine initialized. Profile: {self.config_manager.active_profile}. Providers: {list(self.providers.keys())}")

    def _initialize_providers(self) -> Dict[str, SearchProvider]:
        """Initialize provider instances based on config and availability."""
        providers = {}
        provider_configs = self.config.get("search_engine", {}).get("providers", {})
        for name, provider_class in self.PROVIDER_CLASSES.items():
            enabled = provider_configs.get(name, {}).get("enabled", True)
            if not enabled: logger.info(f"Provider '{name}' disabled in config."); continue
            config = provider_configs.get(name, {})
            try:
                instance = provider_class(config=config)
                if instance.get_provider_info()['available']: providers[name] = instance
            except Exception as e: logger.error(f"Failed to instantiate provider '{name}': {e}", exc_info=True)
        return providers

    def _get_config(self) -> Dict[str, Any]: return self.config # Simple retrieval for now

    def get_available_providers(self) -> List[str]: return list(self.providers.keys())

    async def execute_search(self, query: str) -> Dict[str, Any]:
        """Execute the full search workflow."""
        start_time = time.time()
        current_config = self._get_config()
        search_config = current_config.get("search_engine", {})
        use_cache = search_config.get("use_cache", True)
        use_fallback = search_config.get("use_fallback", True)
        default_provider = search_config.get("default_provider", "cohere")
        use_neuro = search_config.get("use_neuro_symbolic", False) and self.neuro_symbolic is not None
        use_kg_update = search_config.get("use_persistent_kg_update", True) and self.persistent_kg is not None

        processed_query_data = None; final_search_result = None
        providers_tried = []; cache_key = query; cache_hit = False
        neuro_symbolic_data = None; knowledge_graph_data = None

        try:
            # 1. Process Query
            logger.info(f"Processing query: '{query[:100]}...'")
            try:
                processed_query_data = process_and_contextualize_query(query, use_global_context=True)
                processed_query = processed_query_data.get("processed_query") or query
                logger.info(f"Using query for search/cache: '{processed_query[:100]}...'")
                cache_key = processed_query # Use processed query for cache
            except Exception as e:
                logger.error(f"Query processing failed: {e}. Using original.", exc_info=True)
                processed_query = query; cache_key = query; processed_query_data = {'error': str(e)}

            # 2. Check Cache
            if use_cache:
                cached_entry = self.cache.get(cache_key)
                if cached_entry and isinstance(cached_entry, CacheEntry) and hasattr(cached_entry, 'value'):
                    try:
                        cached_package = cached_entry.value
                        if isinstance(cached_package, dict):
                            logger.info(f"Cache hit for query key: '{cache_key[:100]}...'")
                            final_search_result = cached_package.get("search_result")
                            providers_tried = cached_package.get("providers_tried", [])
                            neuro_symbolic_data = cached_package.get("neuro_symbolic_data")
                            knowledge_graph_data = cached_package.get("knowledge_graph_data")
                            cache_hit = True
                        else: logger.warning(f"Invalid cache value type for '{cache_key}': {type(cached_package)}. Discarding."); self.cache.remove(cache_key)
                    except Exception as e: logger.error(f"Error processing cache entry '{cache_key}': {e}. Discarding.", exc_info=True); self.cache.remove(cache_key)
                elif cached_entry: logger.warning(f"Invalid cache data type for '{cache_key}': {type(cached_entry)}. Discarding."); self.cache.remove(cache_key)
                else: logger.info(f"Cache miss for '{cache_key[:100]}...'")
            else: logger.info("Caching disabled.")

            # 3. Execute Live Search (if not cached)
            live_provider_result = None # Store result from the provider loop
            if not cache_hit:
                neuro_symbolic_data = None; knowledge_graph_data = None # Reset derived data
                provider_to_try = default_provider
                attempts = 0; max_attempts = 1 + (search_config.get("max_fallbacks", 2) if use_fallback else 0)
                providers_tried = [] # Reset for live search

                while attempts < max_attempts:
                    attempts += 1
                    provider_instance = self.providers.get(provider_to_try)
                    if not provider_instance or not provider_instance.get_provider_info()['available']:
                        logger.warning(f"Provider '{provider_to_try}' unavailable. Trying fallback...")
                        next_provider = self._get_fallback_provider(providers_tried + [provider_to_try])
                        if not next_provider: break
                        provider_to_try = next_provider; continue

                    logger.info(f"Attempt {attempts}/{max_attempts}: Trying provider '{provider_to_try}' for query: '{processed_query[:100]}...'")
                    providers_tried.append(provider_to_try)
                    try:
                        live_provider_result = await provider_instance.search(processed_query)
                        if live_provider_result and live_provider_result.get("success"):
                            logger.info(f"Provider '{provider_to_try}' succeeded.")
                            final_search_result = live_provider_result # Assign successful result
                            break # Exit loop on success
                        else: logger.warning(f"Provider '{provider_to_try}' failed: {live_provider_result.get('error', 'Unknown') if live_provider_result else 'No result'}")
                    except Exception as provider_err:
                        logger.error(f"Provider '{provider_to_try}' raised exception: {provider_err}", exc_info=True)
                        live_provider_result = {'provider': provider_to_try, 'success': False, 'text': f"Exec error: {provider_err}", 'documents': [], 'error': f"{type(provider_err).__name__}: {provider_err}", 'raw_response': None, 'execution_time': 0}

                    next_provider = self._get_fallback_provider(providers_tried)
                    if not next_provider: logger.warning("No more fallback providers."); break
                    provider_to_try = next_provider

            # 4. Post-processing (if search was successful overall)
            is_successful = cache_hit or (final_search_result and final_search_result.get("success"))

            if is_successful and final_search_result:
                # Neuro-Symbolic
                if use_neuro and self.neuro_symbolic:
                    logger.info("Applying neuro-symbolic processing...")
                    try:
                        docs = final_search_result.get("documents", [])
                        if docs:
                             scores = self.neuro_symbolic.logical_reasoning(processed_query, docs)
                             neuro_symbolic_data = {"document_scores": scores}
                             if len(docs) == len(scores):
                                 for i, score in enumerate(scores):
                                     final_search_result['documents'][i]['neuro_symbolic_score'] = float(score) if score is not None else None
                    except Exception as ns_err: logger.error(f"Neuro-symbolic failed: {ns_err}", exc_info=True); neuro_symbolic_data = {"error": str(ns_err)}

                # KG Update
                if use_kg_update and self.persistent_kg:
                    logger.info("Updating persistent knowledge graph...")
                    try:
                        # Construct the package expected by query_processor
                        result_package_for_kg = {
                             "query": query,
                             "processed_query_data": processed_query_data, # Pass processed data
                             "search_result": final_search_result # Pass the successful result
                        }
                        updated = self.query_processor.update_from_search_result(query, result_package_for_kg, user_feedback=None)
                        knowledge_graph_data = {"persistent_kg_updated": updated}
                    except Exception as kg_err: logger.error(f"KG update failed: {kg_err}", exc_info=True); knowledge_graph_data = {"persistent_kg_error": str(kg_err)}

            # 5. Cache Result (if successful, live, enabled)
            if is_successful and use_cache and not cache_hit and final_search_result:
                 try:
                     search_result_copy = copy.deepcopy(final_search_result) # Deep copy
                     search_result_copy.pop('raw_response', None) # Remove raw response
                     cache_package = {
                         "search_result": search_result_copy, "providers_tried": providers_tried,
                         "neuro_symbolic_data": neuro_symbolic_data, "knowledge_graph_data": knowledge_graph_data,
                         "timestamp": datetime.now().isoformat()
                     }
                     ttl = search_config.get("cache_ttl_seconds", 86400)
                     self.cache.set(cache_key, cache_package, ttl=ttl)
                     logger.info(f"Result cached for '{cache_key[:100]}...' (TTL {ttl}s)")
                 except TypeError as serial_err: logger.error(f"Cache serialization failed for '{cache_key}': {serial_err}", exc_info=False)
                 except Exception as cache_err: logger.error(f"Cache save failed for '{cache_key}': {cache_err}", exc_info=True)

            # --- Prepare Final Result Package ---
            total_time = time.time() - start_time
            logger.info(f"Search workflow completed ({total_time:.2f}s). CacheHit: {cache_hit}. Success: {is_successful}.")

            # Determine the final result to return (prioritize success)
            effective_result = final_search_result if is_successful else (live_provider_result or { # Use live result if fail, or empty fail dict
                 'provider': None, 'success': False, 'text': "Search failed: No provider succeeded.",
                 'documents': [], 'error': "No provider succeeded.", 'raw_response': None, 'execution_time': 0
            })

            return {
                "original_query": query, "processed_query_data": processed_query_data,
                "search_result": effective_result, # Contains success status, text, docs, error
                "providers_tried": providers_tried, "total_execution_time": total_time,
                "cache_hit": cache_hit, "error": None if is_successful else effective_result.get("error"), # Top-level error if failed
                "timestamp": datetime.now().isoformat(), "neuro_symbolic_data": neuro_symbolic_data,
                "knowledge_graph_data": knowledge_graph_data
            }

        except Exception as e: # Catch unexpected workflow errors
            total_time = time.time() - start_time
            logger.critical(f"Critical search workflow error ({total_time:.2f}s): {e}", exc_info=True)
            return {
                "original_query": query, "processed_query_data": processed_query_data,
                "search_result": {'provider': None, 'success': False, 'text': f"Workflow failed: {e}", 'documents': [], 'error': f"Workflow error: {e}", 'raw_response': None, 'execution_time': 0},
                "providers_tried": providers_tried, "total_execution_time": total_time,
                "cache_hit": False, "error": f"Critical workflow error: {str(e)}",
                "timestamp": datetime.now().isoformat(), "neuro_symbolic_data": None, "knowledge_graph_data": None
            }

    def _get_fallback_provider(self, providers_already_tried: List[str]) -> Optional[str]:
        """Select the next available provider not already tried."""
        available = list(self.providers.keys()); random.shuffle(available)
        for provider in available:
            instance = self.providers.get(provider)
            if provider not in providers_already_tried and instance and instance.get_provider_info()['available']:
                logger.debug(f"Selected fallback provider: {provider}")
                return provider
        logger.debug("No suitable fallback provider found.")
        return None

    async def generate_exploration_suggestions(self, topic: str) -> Dict[str, Any]:
        """Generate exploratory suggestions for a topic."""
        start_time = time.time()
        search_config = self._get_config().get("search_engine", {})
        exp_config = search_config.get("exploration", {})
        provider_name = exp_config.get("provider", search_config.get("default_provider", "cohere"))
        provider = self.providers.get(provider_name)

        if not provider or not provider.get_provider_info()['available']:
            return {'success': False, 'topic': topic, 'error': f"Provider '{provider_name}' unavailable.", 'provider': provider_name, 'suggestions': None, 'execution_time': 0}

        prompt_template = exp_config.get("prompt_template", """
You are an AI assistant generating exploratory ideas for "{topic}" in JSON format:
{{
"sub_topics": ["Sub-topic 1", ...],
"key_questions": ["Question 1?", ...],
"influential_works": ["Researcher/Paper", ...],
"opposing_viewpoints": ["Debate Point 1", ...]
}}
Respond ONLY with the valid JSON object.
""")
        prompt = prompt_template.format(topic=topic)
        params = exp_config.get("provider_params", {}).get(provider_name, {})
        model = params.get("model", provider.config.get("model"))
        temp = params.get("temperature", 0.5); max_tokens = params.get("max_tokens", 1000)

        logger.info(f"Generating exploration for '{topic}' using '{provider_name}' (Model: {model}).")
        try:
            llm_response = await provider.search(prompt, model=model, temperature=temp, max_tokens=max_tokens, system_prompt="Generate structured JSON output.")
            exec_time = time.time() - start_time
            if llm_response and llm_response.get("success"):
                text = llm_response.get("text", "").strip()
                try:
                    json_match = re.search(r'({.*?})', text, re.DOTALL) # Simple JSON block finder
                    json_string = json_match.group(1) if json_match else text[text.find('{'):text.rfind('}')+1]
                    suggestions = json.loads(json_string)
                    expected = ['sub_topics', 'key_questions', 'influential_works', 'opposing_viewpoints']
                    # Ensure keys exist, default to empty list
                    suggestions = {k: suggestions.get(k, []) for k in expected}
                    logger.info(f"Exploration parsed successfully ({exec_time:.2f}s).")
                    return {'success': True, 'topic': topic, 'suggestions': suggestions, 'provider': provider_name, 'error': None, 'execution_time': exec_time, 'raw_llm_response': text}
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse exploration JSON: {json_err}. Raw: {text[:500]}...", exc_info=False)
                    return {'success': False, 'topic': topic, 'error': f"JSON parse failed: {json_err}", 'provider': provider_name, 'suggestions': None, 'execution_time': exec_time, 'raw_llm_response': text}
            else:
                err = llm_response.get('error', 'Provider failed.')
                logger.error(f"Exploration provider '{provider_name}' failed: {err}")
                return {'success': False, 'topic': topic, 'error': f"LLM error: {err}", 'provider': provider_name, 'suggestions': None, 'execution_time': exec_time}
        except Exception as e:
            exec_time = time.time() - start_time; logger.error(f"Exploration generation error: {e}", exc_info=True)
            return {'success': False, 'topic': topic, 'error': f"Exec error: {str(e)}", 'provider': provider_name, 'suggestions': None, 'execution_time': exec_time}

# --- Singleton Accessor ---
_search_engine_instance: Optional[SearchEngine] = None
_search_engine_lock = threading.Lock()
def get_search_engine() -> SearchEngine:
    global _search_engine_instance
    if _search_engine_instance is None:
        with _search_engine_lock:
            if _search_engine_instance is None:
                logger.info("Creating SearchEngine singleton instance...")
                _search_engine_instance = SearchEngine()
    return _search_engine_instance

# --- Async Runner Functions ---
async def run_search(query: str) -> Dict[str, Any]:
    return await get_search_engine().execute_search(query)

async def run_exploration(topic: str) -> Dict[str, Any]:
    return await get_search_engine().generate_exploration_suggestions(topic)

