# query_processor.py
"""
Query Processor Module for AI-powered Search

Handles:
- Query cleaning, normalization, tokenization, stemming, lemmatization.
- Entity extraction and intent determination.
- Context generation based on history and knowledge graph.
- Updating the knowledge graph from search results and feedback.
- Provides an interface for knowledge graph implementations (NetworkX, Simple).
- Singleton access for QueryProcessor and QueryContextManager.
"""

import re
import string
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from collections import defaultdict, Counter
import math
import random
import hashlib
from functools import lru_cache
import copy # +++ Added copy +++

# Optional dependencies (gracefully handle if not available)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not found. Some KG features might be limited.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not found. Falling back to SimpleKnowledgeGraph implementation.")

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        logging.warning(f"Failed to download NLTK data: {e}")
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Query processing features will be limited.")

# Configure logging
logger = logging.getLogger("query_processor")

# Default configurations
DEFAULT_PROCESSOR_CONFIG = {
    'stemming_enabled': True,
    'lemmatization_enabled': True,
    'stopwords_removal_enabled': True,
    'entity_extraction_enabled': True,
    'min_query_length': 3,
    'max_query_length': 100,
    'max_query_history': 20,
    'min_token_length': 2,
    'min_context_similarity': 0.3,
    'knowledge_graph': {
        'enabled': True,
        'storage_path': '.knowledge',
        'min_entity_confidence': 0.7,
        'min_relation_confidence': 0.8,
        'learning_rate': 0.1,
        'max_entities': 10000,
        'max_relations': 50000,
        'auto_save_interval': 300, # seconds
        'feedback_weight': 0.2,
        'min_feedback_count': 5,
        'similarity_threshold': 0.85,
        'pruning_factor': 0.01,
        'exploration_hop_distance': 1,
        'exploration_min_edge_weight': 0,
        'exploration_relationship_types': ['related_to', 'is_a', 'part_of', 'subtopic_of']
    }
}

minimal_stopwords = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like',
    'of', 'from', 'this', 'that', 'these', 'those', 'it', 'its',
    'what', 'who', 'when', 'where', 'why', 'how'
}

# --- Query Context Manager ---
class QueryContextManager:
    """Manages context for queries across sessions."""
    def __init__(self):
        self.global_context = {}
        self.session_context = {}
        self.context_lock = threading.RLock()
        logger.info("QueryContextManager initialized.")

    def get_global_context(self) -> Dict[str, Any]:
        with self.context_lock:
            return self.global_context.copy()

    def update_global_context(self, context: Dict[str, Any]) -> None:
        with self.context_lock:
            self.global_context.update(context)
            logger.debug(f"Global context updated with {len(context)} items")

    def clear_global_context(self) -> None:
        with self.context_lock:
            self.global_context.clear()
            logger.info("Global context cleared")

    # ... (Session context methods remain the same) ...
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        with self.context_lock:
            return self.session_context.get(session_id, {}).copy()

    def update_session_context(self, session_id: str, context: Dict[str, Any]) -> None:
        with self.context_lock:
            if session_id not in self.session_context:
                self.session_context[session_id] = {}
            self.session_context[session_id].update(context)
            logger.debug(f"Session context updated for {session_id}")

    def clear_session_context(self, session_id: str) -> None:
        with self.context_lock:
            if session_id in self.session_context:
                del self.session_context[session_id]
                logger.debug(f"Session context cleared for {session_id}")

# --- Query Processor ---
class QueryProcessor:
    """Class for processing search queries with semantic understanding"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = copy.deepcopy(DEFAULT_PROCESSOR_CONFIG) # Use deepcopy
        if config:
            self._update_config_recursive(self.config, config)

        # Initialize NLTK components
        self.stemmer = None
        self.lemmatizer = None
        self.stop_words = minimal_stopwords.copy()
        if NLTK_AVAILABLE:
            if self.config['stemming_enabled']: self.stemmer = PorterStemmer()
            if self.config['lemmatization_enabled']: self.lemmatizer = WordNetLemmatizer()
            if self.config['stopwords_removal_enabled']:
                try: self.stop_words = set(stopwords.words('english'))
                except LookupError: logger.warning("NLTK stopwords not found. Using minimal list.")
        else:
            logger.warning("NLTK not available. Limited query processing.")

        # Initialize query history
        self.query_history = []
        self.query_history_lock = threading.RLock()

        # Initialize knowledge graph
        self.knowledge_graph: Optional[KnowledgeGraph] = None # Type hint
        if self.config['knowledge_graph']['enabled']:
            if NETWORKX_AVAILABLE:
                try: self.knowledge_graph = NetworkXKnowledgeGraph(self.config['knowledge_graph'])
                except Exception as e:
                    logger.error(f"Failed to initialize NetworkXKnowledgeGraph: {e}. Disabling KG.", exc_info=True)
                    self.config['knowledge_graph']['enabled'] = False
            else: # Fallback to simple graph
                try:
                    self.knowledge_graph = SimpleKnowledgeGraph(self.config['knowledge_graph'])
                    logger.info("Using SimpleKnowledgeGraph implementation (NetworkX not found).")
                except Exception as e:
                    logger.error(f"Failed to initialize SimpleKnowledgeGraph: {e}. Disabling KG.", exc_info=True)
                    self.config['knowledge_graph']['enabled'] = False

        logger.info("QueryProcessor initialized.")

    def _update_config_recursive(self, base_config: Dict[str, Any], update_config: Dict[str, Any]) -> None:
        """Recursively update configuration dictionary."""
        for key, value in update_config.items():
            if key in base_config:
                if isinstance(value, dict) and isinstance(base_config[key], dict):
                    self._update_config_recursive(base_config[key], value)
                else:
                    base_config[key] = value

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a search query for semantic understanding."""
        start_time = time.time()

        if not query or not isinstance(query, str):
            logger.warning("Empty or invalid query received.")
            # Return a structure consistent with successful processing
            return {'original_query': query, 'processed_query': '', 'tokens': [], 'entities': [],
                    'stemmed_tokens': [], 'lemmatized_tokens': [], 'intent': 'unknown',
                    'context': {}, 'timestamp': datetime.now().isoformat(), 'processing_time': 0.0,
                    'error': 'Empty or invalid query'}

        try:
            cleaned_query = self._clean_query(query)
            tokens = self._tokenize(cleaned_query)

            filtered_tokens = tokens
            if self.config['stopwords_removal_enabled']:
                filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words and len(token) >= self.config['min_token_length']]

            stemmed_tokens = []
            if self.config['stemming_enabled'] and self.stemmer:
                stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]

            lemmatized_tokens = []
            if self.config['lemmatization_enabled'] and self.lemmatizer:
                lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

            # Pass tuple(tokens) to make it hashable for lru_cache
            entities = []
            if self.config['entity_extraction_enabled']:
                entities = self._extract_entities(query, tuple(tokens)) # Keep tuple() here for _extract_entities cache

            intent = self._determine_intent(query, tokens, entities)
            context = self._get_context(query, filtered_tokens, entities)

            # Add to query history
            with self.query_history_lock:
                self.query_history.append({
                    'query': query, 'tokens': tokens, 'filtered_tokens': filtered_tokens,
                    'entities': entities, 'intent': intent, 'timestamp': datetime.now().isoformat()
                })
                max_history = self.config['max_query_history']
                if len(self.query_history) > max_history:
                    self.query_history = self.query_history[-max_history:]

            processing_time = time.time() - start_time
            processed_data = {
                'original_query': query, 'processed_query': cleaned_query, 'tokens': tokens,
                'filtered_tokens': filtered_tokens, 'entities': entities, 'stemmed_tokens': stemmed_tokens,
                'lemmatized_tokens': lemmatized_tokens, 'intent': intent, 'context': context,
                'timestamp': datetime.now().isoformat(), 'processing_time': processing_time, 'error': None
            }

            logger.debug(f"Query '{query[:50]}...' processed in {processing_time:.4f}s")
            return processed_data

        except Exception as e:
            logger.error(f"Error during query processing for '{query[:50]}...': {e}", exc_info=True)
            processing_time = time.time() - start_time
            return {'original_query': query, 'processed_query': query, 'tokens': [], 'entities': [],
                    'stemmed_tokens': [], 'lemmatized_tokens': [], 'intent': 'unknown',
                    'context': {}, 'timestamp': datetime.now().isoformat(), 'processing_time': processing_time,
                    'error': f"Processing failed: {str(e)}"}

    def update_from_search_result(
        self,
        query: str,
        result_package: Dict[str, Any], # Expect the full package from search_engine
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update knowledge graph from search results."""
        if not self.knowledge_graph or not self.config['knowledge_graph']['enabled']:
            logger.debug("KG disabled or unavailable, skipping update.")
            return False

        search_result = result_package.get('search_result')
        if not search_result or not isinstance(search_result, dict):
            logger.warning("Invalid or missing 'search_result' in result package, skipping KG update.")
            return False

        result_text = search_result.get('text', '')
        if not result_text:
            logger.debug("No result text for KG update.")
            return False

        # Extract query data (can reuse from result_package if available)
        query_data = result_package.get('processed_query_data')
        if not query_data:
            query_data = self.process_query(query)

        try:
            # Extract entities and relationships
            # Pass tuple to _extract_entities for its cache
            entities = self._extract_entities(result_text, tuple(query_data.get('tokens', []))) # Keep tuple() here

            # *** FIX: Pass LISTS directly to _extract_relationships ***
            relationships = self._extract_relationships(
                query,
                result_text,
                query_data.get('entities', []), # Pass list directly
                entities                         # Pass list directly
            )

            # Calculate confidence scores
            source_docs = search_result.get('documents', [])
            confidence_scores_map = self._calculate_confidence(entities, relationships, source_docs, user_feedback)

            updates_made = 0
            min_entity_confidence = self.config['knowledge_graph']['min_entity_confidence']
            entity_scores_by_id = confidence_scores_map.get('entities', {}) # ID -> score map

            # Add entities
            for entity_dict in entities: # Iterate original list
                if not isinstance(entity_dict, dict) or 'id' not in entity_dict: continue
                entity_id = entity_dict['id']
                score = entity_scores_by_id.get(entity_id, entity_dict.get('confidence', 0.5)) # Use ID for lookup
                if score > min_entity_confidence:
                    try:
                        if self.knowledge_graph.add_entity(
                            entity_id=entity_id, name=entity_dict['name'],
                            entity_type=entity_dict.get('type', 'unknown'),
                            properties=entity_dict.get('properties', {}), confidence=score):
                            updates_made += 1
                    except Exception as e:
                        logger.error(f"Error adding entity '{entity_dict.get('name')}' ({entity_id}) to KG: {e}", exc_info=False)

            # Add relationships
            min_relation_confidence = self.config['knowledge_graph']['min_relation_confidence']
            relationship_scores_by_id = confidence_scores_map.get('relationships', {}) # RelID -> score map

            for relationship_dict in relationships: # Iterate original list
                if not isinstance(relationship_dict, dict) or not all(k in relationship_dict for k in ['source_id', 'target_id', 'type']): continue
                # Recreate unique ID used in _calculate_confidence
                rel_id = f"{relationship_dict['source_id']}_{relationship_dict['type']}_{relationship_dict['target_id']}"
                score = relationship_scores_by_id.get(rel_id, relationship_dict.get('confidence', 0.5)) # Use RelID for lookup
                if score > min_relation_confidence:
                    try:
                        if self.knowledge_graph.add_relationship(
                            source_id=relationship_dict['source_id'], target_id=relationship_dict['target_id'],
                            relation_type=relationship_dict['type'],
                            properties=relationship_dict.get('properties', {}), confidence=score):
                            updates_made += 1
                    except Exception as e:
                        logger.error(f"Error adding relationship {rel_id} to KG: {e}", exc_info=False)

            logger.debug(f"Attempted KG update from result: {updates_made} updates.")
            if updates_made > 0 and hasattr(self.knowledge_graph, 'save_if_needed'):
                self.knowledge_graph.save_if_needed()

            return updates_made > 0

        except Exception as e:
            # Log the actual error that occurs during the update process
            logger.error(f"Error during KG update process: {e}", exc_info=True) # Keep True here
            return False


    def _clean_query(self, query: str) -> str:
        """Clean query: strip, normalize whitespace, enforce length limits."""
        cleaned = query.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)

        min_length = self.config['min_query_length']
        max_length = self.config['max_query_length']

        if len(cleaned) < min_length: logger.debug(f"Query '{cleaned}' too short.")
        if len(cleaned) > max_length:
            logger.debug(f"Query too long, truncating to {max_length} chars.")
            cleaned = cleaned[:max_length]
        return cleaned

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if not text: return []

        if NLTK_AVAILABLE:
            try: return word_tokenize(text)
            except Exception as e: logger.warning(f"NLTK tokenization failed: {e}. Falling back.")

        # Fallback
        text_for_split = re.sub(r'[^\w\s-]', ' ', text)
        tokens = text_for_split.split()
        min_len = self.config['min_token_length']
        return [token for token in tokens if len(token) >= min_len]

    # Use tuple for tokens to make it hashable for cache
    @lru_cache(maxsize=128)
    def _extract_entities(self, text: str, tokens: Optional[Tuple[str]] = None) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if not text: return []
        entities_map: Dict[str, Dict[str, Any]] = {} # ID -> entity dict

        # 1. Pattern Matching (Simplified Proper Noun Example)
        if tokens is None: tokens_list = self._tokenize(text)
        else: tokens_list = list(tokens) # Convert tuple back to list if needed

        # --- Add more robust NER here (Spacy/NLTK if available, more regex) ---
        # Example using Spacy if available (add self.nlp init if using this)
        # try:
        #     if hasattr(self, 'nlp') and self.nlp:
        #         doc = self.nlp(text[:10000]) # Limit length
        #         for ent in doc.ents:
        #             if len(ent.text) >= self.config['min_token_length']:
        #                 name = ent.text.strip()
        #                 ent_id = self._generate_entity_id(name)
        #                 current_confidence = 0.8
        #                 if ent_id not in entities_map or entities_map[ent_id]['confidence'] < current_confidence:
        #                     entities_map[ent_id] = {
        #                         'id': ent_id, 'name': name, 'type': ent.label_.lower(),
        #                         'confidence': current_confidence, 'properties': {'source': 'spacy_ner'}
        #                     }
        # except Exception as spacy_err:
        #     logger.warning(f"Spacy NER failed: {spacy_err}")
        # --- End Spacy Example ---

        # Basic Capitalized Word Fallback (less reliable)
        for i, token in enumerate(tokens_list):
            is_capitalized = token and token[0].isupper() and not token.isupper()
            is_sentence_start = (i == 0) or (i > 0 and tokens_list[i-1].endswith(('.', '!', '?')))
            if is_capitalized and not is_sentence_start and len(token) >= self.config['min_token_length']:
                name = token # Simplistic: assumes single token proper noun
                ent_id = self._generate_entity_id(name)
                if ent_id not in entities_map or entities_map[ent_id]['confidence'] < 0.6:
                    entities_map[ent_id] = {
                        'id': ent_id, 'name': name, 'type': 'proper_noun_guess',
                        'confidence': 0.6, 'properties': {'source': 'pattern_capitalized'}
                    }

        # 2. Knowledge Graph Lookup
        if self.knowledge_graph and self.config['knowledge_graph']['enabled']:
            try:
                known_entities = self.knowledge_graph.find_entities_in_text(text)
                for entity in known_entities:
                    if isinstance(entity, dict) and 'id' in entity:
                        ent_id = entity['id']
                        kg_confidence = entity.get('confidence', 0.95)
                        if ent_id not in entities_map or entities_map[ent_id]['confidence'] < kg_confidence:
                            entity_copy = entity.copy() # Avoid modifying original KG data if cached
                            entity_copy['confidence'] = kg_confidence
                            entity_copy.setdefault('properties', {})['source'] = 'knowledge_graph'
                            entities_map[ent_id] = entity_copy
            except Exception as e:
                logger.warning(f"Error finding entities via KG: {e}", exc_info=False)

        return list(entities_map.values())


    # *** FIX: Removed @lru_cache decorator ***
    # @lru_cache(maxsize=64)
    def _extract_relationships(
        self,
        query: str,
        result_text: str,
        # *** FIX: Changed Tuple to List for type hints ***
        query_entities: Optional[List[Dict[str, Any]]] = None,
        result_entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        if not result_text: return []

        # *** FIX: Use arguments directly (already lists) ***
        query_entities_list = query_entities if query_entities is not None else self._extract_entities(query)
        result_entities_list = result_entities if result_entities is not None else self._extract_entities(result_text)

        relationships = []
        processed_pairs = set() # (source_id, target_id)

        # Map entity names (lower) to ID for quick lookup in result entities
        result_entity_ids_by_name = {
            e['name'].lower(): e['id']
            for e in result_entities_list if isinstance(e, dict) and 'id' in e and 'name' in e
        }

        relation_verbs = {
            'is': ('is_a', 0.9), 'was': ('is_a', 0.85), 'are': ('is_a', 0.9), 'were': ('is_a', 0.85),
            'has': ('has_property', 0.8), 'have': ('has_property', 0.8),
            'contains': ('contains', 0.8), 'includes': ('contains', 0.8),
            'causes': ('causes', 0.7), 'leads to': ('causes', 0.7),
            'related to': ('related_to', 0.5), 'part of': ('part_of', 0.8),
            'located in': ('located_in', 0.9), 'founded by': ('founded_by', 0.9),
            'created by': ('created_by', 0.9),
            # ... add more verbs/patterns ...
        }
        verb_pattern = r'\b(' + '|'.join(re.escape(v) for v in relation_verbs.keys()) + r')\b'

        # Sentence Splitting (using a simpler approach)
        sentences = [s.strip() for s in re.split(r'[.!?]+', result_text) if s.strip()]

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Find entities mentioned in this sentence
            sentence_entities_present = []
            for entity in result_entities_list:
                 if isinstance(entity, dict) and 'name' in entity and entity['name'].lower() in sentence_lower:
                     sentence_entities_present.append(entity)

            if len(sentence_entities_present) < 2: continue

            # Check pairs within the sentence
            for i in range(len(sentence_entities_present)):
                for j in range(len(sentence_entities_present)):
                    if i == j: continue

                    ent1 = sentence_entities_present[i]
                    ent2 = sentence_entities_present[j]

                    # Ensure they are valid dicts with id and name
                    if not (isinstance(ent1, dict) and 'id' in ent1 and 'name' in ent1 and
                            isinstance(ent2, dict) and 'id' in ent2 and 'name' in ent2):
                        continue

                    ent1_id, ent1_name = ent1['id'], ent1['name']
                    ent2_id, ent2_name = ent2['id'], ent2['name']

                    pair_key = tuple(sorted((ent1_id, ent2_id)))
                    if pair_key in processed_pairs: continue

                    try:
                        # Simple verb check between mentions (can be improved significantly)
                        pos1 = sentence_lower.find(ent1_name.lower())
                        pos2 = sentence_lower.find(ent2_name.lower())
                        if pos1 == -1 or pos2 == -1: continue

                        start = min(pos1, pos2) + len(ent1_name if pos1 < pos2 else ent2_name)
                        end = max(pos1, pos2)
                        text_between = sentence_lower[start:end].strip()

                        verb_matches = re.findall(verb_pattern, text_between)
                        if verb_matches:
                            verb = verb_matches[0]
                            rel_type, base_confidence = relation_verbs[verb]
                            source_id, target_id = (ent1_id, ent2_id) if pos1 < pos2 else (ent2_id, ent1_id)
                            source_name, target_name = (ent1_name, ent2_name) if pos1 < pos2 else (ent2_name, ent1_name)

                            relationships.append({
                                'source_id': source_id, 'target_id': target_id,
                                'source_name': source_name, 'target_name': target_name, # Include names
                                'type': rel_type, 'confidence': base_confidence * 0.8,
                                'properties': {'source': 'verb_detection', 'sentence': sentence[:100]}
                            })
                            processed_pairs.add(pair_key)
                        # Weak co-occurrence relation if no verb found
                        elif not processed_pairs.intersection({pair_key}): # Check set intersection
                             relationships.append({
                                'source_id': ent1_id, 'target_id': ent2_id,
                                'source_name': ent1_name, 'target_name': ent2_name,
                                'type': 'related_to', 'confidence': 0.4,
                                'properties': {'source': 'co-occurrence', 'sentence': sentence[:100]}
                            })
                             processed_pairs.add(pair_key)

                    except Exception as e:
                        logger.debug(f"Error processing relationship between '{ent1_name}' and '{ent2_name}': {e}")

        return relationships


    # Note: Entities and Relationships are lists of dicts here
    def _calculate_confidence(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        source_documents: List[Union[Dict[str, Any], str]],
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, float]]: # Return type: {'entities': {ID: score}, 'relationships': {ID: score}}
        """Calculate confidence scores mapped by ID."""
        entity_scores: Dict[str, float] = {} # entity_id -> score
        relationship_scores: Dict[str, float] = {} # rel_id -> score

        # Initialize with base confidence
        for entity in entities:
            if isinstance(entity, dict) and 'id' in entity:
                entity_scores[entity['id']] = entity.get('confidence', 0.5)
            else: logger.warning(f"Skipping invalid entity in confidence init: {entity}")

        for relationship in relationships:
             if isinstance(relationship, dict) and all(k in relationship for k in ['source_id', 'target_id', 'type']):
                 rel_id = f"{relationship['source_id']}_{relationship['type']}_{relationship['target_id']}"
                 relationship_scores[rel_id] = relationship.get('confidence', 0.5)
             else: logger.warning(f"Skipping invalid relationship in confidence init: {relationship}")

        # Factor 1: Source diversity
        num_sources = len(source_documents)
        if num_sources > 0:
            entity_mention_sources = defaultdict(set)
            relationship_mention_sources = defaultdict(set) # Uses rel_id

            # Create maps for efficient lookup (ID -> name, RelID -> names)
            entity_id_to_name = {e['id']: e['name'] for e in entities if isinstance(e, dict) and 'id' in e and 'name' in e}
            rel_id_to_names = {
                f"{r['source_id']}_{r['type']}_{r['target_id']}": (r['source_name'], r['target_name'])
                for r in relationships if isinstance(r, dict) and all(k in r for k in ['source_id', 'target_id', 'type', 'source_name', 'target_name'])
            }

            for idx, doc in enumerate(source_documents):
                doc_text = ''
                if isinstance(doc, dict): doc_text = doc.get('text', '') or doc.get('snippet', '') or doc.get('title', '')
                elif isinstance(doc, str): doc_text = doc
                if not doc_text: continue
                doc_text_lower = doc_text.lower()

                # Check entities using ID map
                for entity_id, entity_name in entity_id_to_name.items():
                    if entity_name.lower() in doc_text_lower:
                        entity_mention_sources[entity_id].add(idx)

                # Check relationships using RelID map
                for rel_id, (source_name, target_name) in rel_id_to_names.items():
                    if source_name.lower() in doc_text_lower and target_name.lower() in doc_text_lower:
                        relationship_mention_sources[rel_id].add(idx)

            # Adjust confidence based on source count
            for entity_id, sources in entity_mention_sources.items():
                if entity_id in entity_scores: # Ensure key exists
                    mention_freq = len(sources) / num_sources
                    boost = min(0.3, mention_freq * 0.3)
                    entity_scores[entity_id] = min(0.98, entity_scores[entity_id] + boost)

            for rel_id, sources in relationship_mention_sources.items():
                if rel_id in relationship_scores: # Ensure key exists
                    mention_freq = len(sources) / num_sources
                    boost = min(0.3, mention_freq * 0.3)
                    relationship_scores[rel_id] = min(0.98, relationship_scores[rel_id] + boost)

        # Factor 2: User feedback
        if user_feedback:
            feedback_weight = self.config['knowledge_graph']['feedback_weight']
            rating = user_feedback.get('rating', 0)
            adjustment = 0
            if isinstance(rating, (int, float)):
                if rating > 3: adjustment = feedback_weight
                elif rating < 3: adjustment = -feedback_weight

            if adjustment != 0:
                for entity_id in list(entity_scores.keys()): # Use list to iterate over copy of keys
                    entity_scores[entity_id] = max(0.01, min(0.99, entity_scores[entity_id] + adjustment))
                for rel_id in list(relationship_scores.keys()):
                    relationship_scores[rel_id] = max(0.01, min(0.99, relationship_scores[rel_id] + adjustment))

        # Return the ID-to-score mappings
        return {'entities': entity_scores, 'relationships': relationship_scores}

    def _determine_intent(self, query: str, tokens: List[str], entities: List[Dict[str, Any]]) -> str:
        """Determine the intent of a query."""
        query_lower = query.lower().strip()
        first_token = tokens[0].lower() if tokens else ""

        # Question intents
        question_starters = {'what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'could', 'should', 'would', 'will', 'do', 'does', 'did'}
        if first_token in question_starters or query_lower.endswith('?'):
            if re.search(r'^(what is|what are|define|meaning of)', query_lower): return 'definition'
            if re.search(r'^(how to|steps to|guide for|tutorial on)', query_lower): return 'how_to'
            if re.search(r'^(why|reason for|explain why)', query_lower): return 'explanation'
            # ... (other specific question types) ...
            return 'question'

        # Command intents
        command_starters = {'find', 'search', 'show', 'list', 'get', 'display', 'give me', 'buy', 'purchase', 'order', 'shop'}
        if first_token in command_starters:
            if first_token in {'buy', 'purchase', 'order', 'shop'}: return 'purchase'
            return 'command_search'

        # Topic intent (heuristic)
        entity_chars = sum(len(e['name']) for e in entities if isinstance(e, dict) and 'name' in e)
        query_len = len(query)
        if query_len > 0 and entities and (entity_chars / query_len) > 0.5:
            return 'topic'

        if len(tokens) <= 5: # Short queries often topics
             return 'topic'

        return 'search' # Default

    def _get_context(self, query: str, filtered_tokens: List[str], entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get relevant context for a query."""
        context = {'related_queries': [], 'entity_info': {}, 'session_context': {}}
        query_set = set(token.lower() for token in filtered_tokens)
        if not query_set: return context # Skip if no tokens

        # 1. Related queries from history
        min_similarity = self.config['min_context_similarity']
        with self.query_history_lock:
            for hist_item in reversed(self.query_history[-10:]): # Check recent history
                hist_tokens = hist_item.get('filtered_tokens', [])
                hist_set = set(token.lower() for token in hist_tokens)
                if not hist_set: continue

                intersection = len(query_set.intersection(hist_set))
                union = len(query_set.union(hist_set))
                similarity = intersection / union if union > 0 else 0

                if similarity >= min_similarity:
                    context['related_queries'].append({
                        'query': hist_item['query'], 'similarity': similarity,
                        'timestamp': hist_item['timestamp'], 'entities': hist_item.get('entities', [])
                    })

        # 2. Entity info from KG
        if self.knowledge_graph and self.config['knowledge_graph']['enabled'] and entities:
            for entity in entities:
                if isinstance(entity, dict) and 'id' in entity:
                    entity_id = entity['id']
                    try:
                        entity_info = self.knowledge_graph.get_entity_info(entity_id)
                        if entity_info:
                             relationships = self.knowledge_graph.get_relationships_for_entity(entity_id)
                             entity_info['relationships'] = relationships
                             context['entity_info'][entity_id] = entity_info
                    except Exception as e: logger.warning(f"Error getting KG info for {entity_id}: {e}")

        # 3. Session context
        context['session_context'] = {'timestamp': datetime.now().isoformat()}
        return context

    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate a consistent ID for an entity."""
        normalized = re.sub(r'\s+', ' ', entity_name.lower().strip())
        name_hash = hashlib.sha1(normalized.encode()).hexdigest()[:12]
        slug = re.sub(r'[^\w]+', '_', normalized)[:40].strip('_')
        slug = re.sub(r'_+', '_', slug)
        return f"{slug}_{name_hash}" if slug else name_hash # Handle empty slug case


    def get_knowledge_graph(self) -> Optional['KnowledgeGraph']:
        """Get the knowledge graph instance."""
        return self.knowledge_graph if self.config['knowledge_graph']['enabled'] else None

    def clear_history(self) -> None:
        """Clear the query history."""
        with self.query_history_lock:
            self.query_history = []
            logger.info("Query history cleared")

# --- Knowledge Graph Base Class ---
class KnowledgeGraph:
    """Abstract base class for knowledge graph implementations"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_save_time = time.time()

    def add_entity(self, entity_id: str, name: str, entity_type: str = 'unknown', properties: Dict[str, Any] = None, confidence: float = 1.0) -> bool:
        raise NotImplementedError

    def add_relationship(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None, confidence: float = 1.0) -> bool:
        raise NotImplementedError

    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def get_relationships(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_relationships_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def find_entities_in_text(self, text: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_exploration_concepts(self, entity_id: str, top_k: int = 5) -> List[str]:
        raise NotImplementedError

    def save(self) -> bool:
        raise NotImplementedError

    def load(self) -> bool:
        raise NotImplementedError

    def save_if_needed(self) -> bool:
        current_time = time.time()
        interval = self.config.get('auto_save_interval', 300)
        if current_time - self.last_save_time > interval:
            saved = self.save()
            if saved: self.last_save_time = current_time
            return saved
        return False

# --- NetworkX Knowledge Graph Implementation ---
class NetworkXKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph implementation using NetworkX"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not NETWORKX_AVAILABLE: raise ImportError("NetworkX required")
        self.graph = nx.DiGraph()
        self.entity_count = 0
        self.relationship_count = 0
        self.storage_path = Path(config.get('storage_path', '.knowledge'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.storage_path / 'knowledge_graph.graphml'
        self.load()
        logger.info(f"Initialized NetworkX KG ({self.graph.number_of_nodes()}N, {self.graph.number_of_edges()}E)")

    def add_entity(self, entity_id: str, name: str, entity_type: str = 'unknown', properties: Dict[str, Any] = None, confidence: float = 1.0) -> bool:
        if not entity_id or not name: return False
        props = properties or {}
        max_entities = self.config.get('max_entities', 10000)

        if self.graph.number_of_nodes() >= max_entities: self._prune_entities()
        if self.graph.number_of_nodes() >= max_entities:
             logger.warning(f"Cannot add entity '{name}', max limit reached.")
             return False

        if entity_id not in self.graph:
            self.graph.add_node(entity_id, name=name, type=entity_type, confidence=confidence, access_count=1, created_at=time.time(), **props)
            self.entity_count = self.graph.number_of_nodes()
            return True
        else: # Update existing
            node_data = self.graph.nodes[entity_id]
            lr = self.config.get('learning_rate', 0.1)
            node_data['confidence'] = min(0.99, node_data.get('confidence', 0.5) * (1 - lr) + confidence * lr)
            node_data['access_count'] = node_data.get('access_count', 0) + 1
            node_data['updated_at'] = time.time()
            if entity_type != 'unknown': node_data['type'] = entity_type
            if props: node_data.update(props)
            return True

    def add_relationship(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None, confidence: float = 1.0) -> bool:
        if not source_id or not target_id or not relation_type: return False
        if source_id not in self.graph or target_id not in self.graph: return False
        props = properties or {}
        max_relations = self.config.get('max_relations', 50000)

        if self.graph.number_of_edges() >= max_relations: self._prune_relationships()
        if self.graph.number_of_edges() >= max_relations:
             logger.warning(f"Cannot add relationship '{relation_type}', max limit reached.")
             return False

        if not self.graph.has_edge(source_id, target_id):
            self.graph.add_edge(source_id, target_id, type=relation_type, confidence=confidence, access_count=1, created_at=time.time(), **props)
            self.relationship_count = self.graph.number_of_edges()
            return True
        else: # Update existing
            edge_data = self.graph.edges[source_id, target_id]
            lr = self.config.get('learning_rate', 0.1)
            edge_data['confidence'] = min(0.99, edge_data.get('confidence', 0.5) * (1 - lr) + confidence * lr)
            edge_data['access_count'] = edge_data.get('access_count', 0) + 1
            edge_data['updated_at'] = time.time()
            if relation_type != 'unknown': edge_data['type'] = relation_type
            if props: edge_data.update(props)
            return True

    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.graph.nodes.get(entity_id) # Returns a view, copy if modification needed

    def get_relationships(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        rels = []
        if source_id in self.graph and target_id in self.graph:
            if self.graph.has_edge(source_id, target_id):
                edge_data = self.graph.get_edge_data(source_id, target_id).copy()
                edge_data.update({'direction': 'outgoing', 'source_id': source_id, 'target_id': target_id,
                                  'source_name': self.graph.nodes[source_id].get('name', source_id),
                                  'target_name': self.graph.nodes[target_id].get('name', target_id)})
                rels.append(edge_data)
            if self.graph.has_edge(target_id, source_id):
                edge_data = self.graph.get_edge_data(target_id, source_id).copy()
                edge_data.update({'direction': 'incoming', 'source_id': target_id, 'target_id': source_id,
                                  'source_name': self.graph.nodes[target_id].get('name', target_id),
                                  'target_name': self.graph.nodes[source_id].get('name', source_id)})
                rels.append(edge_data)
        return rels

    def get_relationships_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        if entity_id not in self.graph: return []
        rels = []
        source_name = self.graph.nodes[entity_id].get('name', entity_id)

        for neighbor in self.graph.successors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor).copy()
            edge_data.update({'direction': 'outgoing', 'source_id': entity_id, 'target_id': neighbor,
                              'source_name': source_name,
                              'target_name': self.graph.nodes[neighbor].get('name', neighbor)})
            rels.append(edge_data)
        for neighbor in self.graph.predecessors(entity_id):
            edge_data = self.graph.get_edge_data(neighbor, entity_id).copy()
            edge_data.update({'direction': 'incoming', 'source_id': neighbor, 'target_id': entity_id,
                              'source_name': self.graph.nodes[neighbor].get('name', neighbor),
                              'target_name': source_name})
            rels.append(edge_data)
        return rels

    def find_entities_in_text(self, text: str) -> List[Dict[str, Any]]:
        # Simple substring matching (can be slow)
        if not text: return []
        found = []
        text_lower = text.lower()
        for node_id, data in self.graph.nodes(data=True):
            name = data.get('name', '')
            if name and name.lower() in text_lower:
                entity_info = data.copy()
                entity_info['id'] = node_id
                found.append(entity_info)
        return found

    def get_exploration_concepts(self, entity_id: str, top_k: int = 5) -> List[str]:
        """Get related concepts using graph traversal (NetworkX)."""
        if entity_id not in self.graph: return []

        hop = self.config.get('exploration_hop_distance', 1)
        min_weight = self.config.get('exploration_min_edge_weight', 0)
        priority_types = set(self.config.get('exploration_relationship_types', []))

        neighbor_nodes = set(nx.ego_graph(self.graph, entity_id, radius=hop)) # undirected=False by default for ego_graph
        neighbor_nodes.discard(entity_id)
        if not neighbor_nodes: return []

        scores = {}
        for node_id in neighbor_nodes:
            score = 0.0
            node_data = self.graph.nodes[node_id]
            score += node_data.get('confidence', 0.5) * math.log1p(node_data.get('access_count', 1))

            # Check direct connection properties
            path_score = 0
            for u, v, data in self.graph.edges(data=True, nbunch=[entity_id, node_id]):
                # Check paths between entity_id and node_id in both directions
                is_relevant_path = (u == entity_id and v == node_id) or (u == node_id and v == entity_id)
                if is_relevant_path and data.get('weight', 0) >= min_weight: # Check optional weight if present
                    path_score += data.get('confidence', 0.5) * (1.5 if data.get('type') in priority_types else 1.0)
            score += path_score
            scores[node_id] = score

        sorted_nodes = sorted(scores, key=scores.get, reverse=True)
        top_names = [self.graph.nodes[nid].get('name', nid) for nid in sorted_nodes[:top_k]]
        logger.debug(f"Exploration concepts for {entity_id} (NX): {top_names}")
        return top_names

    def save(self) -> bool:
        try:
            temp_file = self.graph_file.with_suffix('.graphml.tmp')
            nx.write_graphml(self.graph, str(temp_file)) # Pass path as string
            temp_file.replace(self.graph_file)
            logger.info(f"Saved NetworkX graph to {self.graph_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save NetworkX graph: {e}", exc_info=True)
            return False

    def load(self) -> bool:
        if not self.graph_file.exists():
            logger.info(f"NX graph file {self.graph_file} not found. Starting fresh.")
            return False
        try:
            self.graph = nx.read_graphml(str(self.graph_file)) # Pass path as string
            self.entity_count = self.graph.number_of_nodes()
            self.relationship_count = self.graph.number_of_edges()
            # Data type conversion for attributes (GraphML might load them as strings)
            for node_id, data in self.graph.nodes(data=True):
                if 'confidence' in data: data['confidence'] = float(data['confidence'])
                if 'access_count' in data: data['access_count'] = int(data['access_count'])
                if 'created_at' in data: data['created_at'] = float(data['created_at'])
                if 'updated_at' in data: data['updated_at'] = float(data.get('updated_at', time.time())) # Add if missing
            for u, v, data in self.graph.edges(data=True):
                if 'confidence' in data: data['confidence'] = float(data['confidence'])
                if 'access_count' in data: data['access_count'] = int(data['access_count'])
                if 'created_at' in data: data['created_at'] = float(data['created_at'])
                if 'updated_at' in data: data['updated_at'] = float(data.get('updated_at', time.time()))
            logger.info(f"Loaded NetworkX graph from {self.graph_file} ({self.entity_count}N, {self.relationship_count}E)")
            return True
        except Exception as e:
            logger.error(f"Failed to load NetworkX graph: {e}. Starting fresh.", exc_info=True)
            self.graph = nx.DiGraph() # Reset graph on load error
            self.entity_count = 0
            self.relationship_count = 0
            return False

    def _prune_entities(self) -> int:
        """Prune low-confidence/access entities."""
        max_entities = self.config.get('max_entities', 10000)
        if self.graph.number_of_nodes() <= max_entities: return 0

        factor = self.config.get('pruning_factor', 0.01)
        num_prune = max(1, int(self.graph.number_of_nodes() * factor))

        # Score nodes: lower is worse (confidence * log(access_count+1))
        node_scores = {
            node_id: data.get('confidence', 0.5) * math.log1p(data.get('access_count', 0))
            for node_id, data in self.graph.nodes(data=True)
        }
        candidates = sorted(node_scores, key=node_scores.get) # Sort by score ascending

        nodes_to_remove = candidates[:num_prune]
        self.graph.remove_nodes_from(nodes_to_remove)
        pruned_count = len(nodes_to_remove)

        self.entity_count = self.graph.number_of_nodes()
        self.relationship_count = self.graph.number_of_edges() # Edges connected to removed nodes are auto-removed

        if pruned_count > 0: logger.info(f"Pruned {pruned_count} NX entities.")
        return pruned_count

    def _prune_relationships(self) -> int:
        """Prune low-confidence/access relationships."""
        max_relations = self.config.get('max_relations', 50000)
        if self.graph.number_of_edges() <= max_relations: return 0

        factor = self.config.get('pruning_factor', 0.01)
        num_prune = max(1, int(self.graph.number_of_edges() * factor))

        # Score edges: lower is worse
        edge_scores = {
            (u, v): data.get('confidence', 0.5) * math.log1p(data.get('access_count', 0))
            for u, v, data in self.graph.edges(data=True)
        }
        candidates = sorted(edge_scores, key=edge_scores.get) # Sort by score ascending

        edges_to_remove = candidates[:num_prune]
        self.graph.remove_edges_from(edges_to_remove)
        pruned_count = len(edges_to_remove)

        self.relationship_count = self.graph.number_of_edges()

        if pruned_count > 0: logger.info(f"Pruned {pruned_count} NX relationships.")
        return pruned_count


# --- Simple Dict-based Knowledge Graph Implementation ---
class SimpleKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph implementation using simple dictionaries"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.entities: Dict[str, Dict[str, Any]] = {} # entity_id -> {data}
        self.relationships: Dict[str, Dict[str, Any]] = {} # rel_id -> {data}
        self.entity_count = 0
        self.relationship_count = 0
        self.entity_name_index: Dict[str, str] = {} # name.lower() -> entity_id
        self.outgoing_rels: Dict[str, List[str]] = defaultdict(list) # source_id -> [rel_id]
        self.incoming_rels: Dict[str, List[str]] = defaultdict(list) # target_id -> [rel_id]
        self.storage_path = Path(config.get('storage_path', '.knowledge'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.storage_path / 'knowledge_graph_simple.json'
        self.load()
        logger.info(f"Initialized SimpleKG ({self.entity_count} entities, {self.relationship_count} relationships)")

    def add_entity(self, entity_id: str, name: str, entity_type: str = 'unknown', properties: Dict[str, Any] = None, confidence: float = 1.0) -> bool:
        if not entity_id or not name: return False
        props = properties or {}
        max_entities = self.config.get('max_entities', 10000)

        if self.entity_count >= max_entities: self._prune_entities()
        if self.entity_count >= max_entities:
            logger.warning(f"Cannot add entity '{name}', max limit reached (SimpleKG).")
            return False

        name_lower = name.lower()
        if entity_id not in self.entities:
            self.entities[entity_id] = {
                'id': entity_id, 'name': name, 'type': entity_type,
                'confidence': confidence, 'access_count': 1,
                'created_at': time.time(), **props
            }
            if name_lower: self.entity_name_index[name_lower] = entity_id
            self.entity_count += 1
            return True
        else: # Update existing
            node_data = self.entities[entity_id]
            lr = self.config.get('learning_rate', 0.1)
            node_data['confidence'] = min(0.99, node_data.get('confidence', 0.5) * (1 - lr) + confidence * lr)
            node_data['access_count'] = node_data.get('access_count', 0) + 1
            node_data['updated_at'] = time.time()
            if entity_type != 'unknown': node_data['type'] = entity_type
            if props: node_data.update(props)
            # Update name index if name changed
            if name_lower and node_data.get('name', '').lower() != name_lower:
                old_name_lower = node_data.get('name', '').lower()
                if old_name_lower in self.entity_name_index and self.entity_name_index[old_name_lower] == entity_id:
                    del self.entity_name_index[old_name_lower]
                node_data['name'] = name
                self.entity_name_index[name_lower] = entity_id
            return True

    def add_relationship(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None, confidence: float = 1.0) -> bool:
        if not source_id or not target_id or not relation_type: return False
        if source_id not in self.entities or target_id not in self.entities: return False
        props = properties or {}
        max_relations = self.config.get('max_relations', 50000)

        if self.relationship_count >= max_relations: self._prune_relationships()
        if self.relationship_count >= max_relations:
            logger.warning(f"Cannot add relationship '{relation_type}', max limit reached (SimpleKG).")
            return False

        rel_id = f"{source_id}_{relation_type}_{target_id}" # Create a unique ID
        if rel_id not in self.relationships:
            self.relationships[rel_id] = {
                'id': rel_id, 'source_id': source_id, 'target_id': target_id,
                'type': relation_type, 'confidence': confidence,
                'access_count': 1, 'created_at': time.time(), **props
            }
            self.outgoing_rels[source_id].append(rel_id)
            self.incoming_rels[target_id].append(rel_id)
            self.relationship_count += 1
            return True
        else: # Update existing
            edge_data = self.relationships[rel_id]
            lr = self.config.get('learning_rate', 0.1)
            edge_data['confidence'] = min(0.99, edge_data.get('confidence', 0.5) * (1 - lr) + confidence * lr)
            edge_data['access_count'] = edge_data.get('access_count', 0) + 1
            edge_data['updated_at'] = time.time()
            if relation_type != 'unknown': edge_data['type'] = relation_type
            if props: edge_data.update(props)
            return True

    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(entity_id)

    def get_relationships(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        rels = []
        source_name = self.entities.get(source_id, {}).get('name', source_id)
        target_name = self.entities.get(target_id, {}).get('name', target_id)

        # Check outgoing from source
        out_rel_ids = self.outgoing_rels.get(source_id, [])
        for rel_id in out_rel_ids:
            rel_data = self.relationships.get(rel_id)
            if rel_data and rel_data.get('target_id') == target_id:
                rel_copy = rel_data.copy()
                rel_copy.update({'direction': 'outgoing', 'source_name': source_name, 'target_name': target_name})
                rels.append(rel_copy)

        # Check incoming to source (outgoing from target)
        in_rel_ids = self.incoming_rels.get(source_id, [])
        for rel_id in in_rel_ids:
            rel_data = self.relationships.get(rel_id)
            if rel_data and rel_data.get('source_id') == target_id:
                 rel_copy = rel_data.copy()
                 rel_copy.update({'direction': 'incoming', 'source_name': target_name, 'target_name': source_name}) # Reversed names
                 rels.append(rel_copy)
        return rels

    def get_relationships_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        if entity_id not in self.entities: return []
        rels = []
        source_name = self.entities.get(entity_id, {}).get('name', entity_id)

        # Outgoing
        out_rel_ids = self.outgoing_rels.get(entity_id, [])
        for rel_id in out_rel_ids:
            rel_data = self.relationships.get(rel_id)
            if rel_data:
                target_id = rel_data.get('target_id')
                target_name = self.entities.get(target_id, {}).get('name', target_id)
                rel_copy = rel_data.copy()
                rel_copy.update({'direction': 'outgoing', 'source_name': source_name, 'target_name': target_name})
                rels.append(rel_copy)

        # Incoming
        in_rel_ids = self.incoming_rels.get(entity_id, [])
        for rel_id in in_rel_ids:
            rel_data = self.relationships.get(rel_id)
            if rel_data:
                source_id_in = rel_data.get('source_id')
                source_name_in = self.entities.get(source_id_in, {}).get('name', source_id_in)
                rel_copy = rel_data.copy()
                rel_copy.update({'direction': 'incoming', 'source_name': source_name_in, 'target_name': source_name})
                rels.append(rel_copy)
        return rels

    def find_entities_in_text(self, text: str) -> List[Dict[str, Any]]:
        if not text: return []
        found = []
        text_lower = text.lower()
        # Use name index for potentially faster lookup
        matched_ids = set()
        for name_lower, entity_id in self.entity_name_index.items():
            if name_lower in text_lower:
                 if entity_id in self.entities and entity_id not in matched_ids:
                     found.append(self.entities[entity_id].copy())
                     matched_ids.add(entity_id)
        return found

    def get_exploration_concepts(self, entity_id: str, top_k: int = 5) -> List[str]:
        """Get related concepts using relationship traversal (Simple)."""
        if entity_id not in self.entities: return []

        priority_types = set(self.config.get('exploration_relationship_types', []))
        scores = defaultdict(float)
        related_rels = self.get_relationships_for_entity(entity_id)

        for rel in related_rels:
            neighbor_id = None
            if rel.get('direction') == 'outgoing': neighbor_id = rel.get('target_id')
            elif rel.get('direction') == 'incoming': neighbor_id = rel.get('source_id')

            if neighbor_id and neighbor_id != entity_id:
                 score = rel.get('confidence', 0.5) * math.log1p(rel.get('access_count', 1))
                 if rel.get('type') in priority_types: score *= 1.5
                 scores[neighbor_id] += score

        scored_neighbors = []
        for neighbor_id, score in scores.items():
            neighbor_info = self.entities.get(neighbor_id)
            if neighbor_info:
                final_score = score + neighbor_info.get('confidence', 0.5) * math.log1p(neighbor_info.get('access_count', 1))
                scored_neighbors.append((neighbor_id, final_score))

        sorted_neighbors = sorted(scored_neighbors, key=lambda item: item[1], reverse=True)
        top_names = [self.entities.get(nid, {}).get('name', nid) for nid, _ in sorted_neighbors[:top_k]]
        logger.debug(f"Exploration concepts for {entity_id} (SimpleKG): {top_names}")
        return top_names

    def save(self) -> bool:
        try:
            data = {'entities': self.entities, 'relationships': self.relationships}
            temp_file = self.graph_file.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f: json.dump(data, f)
            temp_file.replace(self.graph_file)
            logger.info(f"Saved SimpleKG to {self.graph_file}")
            return True
        except Exception as e: logger.error(f"Failed to save SimpleKG: {e}", exc_info=True); return False

    def load(self) -> bool:
        if not self.graph_file.exists(): logger.info(f"SimpleKG file {self.graph_file} not found."); return False
        try:
            with open(self.graph_file, 'r') as f: data = json.load(f)
            self.entities = data.get('entities', {})
            self.relationships = data.get('relationships', {})

            # Rebuild indices
            self.entity_name_index = {v.get('name', '').lower(): k for k, v in self.entities.items() if v.get('name')}
            self.outgoing_rels, self.incoming_rels = defaultdict(list), defaultdict(list)
            for rel_id, rel_data in self.relationships.items():
                src, tgt = rel_data.get('source_id'), rel_data.get('target_id')
                if src: self.outgoing_rels[src].append(rel_id)
                if tgt: self.incoming_rels[tgt].append(rel_id)

            self.entity_count = len(self.entities)
            self.relationship_count = len(self.relationships)
            logger.info(f"Loaded SimpleKG from {self.graph_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load SimpleKG: {e}", exc_info=True)
            self.entities, self.relationships = {}, {}; self.entity_name_index = {}
            self.outgoing_rels, self.incoming_rels = defaultdict(list), defaultdict(list)
            self.entity_count, self.relationship_count = 0, 0
            return False

    def _prune_entities(self) -> int:
        """Prune low-confidence/access entities."""
        max_entities = self.config.get('max_entities', 10000)
        if self.entity_count <= max_entities: return 0

        factor = self.config.get('pruning_factor', 0.01)
        num_prune = max(1, int(self.entity_count * factor))

        candidates = sorted(self.entities.items(), key=lambda i: (i[1].get('confidence', 0.5), i[1].get('access_count', 0)))
        pruned_count = 0
        rel_ids_to_remove = set()

        for entity_id, data in candidates[:num_prune]:
            name_lower = data.get('name', '').lower()
            if name_lower in self.entity_name_index and self.entity_name_index[name_lower] == entity_id:
                del self.entity_name_index[name_lower]
            del self.entities[entity_id]
            pruned_count += 1
            rel_ids_to_remove.update(self.outgoing_rels.pop(entity_id, []))
            rel_ids_to_remove.update(self.incoming_rels.pop(entity_id, []))

        # Remove affected relationships
        removed_rel_count = 0
        for rel_id in list(rel_ids_to_remove): # Iterate list copy
            if rel_id in self.relationships:
                rel_data = self.relationships[rel_id]
                # Remove from other entity's indices
                src, tgt = rel_data.get('source_id'), rel_data.get('target_id')
                if src and src in self.outgoing_rels and rel_id in self.outgoing_rels[src]: self.outgoing_rels[src].remove(rel_id)
                if tgt and tgt in self.incoming_rels and rel_id in self.incoming_rels[tgt]: self.incoming_rels[tgt].remove(rel_id)
                del self.relationships[rel_id]
                removed_rel_count += 1

        self.entity_count -= pruned_count
        self.relationship_count -= removed_rel_count

        if pruned_count > 0: logger.info(f"Pruned {pruned_count} SimpleKG entities and {removed_rel_count} relationships.")
        return pruned_count

    def _prune_relationships(self) -> int:
        """Prune low-confidence/access relationships."""
        max_relations = self.config.get('max_relations', 50000)
        if self.relationship_count <= max_relations: return 0

        factor = self.config.get('pruning_factor', 0.01)
        num_prune = max(1, int(self.relationship_count * factor))

        candidates = sorted(self.relationships.items(), key=lambda i: (i[1].get('confidence', 0.5), i[1].get('access_count', 0)))
        pruned_count = 0

        for rel_id, data in candidates[:num_prune]:
            src, tgt = data.get('source_id'), data.get('target_id')
            if src in self.outgoing_rels and rel_id in self.outgoing_rels[src]: self.outgoing_rels[src].remove(rel_id)
            if tgt in self.incoming_rels and rel_id in self.incoming_rels[tgt]: self.incoming_rels[tgt].remove(rel_id)
            del self.relationships[rel_id]
            pruned_count += 1

        self.relationship_count -= pruned_count

        if pruned_count > 0: logger.info(f"Pruned {pruned_count} SimpleKG relationships.")
        return pruned_count


# --- Function to process and contextualize query ---
def process_and_contextualize_query(query: str, use_global_context: bool = True) -> Dict[str, Any]:
    """Process and contextualize a query using the singleton QueryProcessor."""
    query_processor = get_query_processor()
    processed_data = query_processor.process_query(query)

    if use_global_context:
        context_manager = get_context_manager()
        try:
            global_ctx = context_manager.get_global_context()
            processed_data.setdefault('context', {}).setdefault('global', {}).update(global_ctx)
        except Exception as e:
            logger.error(f"Error getting global context: {e}", exc_info=False)

    return processed_data

# --- Singleton Accessors ---
_query_processor_instance: Optional[QueryProcessor] = None
_query_processor_lock = threading.Lock()

def get_query_processor() -> QueryProcessor:
    global _query_processor_instance
    if _query_processor_instance is None:
        with _query_processor_lock:
            if _query_processor_instance is None:
                logger.info("Creating QueryProcessor singleton instance.")
                _query_processor_instance = QueryProcessor()
    return _query_processor_instance

_query_context_manager_instance: Optional[QueryContextManager] = None
_query_context_manager_lock = threading.Lock()

def get_context_manager() -> QueryContextManager:
    global _query_context_manager_instance
    if _query_context_manager_instance is None:
        with _query_context_manager_lock:
            if _query_context_manager_instance is None:
                logger.info("Creating QueryContextManager singleton instance.")
                _query_context_manager_instance = QueryContextManager()
    return _query_context_manager_instance
