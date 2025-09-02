# result_analyzer.py

"""
Result Analyzer Module for AI-powered Search

This module provides advanced result verification and analysis capabilities to:

1. Detect potential hallucinations in AI-generated responses
2. Verify facts against multiple sources
3. Calculate confidence scores for search results
4. Extract key information from responses
5. Compare results across different sources
6. Process counterfactual reasoning for "what-if" scenarios
7. Apply causal inference to generate hypothetical outcomes

The goal is to improve the reliability and trustworthiness of AI search results.
"""

import re
import json
import string
import hashlib
import logging
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import math
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, Union
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("result_analyzer")

# Default configuration (can be overridden)
DEFAULT_ANALYZER_CONFIG = {
    'min_confidence_threshold': 0.6,  # Threshold below which results might be flagged
    'source_diversity_weight': 0.3,  # Weight for source diversity in confidence score
    'factual_consistency_weight': 0.4,  # Weight for factual consistency in confidence score
    'source_credibility_weight': 0.3,  # Weight for source credibility in confidence score
    'min_sources_required': 2,  # Minimum sources for reliable analysis
    'enable_strict_verification': False,  # If True, requires stronger keyword overlap
    'max_token_similarity': 0.85,  # Threshold for detecting near-verbatim copying
    'statement_min_length': 5,  # Minimum words for a statement to be extracted
    'term_min_length': 3,  # Minimum length for extracted key terms
    'verification_overlap_threshold': 0.6,  # Min keyword overlap for verification
    'cache_size': 100,  # Size of the internal results cache
    'enable_counterfactual_reasoning': True,  # Enable counterfactual "what-if" reasoning
    'counterfactual_confidence_threshold': 0.7,  # Minimum confidence for counterfactual reasoning
    'causal_relation_threshold': 0.6,  # Minimum strength for causal relations
    'max_causal_depth': 3  # Maximum depth of causal chains to explore
}


class ResultAnalyzer:
    """
    Main class for analyzing and verifying search results
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the result analyzer
        Args:
            config (dict, optional): Configuration settings, merged with defaults.
        """
        # Merge provided config with defaults
        base_config = DEFAULT_ANALYZER_CONFIG.copy()
        if config:
            base_config.update(config)
        self.config = base_config

        # Cache for analyzed results (using LRU for automatic size management)
        self._results_cache = lru_cache(maxsize=self.config.get('cache_size', 100))(self._analyze_result_internal)
        self._term_cache = lru_cache(maxsize=500)(self._extract_key_terms_internal)
        self._statement_cache = lru_cache(maxsize=200)(self._extract_key_statements_internal)
        
        # Initialize counterfactual reasoner if enabled
        if self.config.get('enable_counterfactual_reasoning', True):
            self.counterfactual_reasoner = CounterfactualReasoner(self.config)
        else:
            self.counterfactual_reasoner = None

    def analyze_result(
        self,
        result_text: str,
        source_documents: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a search result for verification and confidence scoring.
        Uses caching based on input content.
        Args:
            result_text (str): The AI-generated result text.
            source_documents (list): List of source document dictionaries.
                Expected format: {'id': str, 'url': str, 'snippet': str, 'text': str, ...}
            query (str, optional): The original query.
        Returns:
            dict: Analysis results including confidence, verification details, etc.
        """
        # Validate inputs
        if not isinstance(result_text, str): result_text = ""
        if not isinstance(source_documents, list): source_documents = []

        # Ensure documents are dicts for consistent processing
        valid_source_documents = [doc for doc in source_documents if isinstance(doc, dict)]

        # Check if this is a counterfactual query and process accordingly
        if self.counterfactual_reasoner and query and self.counterfactual_reasoner.is_counterfactual_query(query):
            logger.info(f"Detected counterfactual query: '{query}'")
            # Process the counterfactual query
            try:
                counterfactual_analysis = self.counterfactual_reasoner.process_query(
                    query, result_text, valid_source_documents
                )
                # If we have a standard analysis as well, combine them
                standard_analysis = self._analyze_standard_result(result_text, valid_source_documents, query)
                
                # Merge standard and counterfactual analyses
                combined_analysis = standard_analysis.copy()
                combined_analysis.update({
                    'is_counterfactual': True,
                    'counterfactual_analysis': counterfactual_analysis,
                })
                
                return combined_analysis
            except Exception as e:
                logger.error(f"Error during counterfactual reasoning: {e}", exc_info=True)
                # Fall back to standard analysis
                logger.info("Falling back to standard analysis for counterfactual query")
        
        # Standard analysis
        return self._analyze_standard_result(result_text, valid_source_documents, query)
    
    def _analyze_standard_result(
        self,
        result_text: str,
        source_documents: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform standard (non-counterfactual) analysis of search results
        """
        # Use the cached internal method
        # Note: lru_cache cannot directly handle dict/list arguments, so we need a stable representation.
        # We create a hash key within the cached method itself.
        try:
            # Pass validated data to the internal cached method
            source_ids_tuple = tuple(sorted(doc.get('id', '') for doc in source_documents))
            analysis_result = self._results_cache(result_text, source_ids_tuple, query)
            
            # Return a copy to prevent modification of the cached object
            return analysis_result.copy()
        except Exception as e:
            logger.error(f"Error during result analysis: {e}", exc_info=True)
            # Return a default error structure
            return {
                'confidence_score': 0.0,
                'potential_hallucinations': ['Analysis failed due to internal error'],
                'verified_facts': [],
                'source_diversity': 0.0,
                'factual_consistency': 0.0,
                'source_credibility': 0.0,
                'key_statements': [],
                'analysis_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _analyze_result_internal(
        self,
        result_text: str,
        source_ids_tuple: Tuple[str, ...],  # Use tuple of IDs for cache key stability
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Internal method performing the analysis. Decorated with lru_cache.
        """
        # If not enough source documents (based on the ID tuple), mark as low confidence
        min_sources = self.config.get('min_sources_required', 2)
        if len(source_ids_tuple) < min_sources:
            logger.warning(f"Insufficient sources ({len(source_ids_tuple)} < {min_sources}) for reliable analysis.")
            # Extract statements even with low confidence
            key_statements = self._extract_key_statements(result_text)
            return {
                'confidence_score': 0.3,  # Low confidence score
                'potential_hallucinations': ['Insufficient sources to verify content'],
                'verified_facts': [],
                'source_diversity': 0.0,
                'factual_consistency': 0.0,
                'source_credibility': 0.0,
                'key_statements': key_statements,
                'analysis_timestamp': datetime.now().isoformat()
            }

        # Extract key statements from the result
        key_statements = self._extract_key_statements(result_text)
        
        # This is a simplification since we don't have the actual source_documents here
        # In a real implementation, we'd need to have access to the full source documents
        placeholder_source_documents = []  # This should be the actual documents in practice
        
        # Verify statements against sources
        verified_facts, suspicious_statements = self._verify_statements(
            key_statements, placeholder_source_documents
        )

        # Calculate metrics
        source_diversity = self._calculate_source_diversity(placeholder_source_documents)
        factual_consistency = self._calculate_factual_consistency(
            key_statements, placeholder_source_documents
        )
        source_credibility = self._calculate_source_credibility(placeholder_source_documents)

        # Calculate overall confidence score
        confidence_score = (
            self.config.get('source_diversity_weight', 0.3) * source_diversity +
            self.config.get('factual_consistency_weight', 0.4) * factual_consistency +
            self.config.get('source_credibility_weight', 0.3) * source_credibility
        )

        # Reduce confidence if hallucinations detected
        if suspicious_statements:
            reduction_factor = 1.0 - (0.1 * len(suspicious_statements) / max(1, len(key_statements)))
            confidence_score *= max(0.5, reduction_factor)  # Don't reduce below 0.5 just for this

        # Ensure confidence is within bounds [0, 1]
        confidence_score = max(0.0, min(1.0, confidence_score))

        analysis = {
            'confidence_score': confidence_score,
            'potential_hallucinations': suspicious_statements,
            'verified_facts': verified_facts,
            'source_diversity': source_diversity,
            'factual_consistency': factual_consistency,
            'source_credibility': source_credibility,
            'key_statements': key_statements,
            'analysis_timestamp': datetime.now().isoformat()
        }

        return analysis

    def _generate_cache_key(self, result_text: str, source_documents: List[Dict[str, Any]], query: Optional[str] = None) -> str:
        """
        Generate a stable cache key for the analysis based on content hash.
        Args:
            result_text (str): The result text.
            source_documents (list): The source documents.
            query (str, optional): The original query.
        Returns:
            str: A hash-based cache key.
        """
        # Create a string representation of the source documents using IDs and snippet hashes
        source_parts = []
        for doc in sorted(source_documents, key=lambda d: d.get('id', '')):  # Sort by ID
            doc_id = doc.get('id', '')
            # Use snippet hash for content representation, fallback to text/url
            content_repr = doc.get('snippet', doc.get('text', doc.get('url', '')))
            content_hash = hashlib.sha1(content_repr.encode()).hexdigest()[:8]  # Short hash
            source_parts.append(f"{doc_id}:{content_hash}")
        source_str = "|".join(source_parts)

        # Combine all inputs into a single string
        combined = f"query:{query or ''}|text:{result_text}|sources:{source_str}"

        # Create a hash of the combined string
        return hashlib.sha256(combined.encode()).hexdigest()

    def _extract_key_statements(self, text: str) -> List[str]:
        """
        Extract key factual statements (sentences) from the result text using caching.
        Args:
            text (str): The result text.
        Returns:
            list: List of key statements (sentences).
        """
        if not text or not isinstance(text, str):
            return []

        return self._statement_cache(text)

    def _extract_key_statements_internal(self, text: str) -> List[str]:
        """Internal method to extract statements, decorated by cache."""
        # Simple sentence splitting using regex (handles ., !, ?)
        # Looks for sentence-ending punctuation followed by space and uppercase letter or end of string.
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)

        # Filter out short sentences and None values from split
        min_length = self.config.get('statement_min_length', 5)
        key_statements = [
            s.strip() for s in sentences
            if s and len(s.strip().split()) >= min_length
        ]

        return key_statements

    def _verify_statements(
        self,
        statements: List[str],
        source_documents: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        Verify statements against source documents based on keyword overlap.
        Args:
            statements (list): List of statements to verify.
            source_documents (list): List of source document dictionaries.
        Returns:
            tuple: (verified_facts, suspicious_statements)
        """
        verified_facts = []
        suspicious_statements = []
        overlap_threshold = self.config.get('verification_overlap_threshold', 0.6)
        strict_mode = self.config.get('enable_strict_verification', False)
        max_token_sim = self.config.get('max_token_similarity', 0.85)

        if not source_documents:  # Cannot verify without sources
            return [], statements

        # Pre-process source documents content and terms
        doc_contents = []
        doc_term_sets = []

        for doc in source_documents:
            content = ""
            if isinstance(doc, dict):
                content += doc.get('title', '') + " "  # Include title
                content += doc.get('snippet', '') + " "
                # Optionally add full text if available and needed, careful with length
                # content += doc.get('text', '')
            elif isinstance(doc, str):  # Allow passing simple strings as docs
                content = doc
            content = content.strip()
            if content:
                doc_contents.append(content)
                doc_term_sets.append(set(self._extract_key_terms(content)))

        if not doc_contents:  # No usable source content
            return [], statements

        for statement in statements:
            statement_terms = set(self._extract_key_terms(statement))
            if not statement_terms:
                suspicious_statements.append(statement)  # Cannot verify empty statement
                continue

            statement_verified = False
            highest_overlap = 0.0
            best_match_doc_idx = -1

            # Check against each document
            for i, doc_terms in enumerate(doc_term_sets):
                if not doc_terms: continue

                # Calculate keyword overlap (Jaccard index)
                intersection = len(statement_terms.intersection(doc_terms))
                union = len(statement_terms.union(doc_terms))
                overlap = intersection / union if union > 0 else 0.0

                if overlap > highest_overlap:
                    highest_overlap = overlap
                    best_match_doc_idx = i

                # Check if overlap meets threshold
                if overlap >= overlap_threshold:
                    # Optional: Check for near-verbatim copying
                    token_similarity = self._calculate_token_similarity(statement, doc_contents[i])
                    if token_similarity >= max_token_sim:
                        logger.warning(f"High token similarity ({token_similarity:.2f}) found for statement: '{statement}' "
                                       f"in source {i}. May indicate copying.")
                    statement_verified = True

                    # In strict mode, we might require multiple sources or higher overlap
                    if not strict_mode:
                        break  # Found sufficient verification in one source

            if statement_verified:
                verified_facts.append(statement)
            else:
                logger.debug(f"Statement marked suspicious (overlap < {overlap_threshold}): '{statement}' (Max overlap: {highest_overlap:.2f})")
                suspicious_statements.append(statement)

        return verified_facts, suspicious_statements

    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text for comparison using caching.
        Args:
            text (str): Text to extract terms from.
        Returns:
            list: List of key terms (lowercase, punctuation removed, stopwords removed).
        """
        if not text or not isinstance(text, str):
            return []

        # Use the internal cached method
        return self._term_cache(text)

    def _extract_key_terms_internal(self, text: str) -> List[str]:
        """Internal method for term extraction, decorated by lru_cache."""
        if not text:
            return []

        # Convert to lowercase
        text_lower = text.lower()

        # Remove punctuation using str.translate for efficiency
        translator = str.maketrans('', '', string.punctuation)
        text_no_punct = text_lower.translate(translator)

        # Split into words
        words = text_no_punct.split()

        # Basic stopwords set
        stopwords_set = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like',
            'of', 'from', 'that', 'this', 'these', 'those', 'it', 'its', 'it\'s',
            'what', 'who', 'when', 'where', 'which', 'why', 'how', 'do', 'does', 'did',
            'me', 'you', 'he', 'she', 'they', 'we', 'my', 'your', 'his', 'her', 'their', 'our',
        }

        min_len = self.config.get('term_min_length', 3)
        terms = [word for word in words if word not in stopwords_set and len(word) >= min_len]

        return terms

    def _calculate_token_similarity(self, statement: str, document_text: str, n: int = 5) -> float:
        """
        Calculate token-level similarity using N-grams to detect near-verbatim copying.
        Args:
            statement (str): The statement to check.
            document_text (str): The document text.
            n (int): The size of N-grams to use (e.g., 5 for 5-word sequences).
        Returns:
            float: Similarity score between 0 (no overlap) and 1 (identical N-grams).
        """
        if not statement or not document_text:
            return 0.0

        # Simple normalization: lowercase and remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        norm_statement = statement.lower().translate(translator)
        norm_document = document_text.lower().translate(translator)

        # Get N-grams from the normalized statement
        def get_ngrams(text: str, n_val: int) -> Set[str]:
            tokens = text.split()
            if len(tokens) < n_val:
                return set()  # Not enough words for an N-gram
            return set(' '.join(tokens[i:i+n_val]) for i in range(len(tokens) - n_val + 1))

        statement_ngrams = get_ngrams(norm_statement, n)
        if not statement_ngrams:
            # If statement is shorter than N, check for direct substring presence
            return 1.0 if norm_statement in norm_document else 0.0

        # Check how many statement N-grams appear verbatim in the document
        doc_ngram_set = get_ngrams(norm_document, n)
        matches = len(statement_ngrams.intersection(doc_ngram_set))

        # Calculate similarity score
        similarity = matches / len(statement_ngrams)
        return similarity

    def _calculate_source_diversity(self, source_documents: List[Dict[str, Any]]) -> float:
        """
        Calculate a score for the diversity of sources based on domain names.
        Args:
            source_documents (list): List of source document dictionaries.
        Returns:
            float: Diversity score between 0 and 1.
        """
        if not source_documents or len(source_documents) < 2:
            return 0.0  # Cannot have diversity with 0 or 1 source

        # Extract domains from document URLs
        domains = []
        for doc in source_documents:
            if isinstance(doc, dict) and 'url' in doc:
                url = doc.get('url')
                if url and isinstance(url, str):
                    # Extract domain using regex (handles http/https and optional www.)
                    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    if domain_match:
                        domain = domain_match.group(1)
                        # Optional: Normalize domain (e.g., remove port if present)
                        domain = domain.split(':')[0]
                        domains.append(domain)

        if not domains:
            return 0.2  # Low score if no URLs/domains found

        num_sources = len(source_documents)  # Use original count before filtering URLs
        num_unique_domains = len(set(domains))

        # Score based on ratio of unique domains to total domains found
        if num_unique_domains == 1:
            diversity_score = 0.1  # Very low diversity if all from one domain
        elif num_unique_domains < self.config.get('min_sources_required', 2):
            diversity_score = 0.3  # Low diversity if fewer unique domains than required sources
        else:
            # Ratio, scaled slightly non-linearly (e.g., using log) to reward more unique sources
            ratio = num_unique_domains / len(domains) if domains else 0
            # Simple linear score for now
            diversity_score = ratio

        # Ensure score is within bounds
        return max(0.0, min(1.0, diversity_score))

    def _calculate_factual_consistency(
        self,
        statements: List[str],
        source_documents: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate a score for factual consistency across sources based on term coverage variance.
        Args:
            statements (list): List of extracted statements.
            source_documents (list): List of source document dictionaries.
        Returns:
            float: Consistency score between 0 and 1.
        """
        if not statements or not source_documents or len(source_documents) < 2:
            return 0.5  # Neutral score if not enough data for comparison

        # Get all unique key terms from all statements
        all_statement_terms = set()
        for statement in statements:
            all_statement_terms.update(self._extract_key_terms(statement))

        if not all_statement_terms:
            return 0.5  # Neutral if no terms extracted from statements

        # For each source, calculate term coverage
        source_coverages = []
        num_valid_docs = 0
        for doc in source_documents:
            # Get document content
            doc_content = ""
            if isinstance(doc, dict):
                doc_content += doc.get('title', '') + " "
                doc_content += doc.get('snippet', '') + " "
                # doc_content += doc.get('text', '') # Optional: include full text
            elif isinstance(doc, str):
                doc_content = doc
            doc_content = doc_content.strip()
            if not doc_content:
                continue  # Skip doc if no content

            num_valid_docs += 1
            doc_terms = set(self._extract_key_terms(doc_content))

            # Calculate what fraction of statement terms appear in this doc
            if not doc_terms:
                coverage = 0.0
            else:
                covered_terms = len(all_statement_terms.intersection(doc_terms))
                coverage = covered_terms / len(all_statement_terms)

            source_coverages.append(coverage)

        if num_valid_docs < 2:
            return 0.5  # Neutral if fewer than 2 documents had content

        # Calculate the average coverage across all valid sources
        avg_coverage = sum(source_coverages) / num_valid_docs

        # Calculate variance in coverage (lower is better, indicates consistency)
        variance = sum((c - avg_coverage) ** 2 for c in source_coverages) / num_valid_docs

        # Convert variance to a consistency score (0 to 1, higher is better)
        # Score decreases as variance increases. Use sqrt(variance) (std dev) for linear scale.
        # Max possible std dev is 0.5 (if half are 1, half are 0). Scale accordingly.
        std_dev = math.sqrt(variance)
        consistency_from_variance = max(0.0, 1.0 - (std_dev / 0.5))  # Normalize std dev

        # Combine average coverage (how much is covered) with consistency (how similarly it's covered)
        # Give slightly more weight to consistency
        consistency_score = 0.4 * avg_coverage + 0.6 * consistency_from_variance

        return max(0.0, min(1.0, consistency_score))

    def _calculate_source_credibility(self, source_documents: List[Dict[str, Any]]) -> float:
        """
        Calculate an aggregate score for the credibility of sources based on simple heuristics
        (TLDs, known domains). Needs enhancement with real reputation data.
        Args:
            source_documents (list): List of source document dictionaries.
        Returns:
            float: Average credibility score between 0 and 1.
        """
        if not source_documents:
            return 0.5  # Neutral score if no sources

        # Define simple credibility heuristics (expand these lists significantly)
        high_credibility_tlds = {'.gov', '.edu'}  # TLDs often associated with higher credibility
        medium_credibility_tlds = {'.org'}  # Often credible, but variable

        # Well-known reputable domains (examples, needs much larger curated list)
        known_high_credibility_domains = {
            'wikipedia.org', 'who.int', 'cdc.gov', 'nih.gov', 'nasa.gov',
            'nature.com', 'science.org', 'cell.com', 'thelancet.com',
            'bbc.com', 'bbc.co.uk', 'reuters.com', 'apnews.com', 'nytimes.com',
            'wsj.com', 'arxiv.org'  # Pre-print server, often reliable but not peer-reviewed
        }

        # Domains that might require caution (examples, context dependent)
        known_lower_credibility_patterns = [
            r'.blogspot.com', r'.wordpress.com',  # General blog platforms
            # Add patterns for known unreliable news sources, forums, etc.
        ]

        credibility_scores = []
        num_valid_domains = 0

        for doc in source_documents:
            score = 0.5  # Start with a neutral score
            domain = None
            if isinstance(doc, dict) and 'url' in doc:
                url = doc.get('url')
                if url and isinstance(url, str):
                    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    if domain_match:
                        domain = domain_match.group(1).lower().split(':')[0]  # Normalize domain

            if domain:
                num_valid_domains += 1

                # Check known high-credibility domains first
                is_known_high = False
                for known_domain in known_high_credibility_domains:
                    if domain == known_domain or domain.endswith('.' + known_domain):
                        score = 0.9  # Assign high score
                        is_known_high = True
                        break

                if is_known_high:
                    credibility_scores.append(score)
                    continue  # Skip further checks if known high

                # Check TLDs
                tld_found = False
                for tld in high_credibility_tlds:
                    if domain.endswith(tld):
                        score = max(score, 0.8)  # Boost score for .gov, .edu
                        tld_found = True
                        break

                if not tld_found:
                    for tld in medium_credibility_tlds:
                        if domain.endswith(tld):
                            score = max(score, 0.65)  # Boost slightly for .org
                            break

                # Check for known lower credibility patterns
                is_known_low = False
                for pattern in known_lower_credibility_patterns:
                    if re.search(pattern, domain):
                        score = min(score, 0.3)  # Penalize score
                        is_known_low = True
                        break

                if is_known_low:
                    credibility_scores.append(score)
                    continue

                # If no specific rule matched, keep the score adjusted by TLD or default 0.5
                credibility_scores.append(score)
            else:
                # If no domain, assign a slightly lower default score
                credibility_scores.append(0.4)

        # Calculate average credibility
        if not credibility_scores:
            return 0.5  # Return neutral if no scores could be calculated

        avg_credibility = sum(credibility_scores) / len(credibility_scores)
        return max(0.0, min(1.0, avg_credibility))

    def check_hallucination_markers(self, text: str) -> Dict[str, Any]:
        """
        Check for linguistic markers often associated with potential hallucinations
        (uncertainty, contradictions, vagueness).
        Args:
            text (str): Text to analyze.
        Returns:
            dict: Analysis of hallucination markers, including lists of matches and an overall risk score (0-1).
        """
        if not text or not isinstance(text, str):
            return {
                'uncertain_language': [],
                'contradictions': [],
                'vague_statements': [],
                'overall_risk_score': 0.0
            }

        # 1. Check for uncertain language ("hedge words")
        uncertainty_phrases = [
            'might be', 'could be', 'may have', 'perhaps', 'possibly', 'seems',
            'appears to', 'likely', 'probably', 'potentially', 'suggests',
            'supposedly', 'allegedly', 'reportedly', 'it is said', 'some believe'
        ]

        uncertain_matches = []
        for phrase in uncertainty_phrases:
            # Use word boundaries (\b) to avoid matching parts of words
            try:
                # Added try-except for complex regex patterns if needed
                matches = re.finditer(r'\b' + re.escape(phrase) + r'\b', text, re.IGNORECASE)
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    context = text[start:end].strip().replace("\n", " ")
                    uncertain_matches.append({
                        'phrase': match.group(0),
                        'context': f"...{context}..."
                    })
            except re.error as e:
                logger.warning(f"Regex error checking uncertainty phrase '{phrase}': {e}")

        # 2. Check for simple contradictions (A but not A, X however Y)
        # These patterns are very basic and prone to false positives/negatives.
        contradiction_patterns = [
            # Pattern: finds "X but/however/although ... not X" or similar negations within a window
            r'(\b\w+\b)\s+(?:but|however|although)\s+(?:.*\s)?(?:is not|are not|was not|were not|isn\'t|aren\'t|wasn\'t|weren\'t)\s+\1\b',
            # Pattern: finds contrasting ideas like "X ... however Y" where Y is different concept
            r'(\b[A-Z][a-z]+\b\s+.*\b(?:but|however|yet)\b.*\b(?:different|opposite|contrary)\b)',
            # Pattern: "not only X but also Y" (less contradiction, more nuance) - maybe ignore
        ]

        contradictions = []
        for pattern in contradiction_patterns:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)  # DOTALL allows . to match newline
                for match in matches:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip().replace("\n", " ")
                    contradictions.append(f"...{context}...")
            except re.error as e:
                logger.warning(f"Regex error checking contradiction pattern: {e}")

        # 3. Check for vague statements
        vague_phrases = [
            'somehow', 'somewhat', 'various', 'many', 'some', 'a few', 'several',
            'often', 'sometimes', 'usually', 'generally', 'typically',
            'people say', 'experts believe', 'it is thought', 'in general', 'basically'
        ]

        vague_statements = []
        for phrase in vague_phrases:
            try:
                # Match vague phrases, potentially followed by common nouns like 'people', 'experts', 'sources'
                matches = re.finditer(r'\b' + re.escape(phrase) + r'(?:\s+(?:people|experts|sources|studies|reports))?\b', text, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip().replace("\n", " ")
                    vague_statements.append(f"...{context}...")
            except re.error as e:
                logger.warning(f"Regex error checking vague phrase '{phrase}': {e}")

        # Calculate overall hallucination risk score based on marker counts
        # Score increases with marker frequency, normalized by text length.
        num_markers = len(uncertain_matches) + len(contradictions) + len(vague_statements)

        # Normalize score (e.g., consider 1 marker per 100 words as high risk)
        num_words = len(text.split())
        if num_words == 0:
            overall_score = 0.0
        else:
            # Simple density score: (markers / words) * scale_factor (e.g., 100 for score per 100 words)
            # Let's target score=1.0 if density is >= 1 marker per 50 words (0.02)
            density = num_markers / num_words
            overall_score = min(1.0, density / 0.02)

        return {
            'uncertain_language': uncertain_matches,
            'contradictions': contradictions,  # List of context strings
            'vague_statements': vague_statements,  # List of context strings
            'overall_risk_score': overall_score  # Renamed for clarity
        }

    def clear_cache(self) -> None:
        """
        Clear the internal analysis caches.
        """
        self._results_cache.cache_clear()
        self._term_cache.cache_clear()
        self._statement_cache.cache_clear()
        logger.info("Result analyzer caches cleared.")

    def get_result_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a user-friendly summary dictionary from the detailed analysis result.
        Args:
            analysis_result (dict): The full analysis result dictionary.
        Returns:
            dict: A simplified summary with 'confidence' level, 'issues' list,
                'verified_count', and 'potential_issues_count'.
        """
        if not analysis_result or not isinstance(analysis_result, dict):
            return {
                'confidence': 'Unknown',
                'issues': ['No analysis available or analysis failed'],
                'verified_count': 0,
                'potential_issues_count': 0
            }

        # Check if this is a counterfactual analysis
        if analysis_result.get('is_counterfactual', False):
            # Create special summary for counterfactual analysis
            counterfactual_summary = {
                'confidence': 'Counterfactual',
                'issues': [],
                'verified_count': len(analysis_result.get('verified_facts', [])),
                'potential_issues_count': 0,
                'is_counterfactual': True,
                'counterfactual_condition': analysis_result.get('counterfactual_analysis', {}).get('condition', 'Unknown condition'),
                'alternative_outcome': 'Based on counterfactual reasoning'
            }
            
            # Check confidence level of counterfactual analysis
            cf_confidence = analysis_result.get('counterfactual_analysis', {}).get('confidence', 0.5)
            if cf_confidence < self.config.get('counterfactual_confidence_threshold', 0.7):
                counterfactual_summary['issues'].append('Low confidence in counterfactual reasoning')
                counterfactual_summary['potential_issues_count'] = 1
                
            return counterfactual_summary

        # Standard analysis summary
        # Determine confidence level text
        confidence_score = analysis_result.get('confidence_score', 0)
        if confidence_score >= 0.8:
            confidence_level = 'High'
        elif confidence_score >= 0.6:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'

        # Collect potential issues based on thresholds
        issues = []
        min_sources = self.config.get('min_sources_required', 2)

        # FIX: Check if source_diversity is a dictionary or a float
        num_docs_analyzed = 0
        if isinstance(analysis_result.get('source_diversity'), dict):
            num_docs_analyzed = len(analysis_result.get('source_diversity', {}).get('domains', []))
        else:
            # If source_diversity is a float (or not a dict), count documents directly
            num_docs_analyzed = len(analysis_result.get('documents', []))

        # Check for potential hallucinations/unverified statements
        hallucinations = analysis_result.get('potential_hallucinations', [])
        if hallucinations:
            # Check if it's just due to insufficient sources
            if len(hallucinations) == 1 and hallucinations[0] == 'Insufficient sources to verify content':
                issues.append('Result could not be verified due to insufficient sources.')
            elif len(hallucinations) > 0:
                issues.append(f'Found {len(hallucinations)} statement(s) that could not be verified against sources.')

        # Check source diversity (example threshold)
        source_diversity = analysis_result.get('source_diversity', 0)
        if source_diversity < 0.4 and num_docs_analyzed >= min_sources:
            issues.append('Limited variety of information sources detected.')

        # Check factual consistency (example threshold)
        factual_consistency = analysis_result.get('factual_consistency', 0)
        if factual_consistency < 0.5 and num_docs_analyzed >= min_sources:
            issues.append('Information may not be consistently supported across sources.')

        # Check source credibility (example threshold)
        source_credibility = analysis_result.get('source_credibility', 0)
        if source_credibility < 0.6 and num_docs_analyzed > 0:
            issues.append('Some sources may have lower credibility.')

        # Check for hallucination markers if that analysis was run
        marker_analysis = analysis_result.get('hallucination_markers')
        if marker_analysis and isinstance(marker_analysis, dict):
            marker_score = marker_analysis.get('overall_risk_score', 0.0)
            if marker_score > 0.6:  # Example threshold for high marker density
                issues.append('Detected language patterns sometimes associated with less factual content.')

        # If confidence is low but no specific issues found, add a general note
        if confidence_level == 'Low' and not issues:
            issues.append('Confidence is low based on overall analysis metrics.')

        return {
            'confidence': confidence_level,
            'issues': issues,
            'verified_count': len(analysis_result.get('verified_facts', [])),
            'potential_issues_count': len(hallucinations) if hallucinations and hallucinations[0] != 'Insufficient sources to verify content' else 0
        }


# ----- Counterfactual Reasoning Implementation -----

class CounterfactualReasoner:
    """
    Processes counterfactual 'what-if' queries through causal reasoning.
    
    This class implements a sophisticated reasoning system that:
    1. Detects counterfactual queries (e.g., "What if X had happened instead of Y?")
    2. Extracts the counterfactual conditions and assumptions
    3. Builds causal graphs from factual knowledge
    4. Applies counterfactual logic to trace alternate outcomes
    5. Generates plausible alternate world states based on causal inference
    
    This enables the search engine to answer hypothetical queries that traditional
    search engines cannot handle, providing deeper reasoning capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the counterfactual reasoner.
        
        Args:
            config: Configuration dictionary with optional parameters
        """
        self.config = config or {}
        
        # Initialize causal model and variables
        self.causal_model = self._initialize_causal_model()
        self.causal_graph = nx.DiGraph()  # Directed graph for representing causal relationships
        
        # Regular expression pattern for detecting counterfactual queries
        self.counterfactual_detector = re.compile(
            r"what\s+if|if\s+(?:only|instead)|(?:had|would|could)\s+have\s+been|suppose\s+that|"
            r"assuming\s+that|imagine\s+if|in\s+a\s+world\s+where|alternate|alternative",
            re.IGNORECASE
        )
        
        # Patterns to extract counterfactual condition
        self.condition_extractors = [
            # "What if X had happened?"
            re.compile(r"what\s+if\s+(.*?)(?:\?|$)", re.IGNORECASE),
            # "If X had happened instead of Y"
            re.compile(r"if\s+(.*?)\s+(?:had|were|was)\s+(.*?)(?:instead|rather)", re.IGNORECASE),
            # "Imagine if X"
            re.compile(r"imagine\s+(?:if|that)\s+(.*?)(?:\?|$|,|\.|;)", re.IGNORECASE),
            # "Assuming that X"
            re.compile(r"assuming\s+that\s+(.*?)(?:\?|$|,|\.|;)", re.IGNORECASE),
            # "In a world where X"
            re.compile(r"in\s+a\s+world\s+where\s+(.*?)(?:\?|$|,|\.|;)", re.IGNORECASE),
        ]
        
        # Maps for causal effects (effect chains)
        self.causal_effects = defaultdict(list)  # variable -> affected variables
        
        # Temporal reasoning markers (for ordering events in causal chains)
        self.temporal_markers = {
            'before': -1,  # earlier in time
            'after': 1,     # later in time 
            'during': 0,    # simultaneous
            'while': 0,     # simultaneous
            'since': 1,     # later than reference
            'until': -1,    # earlier than reference
        }
        
        # Standard causal verbs and phrases (for detecting causal statements)
        self.causal_markers = [
            # Strong causation
            "causes", "caused by", "leads to", "results in", "creates", "produces",
            "triggers", "induces", "generates", "brings about", "effect of",
            
            # Weaker or partial causation 
            "contributes to", "influences", "affects", "impacts", "plays a role in",
            "is a factor in", "is associated with", "correlates with", 
            
            # Prevention or blocking
            "prevents", "stops", "blocks", "inhibits", "reduces", "decreases",
            "diminishes", "limits", "constrains", "restricts",
            
            # Enhancement
            "increases", "enhances", "amplifies", "accelerates", "promotes",
            "facilitates", "enables", "allows", "permits",
            
            # Conditional causation
            "if", "when", "whenever", "provided that", "assuming that", "in cases where"
        ]
        
        # Certainty modifiers for adjusting confidence in causal relationships
        self.certainty_modifiers = {
            # High certainty (strengthen causal link)
            "definitely": 1.3, "certainly": 1.3, "always": 1.3, "invariably": 1.3,
            "necessarily": 1.25, "undoubtedly": 1.25, "inevitably": 1.25,
            
            # Medium certainty (neutral)
            "generally": 1.0, "typically": 1.0, "usually": 1.0, "often": 1.0,
            "frequently": 1.0, "commonly": 1.0,
            
            # Low certainty (weaken causal link)
            "sometimes": 0.7, "occasionally": 0.7, "may": 0.6, "might": 0.6,
            "could": 0.6, "possibly": 0.5, "perhaps": 0.5, "potentially": 0.6
        }
        
        # Track counterfactual queries processed
        self.history = []
    
    def _initialize_causal_model(self) -> Dict[str, Any]:
        """Initialize the base causal model used for reasoning."""
        return {
            'variables': {},  # Variables observed in the system
            'base_relations': {},  # Basic causal relationships
            'domain_knowledge': {},  # Domain-specific causal models
        }
    
    def is_counterfactual_query(self, query: str) -> bool:
        """
        Determine if a query requires counterfactual reasoning.
        
        Args:
            query: The search query
            
        Returns:
            bool: True if the query is counterfactual in nature
        """
        if not query:
            return False
        
        # Check using regular expression pattern
        if self.counterfactual_detector.search(query):
            return True
            
        # Check for more subtle counterfactual indicators
        lower_query = query.lower()
        subtle_indicators = [
            " would happen ", " would change ", " would be different ",
            "would have been", "could have been", "might have been", 
            "instead of", "rather than", "if not for", "if only"
        ]
        
        for indicator in subtle_indicators:
            if indicator in lower_query:
                return True
                
        return False
    
    def _extract_counterfactual_condition(self, query: str) -> str:
        """
        Extract the counterfactual condition from the query.
        
        Args:
            query: The counterfactual query
            
        Returns:
            str: The extracted condition, or empty string if no clear condition found
        """
        # Try each extraction pattern
        for pattern in self.condition_extractors:
            match = pattern.search(query)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, use a simple heuristic
        # For queries like "What if X?" extract X
        if query.lower().startswith("what if "):
            # Remove "what if " and any trailing punctuation
            condition = query[8:].strip()
            condition = re.sub(r'[?!.]*$', '', condition).strip()
            return condition
            
        # For "If X, then Y" format
        match = re.search(r"if\s+(.*?)(?:,|then)", query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        # Fall back to the query itself without common prefixes
        clean_query = query
        prefixes = ["what if", "if", "imagine", "consider", "suppose", "assuming"]
        for prefix in prefixes:
            if clean_query.lower().startswith(prefix):
                clean_query = clean_query[len(prefix):].strip()
                break
                
        # Remove trailing punctuation
        clean_query = re.sub(r'[?!.]*$', '', clean_query).strip()
        
        return clean_query
    
    def _identify_variables_and_values(self, condition: str) -> Dict[str, Any]:
        """
        Identify variables and their counterfactual values from a condition.
        
        Args:
            condition: The counterfactual condition
            
        Returns:
            dict: Dictionary of {variable: value} pairs
        """
        variables = {}
        
        # Check for patterns like "X had been Y" or "X was Y" or "X is Y"
        is_patterns = [
            r"(\b[A-Za-z0-9\s]+\b)\s+(?:had|were|was)\s+(?:been\s+)?(\b[A-Za-z0-9\s]+\b)",
            r"(\b[A-Za-z0-9\s]+\b)\s+(?:is|are|became)\s+(\b[A-Za-z0-9\s]+\b)",
            r"(\b[A-Za-z0-9\s]+\b)\s+(?:increased|decreased|changed|reduced|grew)\s+(?:to|by)?\s+(\b[A-Za-z0-9\s]+\b)"
        ]
        
        for pattern in is_patterns:
            matches = re.findall(pattern, condition, re.IGNORECASE)
            for var, val in matches:
                variables[var.strip()] = val.strip()
        
        # If no structured pattern matches, treat the entire condition as a single variable-value pair
        if not variables:
            if " had " in condition:
                parts = condition.split(" had ", 1)
                if len(parts) == 2:
                    variables[parts[0].strip()] = parts[1].strip()
            else:
                variables['condition'] = condition.strip()
                
        return variables
    
    def _extract_entities_and_relationships(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str, float]]]:
        """
        Extract entities and causal relationships from text.
        
        Args:
            text: Source text to analyze
            
        Returns:
            tuple: (entities, relationships) where relationships are (source, target, type, weight)
        """
        # Simple entity extraction (could be enhanced with NER models)
        entities = set()
        
        # Extract noun phrases as potential entities
        np_pattern = r'\b([A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)*)\b'
        noun_phrases = re.findall(np_pattern, text)
        entities.update(noun_phrases)
        
        # Extract causal relationships
        relationships = []
        
        # Check for causal markers
        for marker in self.causal_markers:
            # Pattern looks for "X {marker} Y" structure
            pattern = rf'([\w\s]+)\s+{re.escape(marker)}\s+([\w\s]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for source, target in matches:
                source = source.strip()
                target = target.strip()
                if source and target and source != target:
                    # Check for certainty modifiers
                    certainty = 0.8  # Default certainty
                    for mod, val in self.certainty_modifiers.items():
                        if mod in source.lower() or mod in target.lower():
                            certainty *= val
                            break
                    
                    relationships.append((source, target, "causes", min(1.0, certainty)))
        
        # Remove duplicates
        entities = list(entities)
        
        return entities, relationships
    
    def _build_causal_graph(self, documents: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a causal graph from source documents.
        
        Args:
            documents: Source documents to extract causal information from
            
        Returns:
            networkx.DiGraph: Directed graph of causal relationships
        """
        G = nx.DiGraph()
        
        # Process each document
        for doc in documents:
            if isinstance(doc, dict):
                # Extract text content
                text = ""
                if 'title' in doc:
                    text += doc['title'] + " "
                if 'snippet' in doc:
                    text += doc['snippet'] + " "
                if 'text' in doc:
                    text += doc['text']
            elif isinstance(doc, str):
                text = doc
            else:
                continue
                
            # Extract entities and relationships
            entities, relationships = self._extract_entities_and_relationships(text)
            
            # Add entities as nodes
            for entity in entities:
                if entity not in G:
                    G.add_node(entity, type='entity')
            
            # Add relationships as edges
            for source, target, rel_type, weight in relationships:
                # Only add if both entities exist or are new
                if source not in G:
                    G.add_node(source, type='entity')
                if target not in G:
                    G.add_node(target, type='entity')
                    
                # Add or update edge
                if G.has_edge(source, target):
                    # Update weight if the new relationship is stronger
                    current_weight = G[source][target]['weight']
                    if weight > current_weight:
                        G[source][target]['weight'] = weight
                else:
                    G.add_edge(source, target, type=rel_type, weight=weight)
        
        return G
    
    def _trace_causal_effects(self, altered_variables: Dict[str, Any], causal_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Trace the causal effects of altered variables through the graph.
        
        Args:
            altered_variables: Dictionary of {variable: new_value} pairs
            causal_graph: The causal graph to trace through
            
        Returns:
            dict: Effects of the counterfactual condition
        """
        effects = {
            'direct_effects': [],   # First-order effects
            'indirect_effects': [], # Higher-order effects
            'uncertain_effects': [] # Effects with low confidence
        }
        
        # Check each altered variable
        for variable, new_value in altered_variables.items():
            # Find the closest node in the graph (exact or approximate match)
            closest_node = self._find_closest_node(variable, causal_graph)
            
            if not closest_node:
                effects['uncertain_effects'].append({
                    'cause': variable,
                    'new_value': new_value,
                    'effect': f"Unknown effects of changes to '{variable}'",
                    'confidence': 0.3
                })
                continue
                
            # Check direct successors (immediate effects)
            for successor in causal_graph.successors(closest_node):
                edge_data = causal_graph[closest_node][successor]
                
                # Record effect with confidence based on edge weight
                effects['direct_effects'].append({
                    'cause': variable,
                    'new_value': new_value,
                    'affected_entity': successor,
                    'relation_type': edge_data.get('type', 'affects'),
                    'confidence': edge_data.get('weight', 0.5)
                })
                
                # Trace indirect effects (up to 2 steps away for simplicity)
                for indirect in causal_graph.successors(successor):
                    if indirect != closest_node:  # Avoid cycles
                        indirect_edge = causal_graph[successor][indirect]
                        # Confidence decreases with distance
                        indirect_confidence = edge_data.get('weight', 0.5) * indirect_edge.get('weight', 0.5) * 0.8
                        
                        if indirect_confidence > 0.4:  # Only include reasonably confident indirect effects
                            effects['indirect_effects'].append({
                                'cause': variable,
                                'intermediate': successor,
                                'affected_entity': indirect,
                                'relation_type': indirect_edge.get('type', 'affects'),
                                'confidence': indirect_confidence
                            })
        
        return effects
    
    def _find_closest_node(self, variable: str, graph: nx.DiGraph) -> Optional[str]:
        """
        Find the closest matching node in the graph for a variable.
        
        Args:
            variable: Variable to find
            graph: Graph to search in
            
        Returns:
            str: Closest matching node, or None if no good match
        """
        # Try exact match first
        if variable in graph:
            return variable
            
        # Try case-insensitive match
        variable_lower = variable.lower()
        for node in graph.nodes():
            if node.lower() == variable_lower:
                return node
                
        # Try partial matches
        best_score = 0.6  # Minimum threshold for partial match
        best_match = None
        
        for node in graph.nodes():
            # Simple substring match score
            var_words = set(variable_lower.split())
            node_words = set(node.lower().split())
            
            # Jaccard similarity
            intersection = len(var_words.intersection(node_words))
            union = len(var_words.union(node_words))
            
            if union > 0:
                score = intersection / union
                if score > best_score:
                    best_score = score
                    best_match = node
        
        return best_match
    
    def _derive_alternate_outcome(self, effects: Dict[str, Any], result_text: str) -> str:
        """
        Derive an alternate outcome based on counterfactual effects.
        
        Args:
            effects: Dictionary of causal effects
            result_text: Original result text to adapt
            
        Returns:
            str: Description of the alternate outcome
        """
        # Start with an introduction
        outcome = "Based on counterfactual reasoning, if this condition were true:\n\n"
        
        # Add direct effects
        if effects['direct_effects']:
            outcome += "Direct effects would likely include:\n"
            for effect in effects['direct_effects']:
                confidence_str = "high" if effect['confidence'] > 0.8 else \
                                "moderate" if effect['confidence'] > 0.5 else "possible"
                outcome += f"- {effect['affected_entity']} would be affected " + \
                          f"({confidence_str} confidence)\n"
            outcome += "\n"
            
        # Add indirect effects
        if effects['indirect_effects']:
            outcome += "Further consequences might include:\n"
            for effect in effects['indirect_effects']:
                confidence_str = "likely" if effect['confidence'] > 0.7 else \
                                "possibly" if effect['confidence'] > 0.4 else "uncertain"
                outcome += f"- {effect['affected_entity']} would {confidence_str} be affected through " + \
                          f"changes to {effect['intermediate']}\n"
            outcome += "\n"
            
        # Add uncertain effects
        if effects['uncertain_effects']:
            outcome += "Additional effects are difficult to determine due to limited information.\n\n"
            
        # Add a note about the confidence of this counterfactual analysis
        overall_confidence = self._calculate_counterfactual_confidence(effects)
        confidence_desc = "high" if overall_confidence > 0.8 else \
                          "moderate" if overall_confidence > 0.6 else \
                          "limited" if overall_confidence > 0.4 else "very low"
                          
        outcome += f"Note: This counterfactual analysis has {confidence_desc} confidence based on " + \
                  f"available causal information."
        
        return outcome
    
    def _calculate_counterfactual_confidence(self, effects: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for the counterfactual analysis.
        
        Args:
            effects: Dictionary of causal effects
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # No effects means very low confidence
        if not effects['direct_effects'] and not effects['indirect_effects']:
            return 0.2
            
        # Calculate weighted average of effect confidences
        total_confidence = 0
        total_weight = 0
        
        # Direct effects have higher weight
        for effect in effects['direct_effects']:
            total_confidence += effect['confidence'] * 2
            total_weight += 2
            
        # Indirect effects have lower weight
        for effect in effects['indirect_effects']:
            total_confidence += effect['confidence']
            total_weight += 1
            
        # Calculate average, defaulting to 0.3 if no valid effects
        if total_weight > 0:
            avg_confidence = total_confidence / total_weight
        else:
            avg_confidence = 0.3
            
        # Scale confidence based on number of effects identified
        # More effects generally means more complete analysis
        effect_count = len(effects['direct_effects']) + len(effects['indirect_effects'])
        
        # Bonus for having more effects, up to a point
        if effect_count >= 5:
            scaling_factor = 1.1  # Slight boost for rich analysis
        elif effect_count >= 3:
            scaling_factor = 1.0  # Neutral
        elif effect_count >= 1:
            scaling_factor = 0.9  # Slight penalty for sparse analysis
        else:
            scaling_factor = 0.7  # Larger penalty for no effects
            
        final_confidence = avg_confidence * scaling_factor
        
        # Ensure confidence is within bounds
        return max(0.1, min(1.0, final_confidence))
    
    def process_query(self, query: str, result_text: str, source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a counterfactual query through causal reasoning.
        
        Args:
            query: The counterfactual query
            result_text: The original search result text
            source_documents: Source documents for building causal knowledge
            
        Returns:
            dict: Counterfactual analysis results
        """
        logger.info(f"Processing counterfactual query: {query}")
        
        # Extract the counterfactual condition
        condition = self._extract_counterfactual_condition(query)
        logger.debug(f"Extracted counterfactual condition: {condition}")
        
        # Identify variables and their values
        altered_variables = self._identify_variables_and_values(condition)
        logger.debug(f"Identified altered variables: {altered_variables}")
        
        # Build causal graph from documents
        causal_graph = self._build_causal_graph(source_documents)
        
        # Calculate some graph metrics for logging
        num_nodes = causal_graph.number_of_nodes()
        num_edges = causal_graph.number_of_edges()
        logger.debug(f"Built causal graph with {num_nodes} nodes and {num_edges} edges")
        
        # Trace causal effects
        effects = self._trace_causal_effects(altered_variables, causal_graph)
        
        # Derive alternate outcome
        alternate_outcome = self._derive_alternate_outcome(effects, result_text)
        
        # Calculate overall confidence
        confidence = self._calculate_counterfactual_confidence(effects)
        
        # Record this query in history
        self.history.append({
            'query': query,
            'condition': condition,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence
        })
        
        # Return the counterfactual analysis
        return {
            'condition': condition,
            'altered_variables': altered_variables,
            'effects': effects,
            'alternate_outcome': alternate_outcome,
            'confidence': confidence,
            'graph_stats': {
                'nodes': num_nodes,
                'edges': num_edges
            }
        }


# --- Singleton Accessor ---
_analyzer_instance: Optional[ResultAnalyzer] = None
_analyzer_lock = threading.Lock()

def get_result_analyzer(config: Optional[Dict[str, Any]] = None) -> ResultAnalyzer:
    """
    Get a singleton instance of the ResultAnalyzer.
    Uses provided config only on first initialization.
    Args:
        config (dict, optional): Configuration settings dictionary, typically
            from config_manager.get_config().
    Returns:
        ResultAnalyzer: The singleton ResultAnalyzer instance.
    """
    global _analyzer_instance

    if _analyzer_instance is None:
        with _analyzer_lock:
            # Double-check inside lock
            if _analyzer_instance is None:
                logger.info("Creating ResultAnalyzer singleton instance.")
                # Extract analyzer-specific config if nested (e.g., under 'analyzer' key)
                analyzer_config = config or {}
                if 'analyzer' in analyzer_config and isinstance(analyzer_config['analyzer'], dict):
                    analyzer_config = analyzer_config['analyzer']
                elif 'result_analyzer' in analyzer_config and isinstance(analyzer_config['result_analyzer'], dict):
                    analyzer_config = analyzer_config['result_analyzer']

                _analyzer_instance = ResultAnalyzer(config=analyzer_config)

    return _analyzer_instance

# --- Convenience Functions (using the singleton) ---

def analyze_search_result(
    result_text: str,
    source_documents: List[Dict[str, Any]],
    query: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None  # Config passed here is only used for initial singleton creation
) -> Dict[str, Any]:
    """
    Analyze a search result using the global singleton ResultAnalyzer.
    Args:
        result_text (str): The result text.
        source_documents (list): List of source document dictionaries.
        query (str, optional): The original query.
        config: Analyzer configuration (used only for first-time singleton init).
    Returns:
        dict: Analysis results.
    """
    try:
        analyzer = get_result_analyzer(config)
        return analyzer.analyze_result(result_text, source_documents, query)
    except Exception as e:
        logger.error(f"Error analyzing result via convenience function: {e}", exc_info=True)
        # Return default error structure
        return {
            'confidence_score': 0.0, 'potential_hallucinations': [f'Analysis failed: {e}'],
            'verified_facts': [], 'source_diversity': 0.0, 'factual_consistency': 0.0,
            'source_credibility': 0.0, 'key_statements': [], 'error': str(e)
        }

def get_analysis_summary(
    analysis_result: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None  # Config passed here is only used for initial singleton creation
) -> Dict[str, Any]:
    """
    Get a simplified summary of the analysis result using the global singleton ResultAnalyzer.
    Args:
        analysis_result (dict): The detailed analysis result.
        config: Analyzer configuration (used only for first-time singleton init).
    Returns:
        dict: Simplified summary.
    """
    try:
        analyzer = get_result_analyzer(config)
        return analyzer.get_result_summary(analysis_result)
    except Exception as e:
        logger.error(f"Error getting analysis summary: {e}", exc_info=True)
        return {'confidence': 'Unknown', 'issues': [f'Summary generation failed: {e}'], 'verified_count': 0, 'potential_issues_count': 0}
