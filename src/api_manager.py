# api_manager.py
"""
API Manager Module for AI-powered Search

This module provides:
1. Centralized API key management via .env files
2. Lazy initialization of API clients
3. Singleton pattern for global access
4. Availability checks for optional provider libraries
5. Basic error handling during client initialization
"""

import os
import logging
import threading # Using threading for lock instead of lru_cache for clarity
from typing import Dict, Any, Optional, Type # Use Type for client classes

from dotenv import load_dotenv

# Optional imports for different providers
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    cohere = None # Set to None if not available
    COHERE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None # Set to None if not available
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None # Set to None if not available
    ANTHROPIC_AVAILABLE = False

# Configure logging
# Corrected basicConfig call
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_manager")

# Load environment variables from .env file
load_dotenv()

class APIManager:
    """
    Manages API keys and client initialization for various AI providers.
    Uses a singleton pattern.
    """

    # Define expected environment variable names
    ENV_VAR_MAP = {
        "cohere": "COHERE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "web_api": "WEB_API_KEY", # Generic key for custom web APIs
        "web_api_url": "WEB_API_URL" # Optional URL for the generic web API
    }

    def __init__(self):
        """Initialize API manager, load keys, prepare client storage."""
        self.api_keys: Dict[str, Optional[str]] = {}
        self.clients: Dict[str, Any] = {} # Store initialized clients
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from environment variables based on ENV_VAR_MAP."""
        logger.info("Loading API keys from environment variables...")
        for provider, env_var in self.ENV_VAR_MAP.items():
            key = os.environ.get(env_var)
            self.api_keys[provider] = key
            if key:
                # Avoid logging the key itself, just its presence
                logger.info(f"{provider.capitalize()} API key found (via {env_var})")
            else:
                logger.warning(f"{provider.capitalize()} API key not found (env var {env_var} missing or empty)")

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get the API key for a specific provider.

        Args:
            provider (str): The name of the provider (e.g., 'cohere', 'openai').

        Returns:
            Optional[str]: The API key if found, otherwise None.
        """
        return self.api_keys.get(provider)

    def get_cohere_client(self) -> Optional['cohere.Client']:
        """
        Get or initialize the Cohere client.

        Returns:
            Optional[cohere.Client]: The initialized client or None if unavailable/failed.
        """
        provider = "cohere"
        if not COHERE_AVAILABLE:
            logger.debug(f"Cannot get {provider} client: cohere library not installed.")
            return None

        # Lazy initialization
        if provider not in self.clients:
            api_key = self.get_api_key(provider)
            if api_key:
                try:
                    client = cohere.Client(api_key)
                    self.clients[provider] = client
                    logger.info(f"{provider.capitalize()} client initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize {provider.capitalize()} client: {e}", exc_info=True)
                    self.clients[provider] = None # Mark as failed
            else:
                logger.warning(f"Cannot initialize {provider.capitalize()} client: API key not found.")
                self.clients[provider] = None # Mark as failed (no key)

        return self.clients.get(provider) # Return client instance or None if init failed

    def get_openai_client(self) -> Optional['openai.OpenAI']:
        """
        Get or initialize the OpenAI client. Uses the modern OpenAI() class.

        Returns:
            Optional[openai.OpenAI]: The initialized client or None if unavailable/failed.
        """
        provider = "openai"
        if not OPENAI_AVAILABLE:
            logger.debug(f"Cannot get {provider} client: openai library not installed.")
            return None

        # Lazy initialization
        if provider not in self.clients:
            api_key = self.get_api_key(provider)
            if api_key:
                try:
                    # Use the modern client initialization
                    client = openai.OpenAI(api_key=api_key)
                    # Perform a simple check if possible (e.g., list models) to validate key early
                    # client.models.list() # Uncomment to test key on init, might add latency/cost
                    self.clients[provider] = client
                    logger.info(f"{provider.capitalize()} client initialized successfully.")
                except openai.AuthenticationError as e:
                     logger.error(f"Failed to initialize {provider.capitalize()} client: Authentication Error - Invalid API Key? Details: {e}", exc_info=False) # Don't log stack trace for auth error
                     self.clients[provider] = None
                except Exception as e:
                    logger.error(f"Failed to initialize {provider.capitalize()} client: {e}", exc_info=True)
                    self.clients[provider] = None # Mark as failed
            else:
                logger.warning(f"Cannot initialize {provider.capitalize()} client: API key not found.")
                self.clients[provider] = None # Mark as failed (no key)

        return self.clients.get(provider) # Return client instance or None if init failed

    def get_anthropic_client(self) -> Optional['anthropic.Anthropic']:
        """
        Get or initialize the Anthropic client.

        Returns:
            Optional[anthropic.Anthropic]: The initialized client or None if unavailable/failed.
        """
        provider = "anthropic"
        if not ANTHROPIC_AVAILABLE:
            logger.debug(f"Cannot get {provider} client: anthropic library not installed.")
            return None

        # Lazy initialization
        if provider not in self.clients:
            api_key = self.get_api_key(provider)
            if api_key:
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    # Perform a simple check if possible (e.g., check connection)
                    # client.count_tokens("test") # Example check
                    self.clients[provider] = client
                    logger.info(f"{provider.capitalize()} client initialized successfully.")
                except anthropic.AuthenticationError as e:
                     logger.error(f"Failed to initialize {provider.capitalize()} client: Authentication Error - Invalid API Key? Details: {e}", exc_info=False)
                     self.clients[provider] = None
                except Exception as e:
                    logger.error(f"Failed to initialize {provider.capitalize()} client: {e}", exc_info=True)
                    self.clients[provider] = None # Mark as failed
            else:
                logger.warning(f"Cannot initialize {provider.capitalize()} client: API key not found.")
                self.clients[provider] = None # Mark as failed (no key)

        return self.clients.get(provider) # Return client instance or None if init failed

    def get_web_api_config(self) -> Optional[Dict[str, str]]:
        """
        Get configuration relevant for a generic Web API provider.

        Returns:
            Optional[Dict[str, str]]: Dictionary containing 'api_key' and 'api_url' if found, otherwise None.
                                      Returns None if the key is missing.
        """
        api_key = self.get_api_key("web_api")
        api_url = self.get_api_key("web_api_url") # Use get_api_key for consistency, loads from WEB_API_URL

        if not api_key:
            logger.debug("Web API config not fully available: API key missing.")
            return None

        config = {
            "api_key": api_key,
            "api_url": api_url or "" # Return empty string if URL is not set
        }
        logger.debug(f"Retrieved Web API config (URL: {'Set' if api_url else 'Not Set'}).")
        return config

# --- Singleton Accessor ---
_api_manager_instance: Optional[APIManager] = None
_api_manager_lock = threading.Lock()

# Using a manual lock-based singleton instead of lru_cache for clarity
def get_api_manager() -> APIManager:
    """
    Get the singleton instance of the APIManager.
    Initializes it on the first call. Thread-safe.

    Returns:
        APIManager: The singleton instance.
    """
    global _api_manager_instance
    if _api_manager_instance is None:
        with _api_manager_lock:
            # Double-check locking pattern
            if _api_manager_instance is None:
                logger.info("Creating APIManager singleton instance.")
                _api_manager_instance = APIManager()
    return _api_manager_instance

