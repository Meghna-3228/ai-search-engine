# cache_manager.py

"""
Cache Manager Module for AI-powered Search

This module provides advanced caching capabilities to improve performance by:

1. Storing and retrieving search results to avoid redundant API calls
2. Implementing intelligent cache invalidation strategies
3. Managing memory and disk storage efficiently
4. Providing metrics on cache performance
5. Supporting multiple caching levels (memory, disk, database)
6. Domain-specific partitioning for specialized index spaces
7. Self-supervised relevance learning from user interactions
8. Reinforcement learning-based result ranking

The goal is to reduce response times and API costs while maintaining result freshness.
"""

import os
import re
import json
import time
import hashlib
import logging
import sqlite3
import threading
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("cache_manager")

class CacheEntry:
    """
    Represents a cached item with metadata
    """
    def __init__(
        self,
        key: str,
        value: Any,
        timestamp: float = None,
        expires_at: float = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a cache entry
        Args:
            key: Unique identifier for the cache entry
            value: The data to be cached
            timestamp: Time when the entry was created (epoch time)
            expires_at: Time when the entry expires (epoch time)
            metadata: Additional information about the entry
        """
        self.key = key
        self.value = value
        self.timestamp = timestamp or time.time()
        self.expires_at = expires_at
        self.metadata = metadata or {}

    def is_expired(self) -> bool:
        """
        Check if the cache entry has expired
        Returns:
            bool: True if expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_age(self) -> float:
        """
        Get the age of the cache entry in seconds
        Returns:
            float: Age in seconds
        """
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the cache entry to a dictionary for serialization
        Returns:
            dict: Dictionary representation of the cache entry
        """
        return {
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp,
            'expires_at': self.expires_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """
        Create a cache entry from a dictionary
        Args:
            data: Dictionary representation of a cache entry
        Returns:
            CacheEntry: A new cache entry object
        """
        # Add basic validation for required keys
        if not all(k in data for k in ['key', 'value', 'timestamp', 'expires_at', 'metadata']):
            raise ValueError("Invalid data format for CacheEntry.from_dict")
        
        return cls(
            key=data['key'],
            value=data['value'],
            timestamp=data['timestamp'],
            expires_at=data['expires_at'],
            metadata=data['metadata']
        )

class MemoryCache:
    """
    In-memory cache implementation
    """
    def __init__(self, max_size: int = 100):
        """
        Initialize an in-memory cache
        Args:
            max_size: Maximum number of entries to store
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()  # Use RLock for potential nested calls within locked methods

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get an entry from the cache
        Args:
            key: The cache key
        Returns:
            CacheEntry: The cache entry if found and not expired, None otherwise
        """
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
                
            if entry.is_expired():
                logger.debug(f"Memory cache entry '{key}' expired. Removing.")
                # Use self.remove to ensure proper cleanup if needed later
                self.remove(key)
                self.misses += 1  # Treat expired as a miss
                return None
                
            self.hits += 1
            # Update access time (optional, for LRU eviction if implemented)
            # entry.last_accessed = time.time()
            return entry

    def set(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an entry to the cache
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (optional)
            metadata: Additional information about the entry (optional)
        """
        with self.lock:
            # Calculate expiration time if TTL is provided
            expires_at = time.time() + ttl if ttl is not None else None
            
            # Create a new cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                expires_at=expires_at,
                metadata=metadata
            )
            
            # Check if adding a new entry will exceed the size limit
            is_update = key in self.cache
            if not is_update and len(self.cache) >= self.max_size:
                self._evict_oldest()
                
            # Add or update the entry
            self.cache[key] = entry
            logger.debug(f"Set memory cache entry '{key}' (TTL: {ttl}s)")

    def remove(self, key: str) -> bool:
        """
        Remove an entry from the cache
        Args:
            key: The cache key
        Returns:
            bool: True if the entry was removed, False if it wasn't in the cache
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Removed memory cache entry '{key}'")
                return True
            return False

    def clear(self) -> None:
        """
        Clear all entries from the cache
        """
        with self.lock:
            self.cache.clear()
            logger.info("Cleared memory cache.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            # Clean expired entries before calculating stats for accuracy
            self.cleanup_expired()
            
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }

    def _evict_oldest(self) -> None:
        """
        Evict the oldest cache entry (based on creation timestamp).
        If multiple entries have the same timestamp, the choice is arbitrary but consistent.
        """
        if not self.cache:
            return
            
        # Find the entry with the minimum timestamp
        try:
            oldest_key = min(self.cache, key=lambda k: self.cache[k].timestamp)
            logger.debug(f"Memory cache full. Evicting oldest entry: '{oldest_key}'")
            self.remove(oldest_key)
        except ValueError:
            # Should not happen if self.cache is not empty, but handles potential edge case
            logger.warning("Could not find oldest entry to evict in non-empty cache.")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        Returns:
            int: Number of entries removed.
        """
        removed_count = 0
        with self.lock:
            # Iterate over a copy of the keys to allow removal during iteration
            keys_to_check = list(self.cache.keys())
            now = time.time()
            
            for key in keys_to_check:
                entry = self.cache.get(key)  # Use get to handle potential race conditions if key was removed
                if entry and entry.expires_at is not None and now > entry.expires_at:
                    if self.remove(key):
                        removed_count += 1
                        
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired entries from memory cache.")
                
            return removed_count

class DiskCache:
    """
    File-based disk cache implementation
    """
    def __init__(self, cache_dir: str = '.cache', max_size_mb: int = 100):
        """
        Initialize a disk cache
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in megabytes
        """
        if max_size_mb <= 0:
            raise ValueError("max_size_mb must be a positive number")
            
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
        # Create cache directory if it doesn't exist
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory '{self.cache_dir}': {e}")
            raise  # Re-raise the error as cache cannot function
            
        # Create index file path
        self.index_file = os.path.join(self.cache_dir, 'index.json')
        
        # Load index if it exists
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the cache index from disk
        Returns:
            dict: Cache index
        """
        if os.path.exists(self.index_file):
            try:
                # Use a lock for file access safety, though primarily for index dict consistency
                with self.lock, open(self.index_file, 'r') as f:
                    loaded_index = json.load(f)
                    if isinstance(loaded_index, dict):
                        logger.info(f"Loaded {len(loaded_index)} index entries from {self.index_file}")
                        return loaded_index
                    else:
                        logger.warning(f"Cache index file '{self.index_file}' contains invalid data type. Ignoring.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load cache index '{self.index_file}': {e}. Starting with empty index.")
            except Exception as e:
                logger.error(f"Unexpected error loading cache index '{self.index_file}': {e}")
                
        return {}  # Return empty dict if file doesn't exist or loading failed

    def _save_index(self) -> None:
        """
        Save the cache index to disk atomically.
        """
        try:
            # Write to a temporary file first
            temp_index_file = self.index_file + ".tmp"
            with self.lock, open(temp_index_file, 'w') as f:
                json.dump(self.index, f)  # No indentation for smaller file size
                
            # Atomically replace the old index file
            os.replace(temp_index_file, self.index_file)
            logger.debug(f"Saved cache index with {len(self.index)} entries.")
        except (IOError, OSError, TypeError) as e:
            logger.error(f"Failed to save cache index to '{self.index_file}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving cache index '{self.index_file}': {e}")

    def _get_cache_file_path(self, key: str) -> str:
        """
        Get the file path for a cache key
        Args:
            key: The cache key
        Returns:
            str: File path for the cache key
        """
        # Use SHA256 for potentially better distribution and collision resistance than MD5
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        
        # Use subdirectories to avoid too many files in one directory (e.g., first 2 chars)
        subdir = hashed_key[:2]
        filename = hashed_key[2:] + ".json"
        full_dir = os.path.join(self.cache_dir, subdir)
        
        # Ensure subdirectory exists
        try:
            # exist_ok=True avoids error if dir exists, but check lock needs
            # RLock or separate lock if this can race across processes/threads
            os.makedirs(full_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache subdirectory '{full_dir}': {e}")
            # Fallback to main cache dir? Or raise error? For now, log and continue.
            return os.path.join(self.cache_dir, hashed_key + ".json")  # Fallback path
            
        return os.path.join(full_dir, filename)

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get an entry from the cache
        Args:
            key: The cache key
        Returns:
            CacheEntry: The cache entry if found and not expired, None otherwise
        """
        with self.lock:
            # Check if key exists in index
            if key not in self.index:
                self.misses += 1
                return None
                
            # Get entry metadata from index
            entry_meta = self.index[key]
            
            # Check if entry has expired based on index data
            if entry_meta.get('expires_at') and time.time() > entry_meta['expires_at']:
                logger.debug(f"Disk cache entry '{key}' expired (per index). Removing.")
                # remove() handles index update and file deletion
                self.remove(key)
                self.misses += 1  # Treat expired as a miss
                return None
                
            # Get file path from index or generate it (prefer index if available)
            file_path = entry_meta.get('file_path') or self._get_cache_file_path(key)
            
            # Check if file exists
            if not os.path.exists(file_path):
                # File missing, remove from index
                logger.warning(f"Cache file '{file_path}' for key '{key}' not found. Removing from index.")
                del self.index[key]
                self._save_index()
                self.misses += 1
                return None
                
            # Load cache entry from file
            try:
                with open(file_path, 'r') as f:
                    entry_data = json.load(f)
                    
                # Create cache entry object
                entry = CacheEntry.from_dict(entry_data)
                
                # Optional: Verify expiration again from file data (in case index was stale)
                if entry.is_expired():
                    logger.debug(f"Disk cache entry '{key}' expired (per file). Removing.")
                    self.remove(key)
                    self.misses += 1
                    return None
                    
                self.hits += 1
                
                # Update last accessed time in index (optional, for LRU eviction)
                # entry_meta['last_accessed'] = time.time()
                # self._save_index()  # Careful: saving index on every get can be slow
                
                return entry
            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.error(f"Failed to load or parse cache entry from '{file_path}': {e}. Removing.")
                self.remove(key)  # Remove corrupted entry
                self.misses += 1
                return None
            except Exception as e:
                logger.error(f"Unexpected error reading cache file '{file_path}': {e}")
                self.misses += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an entry to the cache
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (optional)
            metadata: Additional information about the entry (optional)
        """
        with self.lock:
            # Calculate expiration time if TTL is provided
            now = time.time()
            expires_at = now + ttl if ttl is not None else None
            
            # Create a new cache entry object
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=now,
                expires_at=expires_at,
                metadata=metadata
            )
            
            # Get file path
            file_path = self._get_cache_file_path(key)
            
            # Save cache entry to file (write to temp then rename for atomicity)
            temp_file_path = file_path + ".tmp"
            entry_size = 0
            
            try:
                entry_json = json.dumps(entry.to_dict())  # Serialize once
                entry_bytes = entry_json.encode('utf-8')  # Encode for size calculation
                entry_size = len(entry_bytes)
                
                # Check if adding this entry exceeds size limit BEFORE writing
                self._enforce_size_limit(additional_size=entry_size)
                
                with open(temp_file_path, 'wb') as f:  # Write bytes directly
                    f.write(entry_bytes)
                    
                os.replace(temp_file_path, file_path)  # Atomic rename
                
                # Update index
                self.index[key] = {
                    'file_path': file_path,
                    'timestamp': now,
                    'expires_at': expires_at,
                    'size': entry_size
                }
                
                logger.debug(f"Set disk cache entry '{key}' at '{file_path}' (Size: {entry_size} bytes, TTL: {ttl}s)")
                
                # Save index
                self._save_index()
            except (IOError, OSError, TypeError, json.JSONDecodeError) as e:
                logger.error(f"Failed to save cache entry to '{file_path}': {e}")
                
                # Clean up temp file if it exists
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass  # Ignore cleanup error
            except Exception as e:
                logger.error(f"Unexpected error saving cache entry '{key}': {e}")

    def remove(self, key: str) -> bool:
        """
        Remove an entry from the cache
        Args:
            key: The cache key
        Returns:
            bool: True if the entry was removed, False if it wasn't in the cache
        """
        with self.lock:
            if key not in self.index:
                return False
                
            # Get file path from index
            entry_meta = self.index[key]
            file_path = entry_meta.get('file_path') or self._get_cache_file_path(key)  # Generate if missing
            
            # Remove file if it exists
            removed_file = False
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    removed_file = True
                except OSError as e:
                    logger.error(f"Failed to remove cache file '{file_path}': {e}")
                    # Continue to remove from index even if file deletion fails
                    
            # Remove from index
            del self.index[key]
            self._save_index()
            
            logger.debug(f"Removed disk cache entry '{key}' (File {'deleted' if removed_file else 'not found/error'}).")
            return True  # Return True even if file delete failed, as index entry is gone

    def clear(self) -> None:
        """
        Clear all entries from the cache.
        This involves removing all generated cache files and the index file.
        """
        with self.lock:
            logger.info(f"Clearing disk cache in directory '{self.cache_dir}'...")
            
            # Remove all indexed cache files first
            num_removed = 0
            for key in list(self.index.keys()):  # Iterate over copy of keys
                if self.remove(key):  # Use remove method to handle file path logic
                    num_removed += 1
                    
            # Clear the in-memory index
            self.index = {}
            
            # Try removing the index file itself
            try:
                if os.path.exists(self.index_file):
                    os.remove(self.index_file)
                if os.path.exists(self.index_file + ".tmp"):  # Clean up temp file too
                    os.remove(self.index_file + ".tmp")
            except OSError as e:
                logger.error(f"Failed to remove cache index file '{self.index_file}': {e}")
                
            # Optional: Remove any potentially orphaned files/subdirs? This can be risky.
            # A safer approach is just removing indexed files.
            
            logger.info(f"Disk cache clear removed {num_removed} indexed files and reset index.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            # Perform cleanup before calculating size for accuracy
            self.cleanup_expired()
            
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            # Calculate total size from index
            total_size = sum(entry.get('size', 0) for entry in self.index.values())
            total_size_mb = total_size / (1024 * 1024)
            
            return {
                'entries': len(self.index),
                'size_bytes': total_size,
                'size_mb': total_size_mb,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }

    def _enforce_size_limit(self, additional_size: int = 0) -> None:
        """
        Enforce the maximum cache size by removing old entries.
        Args:
            additional_size: Estimated size of a new entry being added.
        """
        # Calculate current size and potential new size
        current_size = sum(entry.get('size', 0) for entry in self.index.values())
        projected_size = current_size + additional_size
        
        # If we're under the limit, no need to evict
        if projected_size <= self.max_size_bytes:
            return
            
        # Calculate how much space we need to free
        space_to_free = projected_size - self.max_size_bytes
        freed_space = 0
        
        # Get entries sorted by timestamp (oldest first)
        # Make sure timestamp exists, default to 0 if not
        try:
            sorted_entries = sorted(
                self.index.items(),
                key=lambda item: item[1].get('timestamp', 0)
            )
        except Exception as e:
            logger.error(f"Error sorting cache index for eviction: {e}")
            return  # Cannot proceed with eviction
            
        logger.info(f"Disk cache size limit ({self.max_size_bytes / (1024*1024):.2f} MB) exceeded. "
                   f"Current: {current_size / (1024*1024):.2f} MB. Need to free {space_to_free / (1024*1024):.2f} MB.")
                   
        # Remove entries until enough space is freed
        removed_count = 0
        for key, entry_meta in sorted_entries:
            entry_size = entry_meta.get('size', 0)
            if self.remove(key):  # remove() handles index update and file deletion
                freed_space += entry_size
                removed_count += 1
                
            # Stop if enough space is freed
            if freed_space >= space_to_free:
                break
                
        logger.info(f"Evicted {removed_count} oldest entries, freeing {freed_space / (1024*1024):.2f} MB.")
        
        # Save index explicitly after potentially many removals
        self._save_index()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the disk cache.
        Returns:
            int: Number of entries removed.
        """
        removed_count = 0
        needs_save = False
        
        with self.lock:
            # Iterate over a copy of the keys to allow removal during iteration
            keys_to_check = list(self.index.keys())
            now = time.time()
            
            for key in keys_to_check:
                entry_meta = self.index.get(key)
                
                # Check expiration based on index data
                if entry_meta and entry_meta.get('expires_at') and now > entry_meta['expires_at']:
                    if self.remove(key):  # remove() handles file deletion and index update (but doesn't save index itself)
                        removed_count += 1
                        needs_save = True  # Mark that index needs saving
                        
            if needs_save:
                self._save_index()  # Save index after cleanup loop
                
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired entries from disk cache.")
                
            return removed_count

class DatabaseCache:
    """
    SQLite-based database cache implementation
    """
    def __init__(self, db_path: str = 'cache.db', max_entries: int = 1000):
        """
        Initialize a database cache
        Args:
            db_path: Path to the SQLite database file
            max_entries: Maximum number of entries to store
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
            
        self.db_path = db_path
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        
        # Use a separate lock for DB operations, distinct from higher-level caches
        self.db_lock = threading.RLock()
        
        # Ensure parent directory exists
        try:
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory for DB '{self.db_path}': {e}")
            raise
            
        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database."""
        # Consider adding timeout and other connection options
        try:
            # WAL mode can improve concurrency but adds extra files
            # conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
            # conn.execute("PRAGMA journal_mode=WAL;")
            conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)  # check_same_thread=False needed if used across threads without external lock
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database '{self.db_path}': {e}")
            raise  # Propagate the error

    def _init_db(self) -> None:
        """
        Initialize the database schema
        """
        # Use db_lock for schema initialization safety
        with self.db_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Create cache table if it doesn't exist
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    expires_at REAL,
                    metadata TEXT
                )
                ''')
                
                # Create index on timestamp for faster eviction
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON cache (timestamp)
                ''')
                
                # Create index on expiration for faster cleanup
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON cache (expires_at)
                ''')
                
                conn.commit()
                conn.close()
                
                logger.info(f"Database cache initialized successfully at '{self.db_path}'.")
            except sqlite3.Error as e:
                logger.error(f"Database cache initialization failed: {e}")
                # Depending on severity, might want to raise an exception

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get an entry from the cache
        Args:
            key: The cache key
        Returns:
            CacheEntry: The cache entry if found and not expired, None otherwise
        """
        # Use db_lock to serialize DB access if check_same_thread=True or for extra safety
        with self.db_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Query for the entry
                cursor.execute(
                    "SELECT key, value, timestamp, expires_at, metadata FROM cache WHERE key = ?",
                    (key,)
                )
                
                result = cursor.fetchone()
                conn.close()
                
                if result is None:
                    self.misses += 1
                    return None
                    
                # Parse result
                db_key, value_str, timestamp, expires_at, metadata_str = result
                
                # Check if entry has expired *before* parsing potentially large JSON
                if expires_at and time.time() > expires_at:
                    logger.debug(f"DB cache entry '{key}' expired. Removing.")
                    # remove() is thread-safe due to db_lock
                    self.remove(key)
                    self.misses += 1  # Treat expired as a miss
                    return None
                    
                # Parse JSON data
                try:
                    value = json.loads(value_str)
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse cached JSON data for key '{key}': {e}. Removing entry.")
                    self.remove(key)
                    self.misses += 1
                    return None
                    
                # Create cache entry object
                entry = CacheEntry(
                    key=db_key,  # Use key from DB for consistency
                    value=value,
                    timestamp=timestamp,
                    expires_at=expires_at,
                    metadata=metadata
                )
                
                self.hits += 1
                return entry
            except sqlite3.Error as e:
                logger.error(f"Database error during get operation for key '{key}': {e}")
                self.misses += 1
                return None
            except Exception as e:
                logger.error(f"Unexpected error during DB get operation for key '{key}': {e}")
                self.misses += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an entry to the cache
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (optional)
            metadata: Additional information about the entry (optional)
        """
        with self.db_lock:
            # Check if we need to make space *before* the transaction
            self._enforce_size_limit()
            
            # Calculate expiration time if TTL is provided
            now = time.time()
            expires_at = now + ttl if ttl is not None else None
            
            try:
                # Serialize value and metadata to JSON
                value_str = json.dumps(value)
                metadata_str = json.dumps(metadata) if metadata else None
            except (TypeError, OverflowError) as e:
                logger.error(f"Failed to serialize data for cache key '{key}': {e}")
                return  # Cannot cache unserializable data
                
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Insert or replace the entry
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO cache (key, value, timestamp, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (key, value_str, now, expires_at, metadata_str)
                )
                
                conn.commit()
                conn.close()
                
                logger.debug(f"Set DB cache entry '{key}' (TTL: {ttl}s)")
            except sqlite3.Error as e:
                logger.error(f"Database error during set operation for key '{key}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error during DB set operation for key '{key}': {e}")

    def remove(self, key: str) -> bool:
        """
        Remove an entry from the cache
        Args:
            key: The cache key
        Returns:
            bool: True if the entry was removed, False if it wasn't in the cache
        """
        with self.db_lock:
            removed = False
            
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Delete the entry
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                
                # Check if any rows were affected
                removed = cursor.rowcount > 0
                
                conn.commit()
                conn.close()
                
                if removed:
                    logger.debug(f"Removed DB cache entry '{key}'.")
            except sqlite3.Error as e:
                logger.error(f"Database error during remove operation for key '{key}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error during DB remove operation for key '{key}': {e}")
                
            return removed

    def clear(self) -> None:
        """
        Clear all entries from the cache
        """
        with self.db_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Delete all entries
                cursor.execute("DELETE FROM cache")
                
                # Optionally reset auto-increment counters etc. if needed
                # cursor.execute("VACUUM")  # To reclaim space, potentially slow
                
                conn.commit()
                conn.close()
                
                logger.info("Cleared database cache.")
            except sqlite3.Error as e:
                logger.error(f"Database error during clear operation: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during DB clear operation: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        Returns:
            dict: Cache statistics
        """
        # Perform cleanup before getting stats for accuracy
        self.cleanup_expired()
        
        with self.db_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get entry count
                cursor.execute("SELECT COUNT(*) FROM cache")
                entry_count = cursor.fetchone()[0]
                
                # Get database size (approximate)
                db_size = 0
                db_size_mb = 0.0
                
                if os.path.exists(self.db_path):
                    try:
                        db_size = os.path.getsize(self.db_path)
                        # Consider size of WAL file if using WAL mode
                        wal_path = self.db_path + "-wal"
                        if os.path.exists(wal_path):
                            db_size += os.path.getsize(wal_path)
                        db_size_mb = db_size / (1024 * 1024)
                    except OSError:
                        logger.warning(f"Could not get size of database file '{self.db_path}'.")
                        
                conn.close()
                
                total_requests = self.hits + self.misses
                hit_rate = self.hits / total_requests if total_requests > 0 else 0
                
                return {
                    'entries': entry_count,
                    'max_entries': self.max_entries,
                    'size_bytes': db_size,
                    'size_mb': db_size_mb,
                    'hits': self.hits,
                    'misses': self.misses,
                    'hit_rate': hit_rate,
                    'total_requests': total_requests
                }
            except sqlite3.Error as e:
                logger.error(f"Database error during get_stats operation: {e}")
                return { 'error': str(e) }  # Return error indication
            except Exception as e:
                logger.error(f"Unexpected error during DB get_stats operation: {e}")
                return { 'error': str(e) }

    def _enforce_size_limit(self) -> None:
        """
        Enforce the maximum number of cache entries by removing the oldest ones.
        This is called internally before adding new entries.
        """
        # No lock here as it's called by set() which already holds the lock
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get entry count efficiently
            cursor.execute("SELECT COUNT(*) FROM cache")
            entry_count = cursor.fetchone()[0]
            
            if entry_count <= self.max_entries:
                conn.close()
                return
                
            # Calculate how many entries to remove
            to_remove = entry_count - self.max_entries
            
            logger.info(f"DB cache entry limit ({self.max_entries}) exceeded. "
                       f"Current: {entry_count}. Need to remove {to_remove} entries.")
                       
            # Remove oldest entries based on timestamp
            # Using a subquery like this is standard and efficient with the index
            cursor.execute(
                "DELETE FROM cache WHERE key IN (SELECT key FROM cache ORDER BY timestamp ASC LIMIT ?)",
                (to_remove,)
            )
            
            removed_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Evicted {removed_count} oldest entries from DB cache.")
        except sqlite3.Error as e:
            logger.error(f"Database error during enforce_size_limit: {e}")
            # Avoid closing connection if it wasn't opened or failed
            if 'conn' in locals() and conn:
                try:
                    conn.close()
                except sqlite3.Error: pass  # Ignore close error if already errored
        except Exception as e:
            logger.error(f"Unexpected error during DB enforce_size_limit: {e}")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        Returns:
            int: Number of entries removed.
        """
        removed_count = 0
        
        with self.db_lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                now = time.time()
                
                # Delete expired entries using the index on expires_at
                cursor.execute(
                    "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,)
                )
                
                # Get number of affected rows
                removed_count = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} expired entries from DB cache.")
            except sqlite3.Error as e:
                logger.error(f"Database error during cleanup_expired: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during DB cleanup_expired: {e}")
                
            return removed_count

class MultiLevelCache:
    """
    Multi-level cache implementation that combines memory, disk, and database caches
    """
    def __init__(
        self,
        memory_size: int = 100,
        disk_cache_dir: str = '.cache',
        disk_max_size_mb: int = 100,
        db_path: str = 'cache.db',
        db_max_entries: int = 1000,
        default_ttl: Optional[int] = None,
        # Allow disabling levels
        enable_memory: bool = True,
        enable_disk: bool = True,
        enable_db: bool = True
    ):
        """
        Initialize a multi-level cache
        Args:
            memory_size: Maximum number of entries in the memory cache
            disk_cache_dir: Directory for the disk cache
            disk_max_size_mb: Maximum disk cache size in megabytes
            db_path: Path to the database cache file
            db_max_entries: Maximum number of entries in the database cache
            default_ttl: Default time-to-live for cache entries in seconds
            enable_memory: Enable memory cache level
            enable_disk: Enable disk cache level
            enable_db: Enable database cache level
        """
        self.memory_cache = MemoryCache(max_size=memory_size) if enable_memory else None
        self.disk_cache = DiskCache(cache_dir=disk_cache_dir, max_size_mb=disk_max_size_mb) if enable_disk else None
        self.db_cache = DatabaseCache(db_path=db_path, max_entries=db_max_entries) if enable_db else None
        
        self.default_ttl = default_ttl
        self.lock = threading.RLock()  # Lock for managing multi-level interactions
        
        if not (self.memory_cache or self.disk_cache or self.db_cache):
            logger.warning("All cache levels are disabled. MultiLevelCache will not store any data.")

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache, checking levels sequentially.
        Promotes found items to higher levels.
        Args:
            key: The cache key
        Returns:
            Any: The cached value if found and not expired, None otherwise
        """
        # Check memory first
        if self.memory_cache:
            entry = self.memory_cache.get(key)
            if entry:
                logger.debug(f"Cache hit for '{key}' in memory.")
                return entry.value
                
        # Check disk next
        if self.disk_cache:
            entry = self.disk_cache.get(key)
            if entry:
                logger.debug(f"Cache hit for '{key}' in disk. Promoting to memory.")
                
                # Promote to memory cache if enabled
                if self.memory_cache:
                    ttl_remaining = entry.expires_at - time.time() if entry.expires_at else None
                    
                    # Ensure ttl is non-negative
                    ttl_to_set = max(0, ttl_remaining) if ttl_remaining is not None else None
                    
                    self.memory_cache.set(
                        key=key,
                        value=entry.value,
                        ttl=ttl_to_set,
                        metadata=entry.metadata
                    )
                    
                return entry.value
                
        # Check database last
        if self.db_cache:
            entry = self.db_cache.get(key)
            if entry:
                logger.debug(f"Cache hit for '{key}' in database. Promoting to memory and disk.")
                
                ttl_remaining = entry.expires_at - time.time() if entry.expires_at else None
                
                # Ensure ttl is non-negative
                ttl_to_set = max(0, ttl_remaining) if ttl_remaining is not None else None
                
                # Promote to disk cache if enabled
                if self.disk_cache:
                    self.disk_cache.set(
                        key=key,
                        value=entry.value,
                        ttl=ttl_to_set,
                        metadata=entry.metadata
                    )
                    
                # Promote to memory cache if enabled (needs to happen after disk potentially)
                if self.memory_cache:
                    self.memory_cache.set(
                        key=key,
                        value=entry.value,
                        ttl=ttl_to_set,
                        metadata=entry.metadata
                    )
                    
                return entry.value
                
        # If not found in any level
        logger.debug(f"Cache miss for '{key}'.")
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        levels: Optional[List[str]] = None
    ) -> None:
        """
        Add an entry to the specified cache levels (or all enabled levels if None).
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (optional, uses default_ttl if not provided)
            metadata: Additional information about the entry (optional)
            levels: Cache levels to use ('memory', 'disk', 'db', or None for all enabled)
        """
        with self.lock:
            # Use default TTL if not provided
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            # Determine which cache levels to use based on 'levels' arg and enabled caches
            use_memory = (levels is None or 'memory' in levels) and self.memory_cache is not None
            use_disk = (levels is None or 'disk' in levels) and self.disk_cache is not None
            use_db = (levels is None or 'db' in levels) and self.db_cache is not None
            
            # Set in each selected cache level
            if use_memory:
                self.memory_cache.set(
                    key=key,
                    value=value,
                    ttl=effective_ttl,
                    metadata=metadata
                )
                
            if use_disk:
                self.disk_cache.set(
                    key=key,
                    value=value,
                    ttl=effective_ttl,
                    metadata=metadata
                )
                
            if use_db:
                self.db_cache.set(
                    key=key,
                    value=value,
                    ttl=effective_ttl,
                    metadata=metadata
                )

    def remove(self, key: str) -> bool:
        """
        Remove an entry from all enabled cache levels.
        Args:
            key: The cache key
        Returns:
            bool: True if the entry was removed from at least one level, False otherwise.
        """
        with self.lock:
            removed_any = False
            
            if self.memory_cache:
                if self.memory_cache.remove(key):
                    removed_any = True
                    
            if self.disk_cache:
                if self.disk_cache.remove(key):
                    removed_any = True
                    
            if self.db_cache:
                if self.db_cache.remove(key):
                    removed_any = True
                    
            return removed_any

    def clear(self) -> None:
        """
        Clear all entries from all enabled cache levels.
        """
        with self.lock:
            if self.memory_cache:
                self.memory_cache.clear()
                
            if self.disk_cache:
                self.disk_cache.clear()
                
            if self.db_cache:
                self.db_cache.clear()
                
            logger.info("Cleared all enabled cache levels.")

    def get_stats(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get statistics for all enabled cache levels.
        Returns:
            dict: Cache statistics for each enabled level ('memory', 'disk', 'db').
            Value is None if the level is disabled.
        """
        with self.lock:
            stats = {}
            
            stats['memory'] = self.memory_cache.get_stats() if self.memory_cache else None
            stats['disk'] = self.disk_cache.get_stats() if self.disk_cache else None
            stats['db'] = self.db_cache.get_stats() if self.db_cache else None
            
            return stats

    def cleanup_expired(self) -> Dict[str, Optional[int]]:
        """
        Remove all expired entries from all enabled cache levels.
        Returns:
            dict: Number of entries removed from each level ('memory', 'disk', 'db').
            Value is None if the level is disabled.
        """
        with self.lock:
            removed_counts = {}
            
            # Memory cache cleanup
            removed_counts['memory'] = self.memory_cache.cleanup_expired() if self.memory_cache else None
            
            # Disk cache cleanup
            removed_counts['disk'] = self.disk_cache.cleanup_expired() if self.disk_cache else None
            
            # Database cache cleanup
            removed_counts['db'] = self.db_cache.cleanup_expired() if self.db_cache else None
            
            logger.info(f"Expired cache cleanup results: {removed_counts}")
            return removed_counts

class DomainPartitionedCache(MultiLevelCache):
    """
    Enhanced cache with domain-specific partitioning capabilities
    """
    def __init__(
        self, 
        num_domains: int = 5,
        domain_discovery_method: str = 'kmeans',
        **kwargs
    ):
        """
        Initialize a domain-partitioned cache
        
        Args:
            num_domains: Number of domains to partition content into
            domain_discovery_method: Method to use for domain discovery ('kmeans', 'lda', 'manual')
            **kwargs: Additional arguments to pass to MultiLevelCache constructor
        """
        super().__init__(**kwargs)
        self.num_domains = num_domains
        self.domain_discovery_method = domain_discovery_method
        
        # Domain-specific components
        self.domain_vectorizer = TfidfVectorizer(max_features=100)
        self.domain_classifier = None
        self.domain_embeddings = {}  # Domain-specific embedding spaces
        self.document_domains = {}   # Mapping of document ID to domain
        self.domain_cache = {}       # Separate cache for each domain
        self.domain_stats = {}       # Statistics for each domain
        
        # Initialize domain caches
        for i in range(num_domains):
            self.domain_cache[i] = {}
            self.domain_stats[i] = {
                'hits': 0,
                'misses': 0,
                'entries': 0,
                'last_hit': None
            }
        
        logger.info(f"Initialized domain-partitioned cache with {num_domains} domains")
    
    def initialize_domains(self, documents: List[Dict[str, Any]]) -> None:
        """
        Cluster documents into domains and initialize domain-specific caches
        
        Args:
            documents: List of documents to cluster
        """
        if not documents:
            logger.warning("No documents provided for domain initialization")
            return
            
        # Extract text from documents
        texts = [doc.get('text', '') for doc in documents]
        if not texts or all(not text for text in texts):
            logger.warning("No text content found in documents for domain initialization")
            return
            
        # Create document vectors
        try:
            X = self.domain_vectorizer.fit_transform(texts)
            
            if self.domain_discovery_method == 'kmeans':
                # Cluster documents into domains using K-means
                self.domain_classifier = KMeans(
                    n_clusters=min(self.num_domains, len(texts)), 
                    random_state=42
                )
                domains = self.domain_classifier.fit_predict(X)
            
                # Map documents to domains
                for i, doc in enumerate(documents):
                    doc_id = doc.get('id', str(hash(doc.get('text', ''))))
                    self.document_domains[doc_id] = int(domains[i])
                    
                # Initialize domain-specific embedding spaces
                for domain_id in range(self.num_domains):
                    # Get documents in this domain
                    domain_docs = [doc for i, doc in enumerate(documents) 
                                 if i < len(domains) and domains[i] == domain_id]
                    if domain_docs:
                        self.domain_embeddings[domain_id] = self._create_domain_embedding_space(domain_docs)
                
                logger.info(f"Initialized {self.num_domains} domain-specific caches with {len(documents)} documents")
                
                # Log domain distribution
                domain_counts = {}
                for domain in domains:
                    domain_counts[int(domain)] = domain_counts.get(int(domain), 0) + 1
                logger.info(f"Domain distribution: {domain_counts}")
                
            else:
                logger.warning(f"Unsupported domain discovery method: {self.domain_discovery_method}")
                
        except Exception as e:
            logger.error(f"Error initializing domains: {e}")
    
    def _create_domain_embedding_space(self, documents: List[Dict[str, Any]]) -> TfidfVectorizer:
        """
        Create a specialized embedding space for a domain
        
        Args:
            documents: List of documents in the domain
            
        Returns:
            TfidfVectorizer: A vectorizer trained on domain-specific documents
        """
        # Extract text from documents
        domain_texts = [doc.get('text', '') for doc in documents]
        
        # Create a domain-specific vectorizer
        domain_vectorizer = TfidfVectorizer(max_features=50)
        try:
            domain_vectorizer.fit(domain_texts)
            return domain_vectorizer
        except:
            return self.domain_vectorizer  # Fallback to global vectorizer
    
    def predict_domain(self, query: str) -> int:
        """
        Predict which domain a query belongs to
        
        Args:
            query: The search query
            
        Returns:
            int: Domain ID
        """
        if not self.domain_classifier:
            return 0  # Default domain if not initialized
        
        try:
            # Transform query using the same vectorizer used for clustering
            query_vector = self.domain_vectorizer.transform([query])
            
            # Predict domain
            domain = int(self.domain_classifier.predict(query_vector)[0])
            return domain
        except Exception as e:
            logger.error(f"Error predicting domain: {e}")
            return 0  # Default to first domain on error
    
    def get(self, key: str, query: str = None) -> Optional[Any]:
        """
        Get a value from the cache, potentially using domain-specific cache
        
        Args:
            key: The cache key
            query: Optional query text to determine domain
            
        Returns:
            Any: The cached value if found, otherwise None
        """
        # If query provided, try domain-specific cache first
        if query and self.domain_classifier:
            domain = self.predict_domain(query)
            
            if domain in self.domain_cache and key in self.domain_cache[domain]:
                # Update domain statistics
                self.domain_stats[domain]['hits'] += 1
                self.domain_stats[domain]['last_hit'] = time.time()
                
                # Return from domain cache
                return self.domain_cache[domain][key]
        
        # Fall back to standard cache
        return super().get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
           metadata: Optional[Dict[str, Any]] = None, query: str = None) -> None:
        """
        Set a value in the cache, potentially in domain-specific cache
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live
            metadata: Additional metadata
            query: Optional query text to determine domain
        """
        # Store in standard cache
        super().set(key, value, ttl, metadata)
        
        # If query provided, also store in domain-specific cache
        if query and self.domain_classifier:
            domain = self.predict_domain(query)
            self.domain_cache[domain][key] = value
            
            # Update domain statistics
            self.domain_stats[domain]['entries'] += 1

    def get_domain_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics about domain cache usage
        
        Returns:
            dict: Statistics for each domain
        """
        stats = {}
        for domain_id, domain_stat in self.domain_stats.items():
            stats[domain_id] = {
                'hits': domain_stat['hits'],
                'entries': domain_stat['entries'],
                'last_hit': domain_stat['last_hit'],
                'documents': sum(1 for domain in self.document_domains.values() 
                               if domain == domain_id)
            }
        return stats

class RLRelevanceRanker:
    """
    Reinforcement Learning based ranker for search results
    """
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                exploration_rate: float = 0.2, feature_dim: int = 10):
        """
        Initialize the RL-based ranker
        
        Args:
            learning_rate: Learning rate for weight updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Probability of exploring vs exploiting
            feature_dim: Dimension of feature vectors
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.feature_dim = feature_dim
        
        # Initialize model weights
        self.weights = np.random.randn(feature_dim) / np.sqrt(feature_dim)
        
        # State tracking for reinforcement learning
        self.state_vectors = {}  # Maps document IDs to feature vectors
        self.interaction_history = []  # Track user interactions for learning
        
        # Feature extraction components
        self.vectorizer = TfidfVectorizer(max_features=self.feature_dim)
        self.is_vectorizer_fitted = False
        
        # Stats
        self.total_rankings = 0
        self.total_updates = 0
        
        logger.info(f"Initialized RL-based ranker with {feature_dim} features")
    
    def extract_features(self, query: str, document: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from query and document for ranking
        
        Args:
            query: Search query
            document: Document to rank
            
        Returns:
            np.ndarray: Feature vector
        """
        # Check if document is valid
        if not document or not query:
            return np.zeros(self.feature_dim)
            
        # Extract text content
        doc_text = document.get('text', '')
        if not doc_text:
            return np.zeros(self.feature_dim)
            
        # Combine query and document for joint feature extraction
        combined_text = f"{query} {doc_text[:1000]}"  # Limit document length
        
        try:
            # Fit vectorizer if needed
            if not self.is_vectorizer_fitted:
                self.vectorizer.fit([combined_text])
                self.is_vectorizer_fitted = True
                
            # Transform to get TF-IDF features
            tfidf_features = self.vectorizer.transform([combined_text]).toarray()[0]
            
            # Add metadata features
            text_length = min(len(doc_text) / 5000.0, 1.0)  # Normalized by 5000 chars
            source_count = min(len(document.get('sources', [])) / 10.0, 1.0)  # Normalized by 10 sources
            recency = document.get('recency', 0.5)  # Default to middle value if not available
            
            # Create additional features array
            additional_features = np.array([text_length, source_count, recency])
            
            # Combine TF-IDF and additional features
            all_features = np.concatenate([
                tfidf_features, 
                additional_features
            ])
            
            # Ensure vector is correct length, padding or truncating if needed
            if len(all_features) < self.feature_dim:
                all_features = np.pad(all_features, (0, self.feature_dim - len(all_features)))
            elif len(all_features) > self.feature_dim:
                all_features = all_features[:self.feature_dim]
                
            # Normalize the feature vector
            return all_features / (np.linalg.norm(all_features) + 1e-8)
            
        except Exception as e:
            logger.error(f"Error extracting ranking features: {e}")
            return np.zeros(self.feature_dim)
    
    def score_document(self, query: str, document: Dict[str, Any]) -> float:
        """
        Score a document using the current policy (weights)
        
        Args:
            query: The search query
            document: Document to score
            
        Returns:
            float: Document score
        """
        # Extract feature vector
        features = self.extract_features(query, document)
        
        # Store state vector for later updates
        doc_id = document.get('id', str(hash(document.get('text', ''))))
        self.state_vectors[doc_id] = features
        
        # With probability exploration_rate, return a random score for exploration
        if np.random.random() < self.exploration_rate:
            return np.random.random()
            
        # Otherwise, compute score using dot product (policy)
        return np.dot(features, self.weights)
    
    def rank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank documents using learned policy
        
        Args:
            query: The search query
            documents: List of documents to rank
            
        Returns:
            List[Dict[str, Any]]: Ranked documents with scores
        """
        if not documents:
            return []
            
        self.total_rankings += 1
        
        # Score all documents
        scored_docs = [(doc, self.score_document(query, doc)) for doc in documents]
        
        # Sort by score in descending order
        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Return documents with scores
        return [{'document': doc, 'score': score} for doc, score in ranked_docs]
    
    def record_interaction(self, query: str, document_id: str, interaction_type: str, value: float = 1.0) -> None:
        """
        Record user interaction for reinforcement learning updates
        
        Args:
            query: The search query
            document_id: ID of the document interacted with
            interaction_type: Type of interaction (click, dwell, etc.)
            value: Value of the interaction (1.0 for positive, -1.0 for negative)
        """
        # Record the interaction
        self.interaction_history.append({
            'query': query,
            'document_id': document_id,
            'interaction': interaction_type,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update model weights based on interaction
        if document_id in self.state_vectors:
            features = self.state_vectors[document_id]
            
            # Simple update rule: w += learning_rate * reward * features
            update = self.learning_rate * value * features
            self.weights += update
            
            # Normalize weights
            self.weights = self.weights / (np.linalg.norm(self.weights) + 1e-8)
            
            self.total_updates += 1
            
            if self.total_updates % 10 == 0:
                logger.info(f"RL ranker updated weights after {self.total_updates} interactions")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ranker
        
        Returns:
            dict: Statistics about the ranker
        """
        return {
            'total_rankings': self.total_rankings,
            'total_updates': self.total_updates,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'feature_dim': self.feature_dim,
            'interactions_recorded': len(self.interaction_history)
        }

class SelfSupervisedRelevanceTrainer:
    """
    Self-supervised learning system for improving search relevance based on user interactions
    """
    def __init__(self, vectorizer_dim: int = 100, update_interval: int = 10):
        """
        Initialize the self-supervised relevance trainer
        
        Args:
            vectorizer_dim: Dimension of the vectorizer
            update_interval: Number of interactions before updating the model
        """
        self.vectorizer = TfidfVectorizer(max_features=vectorizer_dim)
        self.vectorizer_fitted = False
        self.update_interval = update_interval
        
        # Positive and negative examples
        self.positive_examples = []  # (query, document) pairs with positive feedback
        self.negative_examples = []  # (query, document) pairs with negative feedback
        
        # Tracking metrics
        self.interaction_count = 0
        self.model_updates = 0
        
        # Relevance model (currently using a simple similarity approach)
        self.query_embeddings = {}  # Cache of query embeddings
        self.doc_embeddings = {}    # Cache of document embeddings
        
        logger.info(f"Initialized self-supervised relevance trainer with dimension {vectorizer_dim}")
    
    def record_interaction(self, query: str, document: Dict[str, Any], is_relevant: bool) -> None:
        """
        Record a relevance interaction
        
        Args:
            query: The search query
            document: The document that received feedback
            is_relevant: Whether the document was relevant to the query
        """
        if not query or not document:
            return
            
        # Extract document text
        doc_text = document.get('text', '')
        if not doc_text:
            return
            
        # Record the example
        example = (query, doc_text)
        
        if is_relevant:
            self.positive_examples.append(example)
        else:
            self.negative_examples.append(example)
            
        self.interaction_count += 1
        
        # Check if we should update the model
        if self.interaction_count % self.update_interval == 0:
            self._update_model()
    
    def _update_model(self) -> None:
        """Update the relevance model based on recorded examples"""
        # Ensure we have enough examples
        if len(self.positive_examples) < 3 or len(self.negative_examples) < 3:
            return
            
        try:
            # Prepare training data
            all_texts = []
            
            # Add positive examples
            for query, doc in self.positive_examples:
                all_texts.append(query)
                all_texts.append(doc)
                
            # Add negative examples
            for query, doc in self.negative_examples:
                all_texts.append(query)
                all_texts.append(doc)
                
            # Fit or update vectorizer
            if not self.vectorizer_fitted:
                self.vectorizer.fit(all_texts)
                self.vectorizer_fitted = True
                
            # Clear embedding caches after model update
            self.query_embeddings = {}
            self.doc_embeddings = {}
            
            self.model_updates += 1
            logger.info(f"Updated relevance model (update #{self.model_updates}) "
                       f"with {len(self.positive_examples)} positive and "
                       f"{len(self.negative_examples)} negative examples")
                       
            # Limit examples count to prevent unbounded growth
            if len(self.positive_examples) > 1000:
                self.positive_examples = self.positive_examples[-1000:]
            if len(self.negative_examples) > 1000:
                self.negative_examples = self.negative_examples[-1000:]
                
        except Exception as e:
            logger.error(f"Error updating relevance model: {e}")
    
    def get_document_relevance(self, query: str, document: Dict[str, Any]) -> float:
        """
        Calculate the relevance of a document to a query
        
        Args:
            query: The search query
            document: The document to calculate relevance for
            
        Returns:
            float: Relevance score between 0 and 1
        """
        if not query or not document or not self.vectorizer_fitted:
            return 0.5  # Default mid-point relevance
            
        doc_text = document.get('text', '')
        if not doc_text:
            return 0.5
            
        try:
            # Get query embedding, using cache if available
            if query in self.query_embeddings:
                query_embedding = self.query_embeddings[query]
            else:
                query_embedding = self.vectorizer.transform([query]).toarray()[0]
                self.query_embeddings[query] = query_embedding
                
            # Get document embedding, using cache if available
            doc_id = document.get('id', str(hash(doc_text)))
            if doc_id in self.doc_embeddings:
                doc_embedding = self.doc_embeddings[doc_id]
            else:
                doc_embedding = self.vectorizer.transform([doc_text]).toarray()[0]
                self.doc_embeddings[doc_id] = doc_embedding
                
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            
            # Scale to 0-1 range
            relevance = (similarity + 1) / 2
            
            return relevance
            
        except Exception as e:
            logger.error(f"Error calculating document relevance: {e}")
            return 0.5
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results based on learned relevance
        
        Args:
            query: The search query
            results: List of result documents
            
        Returns:
            List[Dict[str, Any]]: Reranked results
        """
        if not results or not self.vectorizer_fitted:
            return results
            
        try:
            # Calculate relevance for each result
            scored_results = [
                (doc, self.get_document_relevance(query, doc))
                for doc in results
            ]
            
            # Sort by relevance score in descending order
            reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
            
            # Return just the documents in new order
            return [doc for doc, _ in reranked_results]
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the trainer
        
        Returns:
            dict: Statistics
        """
        return {
            'positive_examples': len(self.positive_examples),
            'negative_examples': len(self.negative_examples),
            'total_interactions': self.interaction_count,
            'model_updates': self.model_updates,
            'model_fitted': self.vectorizer_fitted,
            'embeddings_cached': len(self.query_embeddings) + len(self.doc_embeddings)
        }

class SearchCache:
    """
    Cache specifically for search results, with query normalization and similarity matching
    """
    def __init__(
        self, 
        cache_impl: Optional[Any] = None,
        default_ttl: int = 86400,
        query_similarity_threshold: Optional[float] = None
    ):
        """
        Initialize a search cache
        
        Args:
            cache_impl: Underlying cache implementation
            default_ttl: Default TTL in seconds
            query_similarity_threshold: Threshold for query similarity
        """
        # Initialize cache implementation
        self.cache = cache_impl or MultiLevelCache(
            memory_size=100,
            default_ttl=default_ttl
        )
        
        self.default_ttl = default_ttl
        self.query_similarity_threshold = query_similarity_threshold or 0.8
        self.query_history = {}  # Maps normalized queries to original queries
        self.lock = threading.RLock()
        
        logger.info(f"Initialized SearchCache with default TTL of {default_ttl}s")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize a query for cache lookup
        
        Args:
            query: The raw query string
            
        Returns:
            str: Normalized query string
        """
        if not query:
            return ""
            
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
        
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity between two queries
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            float: Similarity score (0-1)
        """
        # Simple character-level Jaccard similarity
        if not query1 or not query2:
            return 0.0
            
        set1 = set(query1)
        set2 = set(query2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
        
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get search results from cache
        
        Args:
            query: The search query
            
        Returns:
            dict: Search results if found, None otherwise
        """
        with self.lock:
            # Normalize the query
            normalized_query = self._normalize_query(query)
            if not normalized_query:
                return None
                
            # Try exact match first
            result = self.cache.get(normalized_query)
            if result:
                logger.debug(f"Cache hit for exact query: '{normalized_query}'")
                return result
                
            # If similarity matching is enabled, try similar queries
            if self.query_similarity_threshold < 1.0:
                for cached_query in self.query_history:
                    similarity = self._calculate_query_similarity(normalized_query, cached_query)
                    if similarity >= self.query_similarity_threshold:
                        result = self.cache.get(cached_query)
                        if result:
                            logger.debug(f"Cache hit for similar query: '{normalized_query}' matched '{cached_query}' with similarity {similarity:.2f}")
                            return result
                
            return None
    
    def set(
        self, 
        query: str, 
        result: Dict[str, Any], 
        ttl: Optional[int] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Cache search results
        
        Args:
            query: The search query
            result: The search result to cache
            ttl: Time-to-live in seconds
            metadata: Additional metadata
        """
        with self.lock:
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            # Normalize the query
            normalized_query = self._normalize_query(query)
            if not normalized_query:
                logger.warning(f"Cannot cache empty normalized query: '{query}'")
                return
                
            # Keep track of the original query for this normalized form
            self.query_history[normalized_query] = query
            
            # Cache the result
            self.cache.set(normalized_query, result, effective_ttl, metadata)
            logger.debug(f"Cached result for query: '{normalized_query}' (TTL: {effective_ttl}s)")
    
    def remove(self, query: str) -> bool:
        """
        Remove a search result from the cache
        
        Args:
            query: The search query
            
        Returns:
            bool: True if removed, False otherwise
        """
        with self.lock:
            normalized_query = self._normalize_query(query)
            if not normalized_query:
                return False
                
            # Remove from history
            if normalized_query in self.query_history:
                del self.query_history[normalized_query]
                
            # Remove from cache
            return self.cache.remove(normalized_query)
    
    def clear(self) -> None:
        """
        Clear all cached search results
        """
        with self.lock:
            self.query_history.clear()
            self.cache.clear()
            logger.info("Search cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache
        
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            cache_stats = self.cache.get_stats() if hasattr(self.cache, 'get_stats') else {}
            stats = {
                'query_history_size': len(self.query_history),
                'cache_stats': cache_stats,
                'default_ttl': self.default_ttl,
                'query_similarity_threshold': self.query_similarity_threshold
            }
            return stats

# --- SearchCache Singleton Accessor ---
_search_cache_instance = None
_search_cache_lock = threading.Lock()

def get_search_cache(config: Optional[Dict[str, Any]] = None) -> SearchCache:
    """
    Get a singleton instance of the SearchCache
    
    Args:
        config: Configuration settings dictionary (used only on first initialization)
        
    Returns:
        SearchCache: The singleton instance
    """
    global _search_cache_instance
    
    if _search_cache_instance is None:
        with _search_cache_lock:
            if _search_cache_instance is None:
                logger.info("Creating SearchCache singleton instance")
                
                # Extract cache-specific config
                cache_config = config or {}
                if 'cache' in cache_config and isinstance(cache_config['cache'], dict):
                    cache_config = cache_config['cache']
                
                # Create the base cache implementation
                cache_impl = MultiLevelCache(
                    memory_size=int(cache_config.get('memory_size', 100)),
                    disk_cache_dir=cache_config.get('disk_cache_dir', '.cache/disk'),
                    disk_max_size_mb=int(cache_config.get('disk_max_size_mb', 100)),
                    db_path=cache_config.get('db_path', '.cache/db/cache.db'),
                    db_max_entries=int(cache_config.get('db_max_entries', 1000)),
                    default_ttl=int(cache_config.get('default_ttl', 86400)),
                    enable_memory=cache_config.get('enable_memory', True),
                    enable_disk=cache_config.get('enable_disk', True),
                    enable_db=cache_config.get('enable_db', True)
                )
                
                # Create the standard search cache
                _search_cache_instance = SearchCache(
                    cache_impl=cache_impl,
                    default_ttl=int(cache_config.get('default_ttl', 86400)),
                    query_similarity_threshold=float(cache_config.get('query_similarity_threshold', 0.8))
                )
                
    return _search_cache_instance

class EnhancedSearchCache(SearchCache):
    """
    Enhanced search cache with domain-specific partitioning and reinforcement learning ranking
    """
    def __init__(
        self,
        cache_impl: Optional[Any] = None,
        default_ttl: int = 86400,
        query_similarity_threshold: Optional[float] = None,
        enable_domain_partitioning: bool = True,
        enable_rl_ranking: bool = True,
        enable_self_supervised_learning: bool = True,
        num_domains: int = 5
    ):
        """
        Initialize an enhanced search cache
        
        Args:
            cache_impl: Underlying cache implementation
            default_ttl: Default TTL in seconds
            query_similarity_threshold: Threshold for query similarity
            enable_domain_partitioning: Whether to enable domain partitioning
            enable_rl_ranking: Whether to enable RL-based ranking
            enable_self_supervised_learning: Whether to enable self-supervised learning
            num_domains: Number of domains to partition into
        """
        # Initialize the base SearchCache
        super().__init__(
            cache_impl=cache_impl,
            default_ttl=default_ttl,
            query_similarity_threshold=query_similarity_threshold
        )
        
        # Domain partitioning
        self.enable_domain_partitioning = enable_domain_partitioning
        if enable_domain_partitioning:
            self.domain_cache = DomainPartitionedCache(
                num_domains=num_domains,
                memory_size=100,
                default_ttl=default_ttl
            )
            logger.info(f"Enabled domain partitioning with {num_domains} domains")
        else:
            self.domain_cache = None
            
        # RL-based ranking
        self.enable_rl_ranking = enable_rl_ranking
        if enable_rl_ranking:
            self.rl_ranker = RLRelevanceRanker(
                learning_rate=0.1,
                discount_factor=0.9,
                exploration_rate=0.1
            )
            logger.info("Enabled reinforcement learning based ranking")
        else:
            self.rl_ranker = None
            
        # Self-supervised learning
        self.enable_self_supervised_learning = enable_self_supervised_learning
        if enable_self_supervised_learning:
            self.relevance_trainer = SelfSupervisedRelevanceTrainer(
                vectorizer_dim=100,
                update_interval=10
            )
            logger.info("Enabled self-supervised relevance learning")
        else:
            self.relevance_trainer = None
    
    def initialize_domains(self, documents: List[Dict[str, Any]]) -> None:
        """
        Initialize domain-specific caches with a corpus of documents
        
        Args:
            documents: List of documents for domain initialization
        """
        if self.enable_domain_partitioning and self.domain_cache:
            self.domain_cache.initialize_domains(documents)
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get search results from cache, with domain-specific lookup
        
        Args:
            query: The search query
            
        Returns:
            dict: Search results if found, None otherwise
        """
        result = None
        
        # Try domain-specific cache first if enabled
        if self.enable_domain_partitioning and self.domain_cache:
            # Normalize the query for domain lookup
            normalized_query = self._normalize_query(query)
            # Get from domain cache
            result = self.domain_cache.get(normalized_query, query=query)
            
        # Fall back to standard cache if no result found
        if result is None:
            result = super().get(query)
            
        # Apply RL-based ranking to the result if enabled and result found
        if (self.enable_rl_ranking and self.rl_ranker and result and 
            isinstance(result, dict) and 'search_result' in result):
            
            search_result = result['search_result']
            documents = search_result.get('documents', [])
            
            if documents:
                # Rank the documents
                ranked_docs = self.rl_ranker.rank_documents(query, documents)
                # Update the result with ranked documents
                search_result['documents'] = [doc['document'] for doc in ranked_docs]
                search_result['ranking_scores'] = [doc['score'] for doc in ranked_docs]
                search_result['ranking_method'] = 'rl'
                
                # Update the cache with the reranked result
                if not result.get('metadata'):
                    result['metadata'] = {}
                result['metadata']['reranked_by'] = 'rl'
                result['metadata']['ranking_time'] = time.time()
                
                # Don't update cache here as it would create a feedback loop
                    
        return result
    
    def set(
        self,
        query: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Cache a search result, with domain-specific storage
        
        Args:
            query: The search query
            result: The search result to cache
            ttl: Time-to-live in seconds
            metadata: Additional metadata
        """
        # Set in standard cache
        super().set(query, result, ttl, metadata)
        
        # Also set in domain-specific cache if enabled
        if self.enable_domain_partitioning and self.domain_cache:
            self.domain_cache.set(
                key=self._normalize_query(query),
                value=result,
                ttl=ttl,
                metadata=metadata,
                query=query
            )
    
    def record_interaction(
        self, 
        query: str, 
        document_id: str, 
        interaction_type: str, 
        document: Optional[Dict[str, Any]] = None,
        is_relevant: Optional[bool] = None
    ) -> None:
        """
        Record user interaction for learning
        
        Args:
            query: The search query
            document_id: ID of the document
            interaction_type: Type of interaction (click, dwell, etc.)
            document: The full document object (optional)
            is_relevant: Whether the document was relevant (optional)
        """
        # Default relevance based on interaction type if not explicitly provided
        if is_relevant is None:
            # Consider clicks and long dwell times as positive relevance signals
            is_relevant = interaction_type in ['click', 'long_dwell', 'save', 'share']
            
        # Record for RL-based ranking
        if self.enable_rl_ranking and self.rl_ranker:
            # Convert interaction to a numeric reward
            if interaction_type == 'click':
                value = 0.5
            elif interaction_type == 'long_dwell':
                value = 1.0
            elif interaction_type == 'short_dwell':
                value = -0.2
            elif interaction_type == 'skip':
                value = -0.5
            else:
                value = 0.0
                
            self.rl_ranker.record_interaction(query, document_id, interaction_type, value)
            
        # Record for self-supervised learning
        if (self.enable_self_supervised_learning and self.relevance_trainer and 
            document is not None and is_relevant is not None):
            self.relevance_trainer.record_interaction(query, document, is_relevant)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all components
        
        Returns:
            dict: Combined statistics
        """
        # Get base stats
        stats = super().get_stats()
        
        # Add domain partitioning stats
        if self.enable_domain_partitioning and self.domain_cache:
            stats['domain_partitioning'] = {
                'enabled': True,
                'num_domains': self.domain_cache.num_domains,
                'domains': self.domain_cache.get_domain_stats()
            }
        else:
            stats['domain_partitioning'] = {'enabled': False}
            
        # Add RL ranking stats
        if self.enable_rl_ranking and self.rl_ranker:
            stats['rl_ranking'] = {
                'enabled': True,
                **self.rl_ranker.get_stats()
            }
        else:
            stats['rl_ranking'] = {'enabled': False}
            
        # Add self-supervised learning stats
        if self.enable_self_supervised_learning and self.relevance_trainer:
            stats['self_supervised_learning'] = {
                'enabled': True,
                **self.relevance_trainer.get_stats()
            }
        else:
            stats['self_supervised_learning'] = {'enabled': False}
            
        return stats

# --- Singleton Accessor ---
_enhanced_search_cache_instance = None
_enhanced_cache_lock = threading.Lock()

def get_enhanced_search_cache(config: Optional[Dict[str, Any]] = None) -> EnhancedSearchCache:
    """
    Get a singleton instance of the EnhancedSearchCache
    
    Args:
        config: Configuration settings dictionary (used only on first initialization)
        
    Returns:
        EnhancedSearchCache: The singleton instance
    """
    global _enhanced_search_cache_instance
    
    if _enhanced_search_cache_instance is None:
        with _enhanced_cache_lock:
            if _enhanced_search_cache_instance is None:
                logger.info("Creating EnhancedSearchCache singleton instance")
                
                # Extract cache-specific config
                cache_config = config or {}
                if 'cache' in cache_config and isinstance(cache_config['cache'], dict):
                    cache_config = cache_config['cache']
                
                # Determine which features to enable
                enable_domain_partitioning = cache_config.get('enable_domain_partitioning', True)
                enable_rl_ranking = cache_config.get('enable_rl_ranking', True)
                enable_self_supervised_learning = cache_config.get('enable_self_supervised_learning', True)
                num_domains = int(cache_config.get('num_domains', 5))
                
                # Create the base cache implementation
                cache_impl = MultiLevelCache(
                    memory_size=int(cache_config.get('memory_size', 100)),
                    disk_cache_dir=cache_config.get('disk_cache_dir', '.cache/disk'),
                    disk_max_size_mb=int(cache_config.get('disk_max_size_mb', 100)),
                    db_path=cache_config.get('db_path', '.cache/db/cache.db'),
                    db_max_entries=int(cache_config.get('db_max_entries', 1000)),
                    default_ttl=int(cache_config.get('default_ttl', 86400)),
                    enable_memory=cache_config.get('enable_memory', True),
                    enable_disk=cache_config.get('enable_disk', True),
                    enable_db=cache_config.get('enable_db', True)
                )
                
                # Create the enhanced search cache
                _enhanced_search_cache_instance = EnhancedSearchCache(
                    cache_impl=cache_impl,
                    default_ttl=int(cache_config.get('default_ttl', 86400)),
                    query_similarity_threshold=float(cache_config.get('query_similarity_threshold', 0.8)),
                    enable_domain_partitioning=enable_domain_partitioning,
                    enable_rl_ranking=enable_rl_ranking,
                    enable_self_supervised_learning=enable_self_supervised_learning,
                    num_domains=num_domains
                )
                
    return _enhanced_search_cache_instance
