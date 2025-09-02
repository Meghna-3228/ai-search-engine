# config_manager.py

"""
Configuration Manager Module for AI-powered Search

This module provides advanced configuration management with:

1. Multiple configuration profiles for different use cases
2. Environment-based configuration loading
3. Dynamic configuration validation
4. Profile switching and merging
5. Configuration persistence and export/import
6. Ability to update specific settings in the active profile

The goal is to make the application highly configurable while maintaining stability.
"""

import os
import sys
import json
# import yaml # Keep commented if PyYAML is optional
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, Tuple # Added Tuple
from datetime import datetime
from functools import lru_cache
import copy
import re

# Configure logging
# Assuming basicConfig is called elsewhere or using Streamlit's default logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger("config_manager")

# Default configuration
DEFAULT_CONFIG = {
    "app": {
        "name": "AI-powered Search",
        "version": "1.0.0",
        "debug": False,
        "log_level": "INFO"
    },
    "search_engine": {
        "default_provider": "cohere",
        "use_cache": True,
        "cache_ttl": 86400, # 24 hours
        "analyze_results": True,
        "use_fallback": True,
        "max_fallbacks": 2,
        "max_concurrent_searches": 5,
        "providers": {
            "cohere": {
                "model": "command",
                "temperature": 0.5,
                "max_tokens": 1000,
                "connector_id": "web-search"
            },
            "openai": {
                "model": "gpt-4-turbo",
                "temperature": 0.5,
                "max_tokens": 1000
            },
            "anthropic": {
                "model": "claude-3-opus-20240229",
                "temperature": 0.5,
                "max_tokens": 1000
            },
            "web_api": {
                "api_url": "",
                "api_key_header": "Authorization",
                "timeout": 10
            }
        } # Added missing closing brace in original
    }, # Added missing comma in original
    "cache": {
        "cache_type": "multi",
        "memory_size": 100,
        "disk_cache_dir": ".cache",
        "disk_max_size_mb": 100,
        "db_path": "cache.db",
        "db_max_entries": 1000,
        "query_similarity_threshold": 0.8,
        "default_ttl": 86400
    },
    "appearance": {
        "theme": "dark",
        "show_visualizations": True,
        "show_sources": True,
        "advanced_mode": False,
        "layout": "wide"
    } # Added missing closing brace in original
} # Added missing closing brace in original

# Schema for configuration validation
CONFIG_SCHEMA = {
    "app": {
        "name": {"type": "str", "required": True},
        "version": {"type": "str", "required": True},
        "debug": {"type": "bool", "required": True},
        "log_level": {"type": "str", "values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "required": True}
    },
    "search_engine": {
        "default_provider": {"type": "str", "required": True}, # Add "values" later from available providers
        "use_cache": {"type": "bool", "required": True},
        "cache_ttl": {"type": "int", "min": 60, "max": 2592000, "required": True},
        "analyze_results": {"type": "bool", "required": True},
        "use_fallback": {"type": "bool", "required": True},
        "max_fallbacks": {"type": "int", "min": 0, "max": 10, "required": True},
        "max_concurrent_searches": {"type": "int", "min": 1, "max": 20, "required": True},
        "providers": {
            "cohere": {
                "model": {"type": "str", "required": False},
                "temperature": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
                "max_tokens": {"type": "int", "min": 100, "max": 4000, "required": False},
                "connector_id": {"type": "str", "required": False}
            },
            "openai": {
                "model": {"type": "str", "required": False},
                "temperature": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
                "max_tokens": {"type": "int", "min": 100, "max": 4000, "required": False}
            },
            "anthropic": {
                "model": {"type": "str", "required": False},
                "temperature": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
                "max_tokens": {"type": "int", "min": 100, "max": 4000, "required": False}
            },
            "web_api": {
                "api_url": {"type": "str", "required": False},
                "api_key_header": {"type": "str", "required": False},
                "timeout": {"type": "int", "min": 1, "max": 60, "required": False}
            } # Added missing closing brace in original
        } # Added missing comma in original
    }, # Added missing closing brace from outer structure? No, seems correct now.
    "cache": {
        "cache_type": {"type": "str", "values": ["multi", "memory", "disk", "db"], "required": True},
        "memory_size": {"type": "int", "min": 10, "max": 1000, "required": True},
        "disk_cache_dir": {"type": "str", "required": True},
        "disk_max_size_mb": {"type": "int", "min": 10, "max": 1000, "required": True},
        "db_path": {"type": "str", "required": True},
        "db_max_entries": {"type": "int", "min": 100, "max": 10000, "required": True},
        "query_similarity_threshold": {"type": "float", "min": 0.5, "max": 1.0, "required": True},
        "default_ttl": {"type": "int", "min": 60, "max": 2592000, "required": True}
    },
    "appearance": {
        "theme": {"type": "str", "values": ["light", "dark", "auto"], "required": True},
        "show_visualizations": {"type": "bool", "required": True},
        "show_sources": {"type": "bool", "required": True},
        "advanced_mode": {"type": "bool", "required": True},
        "layout": {"type": "str", "values": ["wide", "centered"], "required": True}
    } # Added missing closing brace in original
} # Added missing closing brace in original

class ConfigError(Exception):
    """Exception raised for configuration errors"""
    pass

class ConfigValidator:
    """
    Validator for configuration values
    """
    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> List[str]:
        """
        Validate the configuration against a schema
        Args:
            config: Configuration to validate
            schema: Schema to validate against
            path: Current path in the configuration
        Returns:
            list: List of validation errors
        """
        errors = []
        # Check if config is a dictionary
        if not isinstance(config, dict):
            errors.append(f"Configuration at '{path}' should be a dictionary, got {type(config).__name__}")
            return errors

        # Check if schema is a dictionary
        if not isinstance(schema, dict):
            # This indicates an internal issue with the schema itself
            logger.error(f"Internal Error: Schema at '{path}' should be a dictionary, got {type(schema).__name__}")
            errors.append(f"Internal Schema Error at '{path}'")
            return errors

        # Check for required fields and validate present fields
        known_schema_keys = set()
        for key, field_schema in schema.items():
            # Skip non-dictionary schema entries (like "type", "required", "values")
            # Check if field_schema itself defines a field (contains 'type') or is a nested schema dict
            if not isinstance(field_schema, dict) or 'type' not in field_schema:
                # It might be a nested schema structure
                if isinstance(field_schema, dict):
                    known_schema_keys.add(key) # This key corresponds to a nested structure
                    # Validate nested structure recursively
                    current_path = f"{path}.{key}" if path else key
                    if key in config and config[key] is not None:
                        if isinstance(config[key], dict):
                            nested_errors = ConfigValidator.validate_config(config[key], field_schema, current_path)
                            errors.extend(nested_errors)
                        else:
                            errors.append(f"Expected dictionary for nested config at '{current_path}', got {type(config[key]).__name__}")
                    elif field_schema.get("required", False): # Check if nested structure itself is required
                         errors.append(f"Required nested configuration '{current_path}' is missing")
                    continue # Move to the next key in the schema
                else:
                    # Should not happen with correct schema definition, ignore non-dict/non-type entries
                    continue

            # --- Process schema entries that define a field (have 'type') ---
            known_schema_keys.add(key)
            current_path = f"{path}.{key}" if path else key

            # Check if the field is required and missing
            required = field_schema.get("required", False)
            if required and (key not in config or config[key] is None):
                errors.append(f"Required field '{current_path}' is missing")
                continue # Skip further validation if required field is missing

            # If field is present, validate it
            if key in config and config[key] is not None:
                value = config[key]
                type_errors = ConfigValidator._validate_type(value, field_schema, current_path)
                errors.extend(type_errors)

        # Check for unknown fields in the config compared to known schema keys
        for key in config:
            if key not in known_schema_keys:
                unknown_path = f"{path}.{key}" if path else key
                errors.append(f"Unknown configuration field '{unknown_path}'")

        return errors


    @staticmethod
    def _validate_type(value: Any, field_schema: Dict[str, Any], path: str) -> List[str]:
        """
        Validate a value against a field schema containing 'type' and constraints.
        Args:
            value: Value to validate
            field_schema: Schema dictionary for the specific field (must contain 'type').
            path: Current path in the configuration for error messages.
        Returns:
            list: List of validation errors for this specific value.
        """
        errors = []
        field_type = field_schema.get("type")

        if field_type is None:
            # Schema definition error
            logger.error(f"Internal Error: Schema for '{path}' is missing 'type'.")
            errors.append(f"Internal Schema Error: Missing type for '{path}'.")
            return errors

        # Validate by type
        if field_type == "str":
            if not isinstance(value, str):
                errors.append(f"Field '{path}' should be a string, got {type(value).__name__}")
            elif "values" in field_schema and value not in field_schema["values"]:
                valid_values = ", ".join(f"'{v}'" for v in field_schema["values"]) # Quote values
                errors.append(f"Field '{path}' should be one of [{valid_values}], got '{value}'")

        elif field_type == "int":
            # Allow float input if it's equivalent to an integer (e.g., 10.0)
            # But the stored type should ideally be int after fixing. Validation checks actual type.
            if not isinstance(value, int):
                errors.append(f"Field '{path}' should be an integer, got {type(value).__name__}")
            else: # Value is an int, check constraints
                if "min" in field_schema and value < field_schema["min"]:
                    errors.append(f"Field '{path}' should be >= {field_schema['min']}, got {value}")
                if "max" in field_schema and value > field_schema["max"]:
                    errors.append(f"Field '{path}' should be <= {field_schema['max']}, got {value}")

        elif field_type == "float":
            if not isinstance(value, (int, float)):
                errors.append(f"Field '{path}' should be a number (float or int), got {type(value).__name__}")
            else: # Value is number, check constraints
                num_value = float(value) # Convert to float for comparison
                if "min" in field_schema and num_value < field_schema["min"]:
                    errors.append(f"Field '{path}' should be >= {field_schema['min']}, got {num_value}")
                if "max" in field_schema and num_value > field_schema["max"]:
                    errors.append(f"Field '{path}' should be <= {field_schema['max']}, got {num_value}")

        elif field_type == "bool":
            if not isinstance(value, bool):
                errors.append(f"Field '{path}' should be a boolean (true/false), got {type(value).__name__}")

        elif field_type == "list":
            if not isinstance(value, list):
                errors.append(f"Field '{path}' should be a list, got {type(value).__name__}")
            else:
                # Validate item type if specified in schema
                if "item_type" in field_schema:
                    item_schema = {"type": field_schema["item_type"]}
                    if "item_values" in field_schema: # Enum constraint for list items
                        item_schema["values"] = field_schema["item_values"]
                    # Add min/max constraints for numeric list items if needed
                    # if "item_min" in field_schema: item_schema["min"] = field_schema["item_min"]
                    # if "item_max" in field_schema: item_schema["max"] = field_schema["item_max"]

                    for i, item in enumerate(value):
                        item_path = f"{path}[{i}]"
                        item_errors = ConfigValidator._validate_type(item, item_schema, item_path)
                        errors.extend(item_errors)

        elif field_type == "dict":
            if not isinstance(value, dict):
                errors.append(f"Field '{path}' should be a dictionary, got {type(value).__name__}")
            else:
                # Validate key and value types if specified
                # Key type validation is less common for JSON-like structures (keys usually strings)
                if "key_type" in field_schema:
                    key_schema = {"type": field_schema["key_type"]}
                    for key in value.keys():
                        key_path = f"{path}.key" # Generic path for key type check
                        key_errors = ConfigValidator._validate_type(key, key_schema, key_path)
                        errors.extend(key_errors)

                if "value_type" in field_schema:
                    value_schema = {"type": field_schema["value_type"]}
                    # Add constraints for dict values if needed (min/max, values etc.)
                    # if "value_min" in field_schema: value_schema["min"] = field_schema["value_min"]
                    # if "value_max" in field_schema: value_schema["max"] = field_schema["value_max"]
                    # if "value_values" in field_schema: value_schema["values"] = field_schema["value_values"]

                    for dict_key, dict_val in value.items():
                        # Path uses the actual key for better error messages
                        value_path = f"{path}.{dict_key}"
                        value_errors = ConfigValidator._validate_type(dict_val, value_schema, value_path)
                        errors.extend(value_errors)

                # Could add recursive validation here if dict values have their own complex sub-schema
                # This is partially handled by the main validate_config structure if the schema is nested.

        else:
            # Unsupported type in schema definition
            logger.error(f"Internal Error: Unsupported schema type '{field_type}' for '{path}'.")
            errors.append(f"Internal Schema Error: Unsupported type '{field_type}' for '{path}'.")

        return errors


    @staticmethod
    def fix_config(config: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """
        Recursively fix invalid configuration values based on the schema.
        Removes unknown keys and attempts to conform values to specified types and constraints.

        Args:
            config: Configuration dictionary to fix.
            schema: Schema dictionary to use for fixing.
            path: Current path in the configuration for logging.

        Returns:
            dict: Fixed configuration dictionary. Returns an empty dict or original if inputs are invalid.
        """
        fixed_config = copy.deepcopy(config)

        # Ensure config is a dictionary
        if not isinstance(fixed_config, dict):
            logger.warning(f"Cannot fix non-dictionary configuration at '{path}'. Returning empty dict.")
            return {}

        # Ensure schema is a dictionary
        if not isinstance(schema, dict):
            logger.warning(f"Cannot fix with non-dictionary schema at '{path}'. Returning original config.")
            return fixed_config

        known_schema_keys = set()
        # Iterate through schema to fix config based on schema definitions
        for key, field_schema in schema.items():
            current_path = f"{path}.{key}" if path else key

            # Handle nested schema structures (no 'type' key at this level)
            if isinstance(field_schema, dict) and 'type' not in field_schema:
                known_schema_keys.add(key)
                required = field_schema.get("required", False) # Nested structure required?

                if key not in fixed_config or fixed_config[key] is None:
                    if required:
                        # If required nested structure is missing, create an empty dict and recurse fix
                        logger.info(f"Adding missing required nested config structure at '{current_path}'.")
                        fixed_config[key] = {}
                        # Recurse to fill defaults within the newly created structure
                        fixed_config[key] = ConfigValidator.fix_config({}, field_schema, current_path)
                    # else: # Optional nested structure is missing, do nothing
                elif isinstance(fixed_config[key], dict):
                    # Recursively fix the existing nested dictionary
                    fixed_config[key] = ConfigValidator.fix_config(fixed_config[key], field_schema, current_path)
                else:
                    # Config has a non-dict value where schema expects a nested structure
                    logger.warning(f"Expected dictionary for nested config at '{current_path}', got {type(fixed_config[key]).__name__}. Replacing with default structure.")
                    # Replace with default fixed structure or empty dict if required
                    if required:
                         fixed_config[key] = ConfigValidator.fix_config({}, field_schema, current_path)
                    else:
                         del fixed_config[key] # Remove invalid entry for optional nested structure

                continue # Move to the next key in the schema

            # Handle schema entries that define a field (have 'type')
            elif isinstance(field_schema, dict) and 'type' in field_schema:
                known_schema_keys.add(key)
                required = field_schema.get("required", False)
                is_missing = key not in fixed_config or fixed_config[key] is None

                if required and is_missing:
                    # Fix missing required field by setting a default
                    default_value = ConfigValidator._get_default_for_field(field_schema, current_path)
                    fixed_config[key] = default_value
                    logger.info(f"Set default value for missing required field '{current_path}': {repr(default_value)}")
                elif key in fixed_config and fixed_config[key] is not None:
                    # Field exists, fix its type and constraints
                    fixed_config[key] = ConfigValidator._fix_type(fixed_config[key], field_schema, current_path)
                # else: # Field is not required and missing, or is None - leave as is or remove?
                      # Current deepcopy behaviour keeps None. If removal is desired, add logic here.

            # else: # Schema entry is not a dict or doesn't define type/nested structure - ignore.
                  # logger.debug(f"Skipping schema entry for '{key}' at '{path}'.")


        # Remove unknown keys found in the config but not in the schema
        keys_to_remove = [key for key in fixed_config if key not in known_schema_keys]
        for key in keys_to_remove:
            removed_path = f"{path}.{key}" if path else key
            logger.warning(f"Removing unknown configuration field '{removed_path}' during fix.")
            try:
                del fixed_config[key]
            except KeyError: # Should not happen with the loop logic, but safety check
                pass

        return fixed_config


    @staticmethod
    def _get_default_for_field(field_schema: Dict[str, Any], path: str) -> Any:
        """Helper to determine the default value for a field based on schema."""
        if "default" in field_schema:
            return field_schema["default"]

        field_type = field_schema.get("type")
        if field_type == "str": return ""
        elif field_type == "int": return field_schema.get("min", 0)
        elif field_type == "float": return field_schema.get("min", 0.0)
        elif field_type == "bool": return False
        elif field_type == "list": return []
        elif field_type == "dict": return {}
        else:
            logger.error(f"Cannot determine default value for field '{path}' with type '{field_type}'. Using None.")
            return None


    @staticmethod
    def _fix_type(value: Any, field_schema: Dict[str, Any], path: str) -> Any:
        """
        Attempt to fix a single value to match the expected type and constraints in the schema.
        Args:
            value: Value to fix.
            field_schema: Schema definition for the field (must contain 'type').
            path: Current path in the configuration for logging.
        Returns:
            Any: The potentially fixed value. Returns original value if type is unsupported or fix fails.
        """
        if not isinstance(field_schema, dict) or "type" not in field_schema:
            logger.error(f"Internal error: Invalid field_schema passed to _fix_type for path '{path}'.")
            return value # Cannot fix without type info

        field_type = field_schema["type"]
        fixed_value = value # Start with original value

        # --- String ---
        if field_type == "str":
            if not isinstance(value, str):
                try:
                    fixed_value = str(value) if value is not None else ""
                    logger.warning(f"Fixed non-string value for '{path}': {repr(value)} -> '{fixed_value}'")
                except Exception:
                    fixed_value = "" # Fallback on conversion error
                    logger.warning(f"Failed to convert value to string for '{path}'. Using empty string.")
            # Check enum constraint after type conversion
            if "values" in field_schema and fixed_value not in field_schema["values"]:
                default_enum = field_schema["values"][0] if field_schema.get("values") else ""
                logger.warning(f"Fixed invalid enum value for '{path}': '{fixed_value}' is not in {field_schema['values']}. Using default: '{default_enum}'")
                fixed_value = default_enum

        # --- Integer ---
        elif field_type == "int":
            if isinstance(value, int):
                fixed_value = value # Already correct type
            elif isinstance(value, float) and value.is_integer():
                fixed_value = int(value) # Convert float like 10.0 to int 10
                logger.warning(f"Converted integer-like float to int for '{path}': {value} -> {fixed_value}")
            else:
                try:
                    # Try converting from string or other types
                    parsed_float = float(value)
                    fixed_value = int(parsed_float)
                    # Check if precision was lost during float conversion if original wasn't float
                    if not isinstance(value, float) and fixed_value != parsed_float:
                         logger.warning(f"Potential precision loss during conversion to int for '{path}': {repr(value)} -> {fixed_value}")
                    else:
                         logger.warning(f"Fixed non-integer value for '{path}': {repr(value)} -> {fixed_value}")
                except (ValueError, TypeError, OverflowError):
                    default_int = field_schema.get("min", 0) # Fallback to min or 0
                    logger.warning(f"Failed to convert to int for '{path}': {repr(value)}. Using default: {default_int}")
                    fixed_value = default_int

            # Apply min/max constraints after ensuring it's an int
            if isinstance(fixed_value, int):
                if "min" in field_schema and fixed_value < field_schema["min"]:
                    logger.warning(f"Clamped out-of-range integer for '{path}': {fixed_value} -> {field_schema['min']} (min)")
                    fixed_value = field_schema["min"]
                if "max" in field_schema and fixed_value > field_schema["max"]:
                    logger.warning(f"Clamped out-of-range integer for '{path}': {fixed_value} -> {field_schema['max']} (max)")
                    fixed_value = field_schema["max"]

        # --- Float ---
        elif field_type == "float":
            if isinstance(value, (int, float)):
                 fixed_value = float(value) # Ensure it's float type
            else:
                try:
                    fixed_value = float(value)
                    logger.warning(f"Fixed non-float value for '{path}': {repr(value)} -> {fixed_value}")
                except (ValueError, TypeError):
                    default_float = field_schema.get("min", 0.0) # Fallback to min or 0.0
                    logger.warning(f"Failed to convert to float for '{path}': {repr(value)}. Using default: {default_float}")
                    fixed_value = default_float

            # Apply min/max constraints after ensuring it's a float
            if isinstance(fixed_value, float):
                if "min" in field_schema and fixed_value < field_schema["min"]:
                    logger.warning(f"Clamped out-of-range float for '{path}': {fixed_value} -> {field_schema['min']} (min)")
                    fixed_value = field_schema["min"]
                if "max" in field_schema and fixed_value > field_schema["max"]:
                    logger.warning(f"Clamped out-of-range float for '{path}': {fixed_value} -> {field_schema['max']} (max)")
                    fixed_value = field_schema["max"]

        # --- Boolean ---
        elif field_type == "bool":
            if not isinstance(value, bool):
                if isinstance(value, str):
                    fixed_value = value.strip().lower() in ["true", "yes", "1", "y", "t", "on"]
                elif isinstance(value, (int, float)):
                    fixed_value = value > 0
                else:
                    fixed_value = bool(value) # Fallback standard bool conversion
                logger.warning(f"Fixed non-boolean value for '{path}': {repr(value)} -> {fixed_value}")

        # --- List ---
        elif field_type == "list":
            if not isinstance(value, list):
                # Attempt to handle common cases like comma-separated strings? Maybe too complex/risky.
                # Default: wrap non-list items (unless None) or default to empty list.
                fixed_value = [value] if value is not None else []
                logger.warning(f"Fixed non-list value for '{path}': {repr(value)} -> {fixed_value}")

            # Fix item types recursively if specified in schema
            if isinstance(fixed_value, list) and "item_type" in field_schema:
                item_schema = {"type": field_schema["item_type"]}
                # Pass item constraints if they exist
                if "item_values" in field_schema: item_schema["values"] = field_schema["item_values"]
                # if "item_min" in field_schema: item_schema["min"] = field_schema["item_min"]
                # if "item_max" in field_schema: item_schema["max"] = field_schema["item_max"]

                fixed_items = []
                for i, item in enumerate(fixed_value):
                    item_path = f"{path}[{i}]"
                    fixed_item = ConfigValidator._fix_type(item, item_schema, item_path)
                    fixed_items.append(fixed_item)
                fixed_value = fixed_items

        # --- Dictionary ---
        elif field_type == "dict":
            if not isinstance(value, dict):
                fixed_value = {}
                logger.warning(f"Fixed non-dict value for '{path}': {repr(value)} -> {fixed_value}")

            # Fix key and value types if specified
            # Note: Fixing keys is complex (type changes, collisions). Usually only value fixing is safe.
            # We will focus on fixing values here.
            if isinstance(fixed_value, dict) and ("key_type" in field_schema or "value_type" in field_schema):
                temp_fixed_dict = {}
                # Schema for value type
                value_schema = None
                if "value_type" in field_schema:
                    value_schema = {"type": field_schema["value_type"]}
                    # Add value constraints if needed
                    # if "value_min" in field_schema: value_schema["min"] = field_schema["value_min"]
                    # if "value_max" in field_schema: value_schema["max"] = field_schema["value_max"]
                    # if "value_values" in field_schema: value_schema["values"] = field_schema["value_values"]

                for key, val in fixed_value.items():
                    # Key fixing (use cautiously)
                    fixed_key = key
                    # if "key_type" in field_schema: ... logic to fix key ...

                    # Value fixing
                    fixed_val = val
                    if value_schema:
                        value_path = f"{path}.{key}" # Path uses original key
                        fixed_val = ConfigValidator._fix_type(val, value_schema, value_path)

                    temp_fixed_dict[fixed_key] = fixed_val
                fixed_value = temp_fixed_dict

        # --- Other Types ---
        else:
            logger.warning(f"Unsupported type '{field_type}' specified in schema for fixing at '{path}'. Returning original value.")

        return fixed_value


class ConfigProfile:
    """
    Represents a configuration profile with metadata.
    """
    def __init__(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        """
        Initialize a configuration profile.
        Args:
            name: Unique name for the profile.
            description: Optional description.
            config: Dictionary containing the configuration settings.
            is_default: Flag indicating if this is the default profile.
            created_at: Timestamp of creation.
            updated_at: Timestamp of last update.
        """
        self.name = name
        self.description = description
        # Ensure config is always a dictionary, deep copy DEFAULT_CONFIG if None
        # Use deepcopy to prevent aliasing issues if config dict is reused
        self.config = copy.deepcopy(config) if config is not None else copy.deepcopy(DEFAULT_CONFIG)
        self.is_default = is_default
        # Handle potential string timestamps from deserialization if needed, though from_dict does it
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profile to a dictionary for serialization.
        Returns:
            dict: Dictionary representation of the profile.
        """
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config, # Already a dict
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else str(self.updated_at)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigProfile':
        """
        Create a profile instance from a dictionary. Robustly handles missing/invalid data.
        Args:
            data: Dictionary representation of a profile.
        Returns:
            ConfigProfile: A new profile instance. Raises ValueError if essential data is missing.
        """
        if not data or "name" not in data:
             raise ValueError("Cannot create ConfigProfile from dict: 'name' is missing.")

        created_at_obj = None
        if "created_at" in data and data["created_at"]:
            try:
                created_at_obj = datetime.fromisoformat(data["created_at"])
            except (TypeError, ValueError):
                 logger.warning(f"Could not parse 'created_at' timestamp '{data['created_at']}' for profile '{data['name']}'. Using None.")

        updated_at_obj = None
        if "updated_at" in data and data["updated_at"]:
             try:
                 updated_at_obj = datetime.fromisoformat(data["updated_at"])
             except (TypeError, ValueError):
                 logger.warning(f"Could not parse 'updated_at' timestamp '{data['updated_at']}' for profile '{data['name']}'. Using None.")

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            config=data.get("config", {}), # Ensure config is dict, default empty
            is_default=data.get("is_default", False),
            created_at=created_at_obj,
            updated_at=updated_at_obj
        )


    def validate(self) -> List[str]:
        """
        Validate the profile's configuration against the global schema.
        Returns:
            list: List of validation error messages. Empty list if valid.
        """
        return ConfigValidator.validate_config(self.config, CONFIG_SCHEMA)

    def fix(self) -> 'ConfigProfile':
        """
        Attempt to fix the profile's configuration based on the global schema.
        Returns a *new* profile instance with the fixed configuration and updated timestamp.
        The original profile instance remains unchanged.

        Returns:
            ConfigProfile: A new profile instance with fixed configuration.
        """
        logger.debug(f"Attempting to fix profile '{self.name}'.")
        fixed_config = ConfigValidator.fix_config(self.config, CONFIG_SCHEMA)
        # Create a new instance to reflect the fix potentially changed the state
        return ConfigProfile(
            name=self.name,
            description=self.description,
            config=fixed_config,
            is_default=self.is_default,
            created_at=self.created_at, # Keep original creation time
            updated_at=datetime.now() # Update modification time for the new fixed instance
        )


    def merge(self, other: 'ConfigProfile') -> 'ConfigProfile':
        """
        Merge this profile's configuration with another profile's configuration.
        The other profile's values take precedence in case of conflicts ('other' overrides 'self').
        Returns a *new* profile instance representing the merge.

        Args:
            other: The profile to merge with.

        Returns:
            ConfigProfile: A new profile instance with the merged configuration.
        """
        if not isinstance(other, ConfigProfile):
            raise TypeError("Can only merge with another ConfigProfile instance.")

        # Create a new profile for the result
        merged_profile = ConfigProfile(
            name=f"{self.name}_merged_{other.name}", # Suggest a name
            description=f"Merged profile from '{self.name}' and '{other.name}'",
            is_default=False, # Merged profile is typically not default
            created_at=datetime.now(), # New entity
            updated_at=datetime.now()
        )

        # Perform deep merge of configurations (other takes precedence)
        merged_config = merge_configs(self.config, other.config)
        merged_profile.config = merged_config
        return merged_profile

class ConfigManager:
    """
    Manages multiple configuration profiles, persistence, and environment overrides.
    Provides methods for creating, updating, deleting, listing, activating,
    importing, exporting, and resetting profiles.
    """
    def __init__(
        self,
        config_dir: str = ".config",
        profiles_file: str = "profiles.json",
        env_prefix: str = "APP_"
    ):
        """
        Initialize the configuration manager.
        Args:
            config_dir: Directory to store configuration files.
            profiles_file: Filename for storing profile definitions.
            env_prefix: Prefix for environment variables used for overrides.
        """
        self.config_dir = Path(config_dir)
        self.profiles_file_name = profiles_file # Store filename separately
        self.profiles_path = self.config_dir / self.profiles_file_name
        self.env_prefix = env_prefix
        self.profiles: Dict[str, ConfigProfile] = {}
        self.active_profile: Optional[str] = None
        self.lock = threading.RLock() # Use RLock for potential nested calls within locked methods

        # Ensure config directory exists
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create config directory '{self.config_dir}': {e}. Configuration may not persist.")
            # Depending on severity, might raise ConfigError(e)

        # Load existing profiles
        self._load_profiles()

        # Ensure a default profile exists and is active if needed
        self._ensure_default_profile()


    def _ensure_default_profile(self):
        """ Ensures at least one profile exists, creating a default if none are loaded,
            and sets an active profile if none is set. """
        with self.lock:
            if not self.profiles:
                logger.info("No profiles found or loaded. Creating default profile.")
                self._create_default_profile() # This also sets it active and saves
            elif self.active_profile is None or self.active_profile not in self.profiles:
                logger.warning(f"Active profile '{self.active_profile}' is invalid or not set.")
                # Find the profile marked as default
                default_profile_name = next((name for name, prof in self.profiles.items() if prof.is_default), None)
                if default_profile_name:
                    self.active_profile = default_profile_name
                    logger.info(f"Setting active profile to default '{self.active_profile}' from loaded profiles.")
                elif self.profiles: # If no default found, use the first one loaded alphabetically
                    first_profile_name = sorted(self.profiles.keys())[0]
                    self.active_profile = first_profile_name
                    logger.info(f"No default profile found. Setting active profile to first loaded (alpha): '{self.active_profile}'.")
                else:
                    # Should not happen if _create_default_profile was called, but safety check.
                     logger.error("Cannot set active profile: No profiles available.")
                # Save the potentially changed active profile setting
                self._save_profiles()


    def _load_profiles(self):
        """Load profiles from the JSON file."""
        if not self.profiles_path.is_file():
            logger.info(f"Profiles file not found at {self.profiles_path}. Skipping load.")
            return

        try:
            with self.profiles_path.open("r", encoding='utf-8') as f:
                data = json.load(f)

            loaded_profiles_data = data.get("profiles", [])
            # Load active profile name, but don't set it yet, validate first
            loaded_active_profile_name = data.get("active_profile")

            loaded_profiles: Dict[str, ConfigProfile] = {}
            load_errors = []

            for profile_data in loaded_profiles_data:
                if not isinstance(profile_data, dict) or "name" not in profile_data:
                    load_errors.append(f"Skipping invalid profile data entry: {profile_data}")
                    continue

                profile_name = profile_data["name"]
                try:
                    profile = ConfigProfile.from_dict(profile_data)

                    # Validate and potentially fix loaded profile config
                    errors = profile.validate()
                    if errors:
                        logger.warning(f"Profile '{profile.name}' loaded with validation errors: {errors}")
                        logger.info(f"Attempting to fix profile '{profile.name}'...")
                        fixed_profile = profile.fix() # fix() returns a new instance

                        # Re-validate the fixed profile
                        fixed_errors = fixed_profile.validate()
                        if fixed_errors:
                            logger.error(f"Fixing profile '{profile.name}' resulted in new validation errors: {fixed_errors}. Skipping load of this profile.")
                            load_errors.append(f"Failed to fix profile '{profile.name}'.")
                            continue # Skip loading this profile
                        else:
                            logger.info(f"Profile '{profile.name}' fixed successfully.")
                            loaded_profiles[fixed_profile.name] = fixed_profile
                    else:
                        # Profile is valid as loaded
                        loaded_profiles[profile.name] = profile

                except (ValueError, TypeError, Exception) as e:
                    logger.error(f"Failed to load individual profile '{profile_name}' from data: {profile_data}. Error: {e}", exc_info=True)
                    load_errors.append(f"Error loading profile '{profile_name}': {e}")

            # Update internal state only after successful processing
            self.profiles = loaded_profiles
            self.active_profile = None # Reset before setting potentially valid one

            # Set active profile if the loaded name is valid
            if loaded_active_profile_name and loaded_active_profile_name in self.profiles:
                self.active_profile = loaded_active_profile_name
            elif loaded_active_profile_name:
                logger.warning(f"Loaded active profile name '{loaded_active_profile_name}' is invalid. Will select a default/first profile.")

            logger.info(f"Loaded {len(self.profiles)} profiles from {self.profiles_path}.")
            if load_errors:
                 logger.warning(f"Encountered errors during profile loading: {len(load_errors)} errors. Details: {load_errors}")

        except (json.JSONDecodeError, IOError, Exception) as e:
            logger.error(f"Failed to load or parse profiles file '{self.profiles_path}': {e}", exc_info=True)
            # Reset state in case of partial load or corruption
            self.profiles = {}
            self.active_profile = None

    def _save_profiles(self):
        """Save the current state of profiles and active profile to the JSON file."""
        # Check if directory exists, attempt creation if missing (e.g., deleted after init)
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot save profiles: Config directory '{self.config_dir}' does not exist and cannot be created: {e}")
            return # Abort save

        with self.lock:
            try:
                # Prepare data for saving
                profiles_data = [profile.to_dict() for profile in self.profiles.values()]
                data_to_save = {
                    "profiles": profiles_data,
                    "active_profile": self.active_profile,
                    "updated_at": datetime.now().isoformat()
                }

                # Save to file atomically (write to temp then rename)
                temp_path = self.profiles_path.with_suffix(f".tmp_{os.getpid()}") # More unique temp file
                with temp_path.open("w", encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)

                # On Windows, replace might fail if target exists. Try removing first.
                if sys.platform == "win32" and self.profiles_path.exists():
                     try:
                          self.profiles_path.unlink()
                     except OSError as e:
                          logger.warning(f"Could not remove existing profiles file before rename: {e}")
                # Atomic rename/replace
                temp_path.replace(self.profiles_path)

                logger.info(f"Saved {len(self.profiles)} profiles to {self.profiles_path}")

            except (IOError, TypeError, OSError, Exception) as e:
                logger.error(f"Failed to save profiles to '{self.profiles_path}': {e}", exc_info=True)
                # Clean up temp file if it still exists
                if temp_path.exists():
                     try:
                          temp_path.unlink()
                     except OSError:
                          logger.warning(f"Could not clean up temporary save file '{temp_path}'.")


    def _create_default_profile(self):
        """Create a default profile using DEFAULT_CONFIG if it doesn't exist."""
        with self.lock:
            default_name = "default"
            if default_name in self.profiles:
                logger.debug(f"Default profile '{default_name}' already exists.")
                # Ensure it's marked as default if no others are
                has_other_default = any(p.is_default for n, p in self.profiles.items() if n != default_name)
                if not has_other_default and not self.profiles[default_name].is_default:
                    self.profiles[default_name].is_default = True
                    self.profiles[default_name].updated_at = datetime.now()
                    logger.info(f"Marked existing profile '{default_name}' as default.")
                    self._save_profiles()
                return

            logger.info(f"Creating new default profile '{default_name}'.")
            default_profile = ConfigProfile(
                name=default_name,
                description="Default configuration profile",
                config=copy.deepcopy(DEFAULT_CONFIG), # Use deep copy
                is_default=True # Mark as default initially
            )

            # Validate the default config itself (should pass if schema is correct)
            errors = default_profile.validate()
            if errors:
                logger.error(f"DEFAULT_CONFIG validation failed: {errors}. Cannot create default profile reliably.")
                # Decide whether to add it anyway (potentially broken) or raise error
                # For now, add it but log error prominently.
                # raise ConfigError(f"DEFAULT_CONFIG validation failed: {errors}")
            else:
                 logger.debug("DEFAULT_CONFIG validated successfully.")

            # Unset default flag on any other existing profiles (shouldn't be any if called from init)
            for prof in self.profiles.values():
                if prof.is_default:
                    prof.is_default = False
                    prof.updated_at = datetime.now()

            self.profiles[default_name] = default_profile
            self.active_profile = default_name # Set as active
            self._save_profiles() # Persist the new default profile


    def get_active_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the currently active profile,
        merged with environment variable overrides. Performs validation and fixing if needed.

        Returns:
            dict: The active, validated, and potentially fixed configuration dictionary.
                  Returns a fixed DEFAULT_CONFIG if no active profile is valid/found.
        """
        with self.lock:
            active_profile_instance = None
            config_to_use = None

            if self.active_profile and self.active_profile in self.profiles:
                active_profile_instance = self.profiles[self.active_profile]
                config_to_use = copy.deepcopy(active_profile_instance.config)
                source_desc = f"active profile '{self.active_profile}'"
            else:
                logger.warning(f"Active profile '{self.active_profile}' not found or none set. Using default configuration as base.")
                config_to_use = copy.deepcopy(DEFAULT_CONFIG)
                source_desc = "DEFAULT_CONFIG"

            # Validate the base configuration
            errors = ConfigValidator.validate_config(config_to_use, CONFIG_SCHEMA)
            if errors:
                logger.warning(f"Configuration from {source_desc} has validation errors: {errors}. Attempting to fix.")
                fixed_config = ConfigValidator.fix_config(config_to_use, CONFIG_SCHEMA)
                # Re-validate after fixing
                fixed_errors = ConfigValidator.validate_config(fixed_config, CONFIG_SCHEMA)
                if fixed_errors:
                    logger.error(f"Fixing configuration from {source_desc} failed. Resulting config still has errors: {fixed_errors}. Proceeding with potentially invalid config.")
                    config_to_use = fixed_config # Use the (still broken) fixed version
                else:
                    logger.info(f"Configuration from {source_desc} fixed successfully.")
                    config_to_use = fixed_config
                    # Optionally update the profile's stored config if it was fixed?
                    # if active_profile_instance:
                    #    active_profile_instance.config = config_to_use
                    #    active_profile_instance.updated_at = datetime.now()
                    #    self._save_profiles() # Save the fix

            # Apply environment overrides to the (potentially fixed) config
            try:
                overridden_config = self._apply_env_overrides(config_to_use)
            except Exception as e:
                logger.error(f"Failed to apply environment overrides: {e}. Using config without overrides.", exc_info=True)
                overridden_config = config_to_use # Fallback to pre-override config

            # Final validation of the overridden config (optional, might impact performance)
            # final_errors = ConfigValidator.validate_config(overridden_config, CONFIG_SCHEMA)
            # if final_errors:
            #    logger.warning(f"Final active configuration (with overrides) has validation errors: {final_errors}")

            return overridden_config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to a configuration dictionary.
        Environment variables like APP_SEARCH_ENGINE__PROVIDERS__COHERE__MODEL=new_model
        will override config['search_engine']['providers']['cohere']['model'].
        Uses double underscore `__` as separator. Case-insensitive matching for config keys.

        Args:
            config: The configuration dictionary to override.

        Returns:
            dict: A new dictionary with overrides applied.
        """
        # Deep copy to avoid modifying the original in-memory config object
        result = copy.deepcopy(config)

        # Iterate through environment variables that start with the prefix
        for env_key, value_str in os.environ.items():
            if env_key.startswith(self.env_prefix):
                # Remove prefix and split by double underscore
                config_path_str = env_key[len(self.env_prefix):]
                parts = config_path_str.split("__")

                if not parts: continue # Skip if env var is just the prefix

                # Navigate through the config dict using lowercase keys for matching
                current_level = result
                path_traversed = []
                valid_path = True
                try:
                    for i, part in enumerate(parts):
                        # Match case-insensitively against existing keys at the current level
                        # Assumes config dict uses consistent (e.g., lowercase) keys internally,
                        # or we find the matching key regardless of env var case.
                        matched_key = None
                        part_lower = part.lower()
                        if isinstance(current_level, dict):
                             for existing_key in current_level:
                                 if existing_key.lower() == part_lower:
                                     matched_key = existing_key
                                     break

                        path_traversed.append(matched_key if matched_key else part) # Use matched key if found

                        if i == len(parts) - 1:
                            # Last part: set the parsed value
                            if matched_key is not None and isinstance(current_level, dict):
                                parsed_value = parse_value(value_str)
                                current_level[matched_key] = parsed_value
                                logger.debug(f"Overrode config '{'.'.join(path_traversed)}' with value '{parsed_value}' from env '{env_key}'.")
                            else:
                                # Path doesn't fully exist or last part isn't in dict
                                logger.warning(f"Cannot apply env override '{env_key}': Path '{'.'.join(path_traversed)}' is invalid or parent is not a dictionary.")
                                valid_path = False
                        else:
                            # Navigate deeper
                            if matched_key is not None and isinstance(current_level.get(matched_key), dict):
                                current_level = current_level[matched_key]
                            else:
                                # Path doesn't exist or isn't a dict - cannot override cleanly.
                                logger.warning(f"Cannot apply env override '{env_key}': Path component '{part}' (-> '{matched_key}') not found or not a dictionary at '{'.'.join(path_traversed)}'.")
                                valid_path = False

                        if not valid_path:
                            break # Stop processing this env var if path becomes invalid

                except Exception as e:
                    logger.error(f"Error applying environment override for '{env_key}' at path '{'.'.join(path_traversed)}': {e}", exc_info=True)

        return result

    def get_profile(self, name: str) -> Optional[ConfigProfile]:
        """
        Get a profile instance by its name. Returns a copy to prevent external modification.
        Args:
            name: The name of the profile.
        Returns:
            Optional[ConfigProfile]: A deep copy of the profile instance, or None if not found.
        """
        with self.lock:
            profile = self.profiles.get(name)
            return copy.deepcopy(profile) if profile else None

    def set_active_profile(self, name: str) -> bool:
        """
        Set the currently active profile.
        Args:
            name: The name of the profile to activate.
        Returns:
            bool: True if the profile was found and activated, False otherwise.
        """
        with self.lock:
            if name not in self.profiles:
                logger.warning(f"Cannot set active profile: Profile '{name}' not found.")
                return False

            if self.active_profile != name:
                self.active_profile = name
                logger.info(f"Set active profile to '{name}'.")
                self._save_profiles() # Persist the change
            else:
                logger.debug(f"Profile '{name}' is already active.")
            return True

    def create_profile(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
        set_active: bool = False,
        source: Optional[str] = None # Optional source info for logging
    ) -> bool:
        """
        Create a new configuration profile. Validates and fixes the config if needed.

        Args:
            name: The unique name for the new profile. Must be non-empty.
            description: Optional description.
            config: Configuration dictionary for the profile. If None, uses DEFAULT_CONFIG.
            is_default: If True, this profile becomes the new default profile.
            set_active: If True, this profile becomes the active profile after creation.
            source: Optional string indicating the source (e.g., 'import', 'clone', 'api').

        Returns:
            bool: True if the profile was created successfully, False otherwise.
        """
        if not name or not isinstance(name, str):
             logger.error("Cannot create profile: Profile name must be a non-empty string.")
             return False

        with self.lock:
            if name in self.profiles:
                logger.warning(f"Cannot create profile: Name '{name}' already exists.")
                return False

            log_prefix = f"Profile '{name}'"
            if source: log_prefix += f" (from {source})"

            # Use provided config or a deep copy of DEFAULT_CONFIG
            profile_config = copy.deepcopy(config) if config is not None else copy.deepcopy(DEFAULT_CONFIG)

            # Create a temporary profile instance for validation/fixing
            temp_profile = ConfigProfile(name=name, config=profile_config)

            # Validate the configuration
            errors = temp_profile.validate()
            if errors:
                logger.warning(f"{log_prefix}: Initial configuration has validation errors: {errors}")
                logger.info(f"Attempting to fix configuration for {log_prefix}...")
                fixed_profile = temp_profile.fix() # fix() returns a new instance

                # Re-validate after fixing
                fixed_errors = fixed_profile.validate()
                if fixed_errors:
                    logger.error(f"{log_prefix}: Fixing configuration failed. Resulting config still has errors: {fixed_errors}. Profile not created.")
                    return False
                else:
                    logger.info(f"{log_prefix}: Configuration fixed successfully.")
                    # Use the config from the fixed profile instance
                    profile_config = fixed_profile.config
            else:
                 logger.debug(f"{log_prefix}: Initial configuration validated successfully.")

            # Create the final profile instance with validated/fixed config
            new_profile = ConfigProfile(
                name=name,
                description=description,
                config=profile_config, # Use the validated/fixed config
                is_default=is_default # Initial default state
            )

            # Handle 'is_default' flag logic
            if new_profile.is_default:
                # Unset the default flag on all other existing profiles
                for prof_name, prof in self.profiles.items():
                    if prof.is_default: # Check name just in case self was somehow in list
                        if prof_name != new_profile.name:
                             prof.is_default = False
                             prof.updated_at = datetime.now() # Mark as updated
                             logger.info(f"Unset default flag for existing profile '{prof_name}'.")
            elif not self.profiles:
                 # If this is the very first profile being added, make it default
                 logger.info(f"{log_prefix}: First profile created, marking as default.")
                 new_profile.is_default = True


            # Add the new profile
            self.profiles[name] = new_profile

            # Set as active if requested OR if it's the only/new default and no active profile is set
            should_activate = set_active or (new_profile.is_default and (self.active_profile is None or self.active_profile not in self.profiles))
            if should_activate:
                self.active_profile = name
                logger.info(f"{log_prefix} created and set as active.")
            else:
                logger.info(f"{log_prefix} created.")

            # Save changes
            self._save_profiles()
            return True

    def update_profile(
        self,
        name: str,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        is_default: Optional[bool] = None
    ) -> bool:
        """
        Update an existing configuration profile. Only provided fields are updated.
        If config is provided, it *replaces* the existing config after validation/fixing.

        Args:
            name: The name of the profile to update.
            description: New description (if provided).
            config: New configuration dictionary (if provided, replaces the old one).
            is_default: New default status (if provided).

        Returns:
            bool: True if the profile was found and updated, False otherwise.
        """
        with self.lock:
            if name not in self.profiles:
                logger.warning(f"Cannot update profile: Profile '{name}' not found.")
                return False

            profile_to_update = self.profiles[name]
            updated = False
            log_prefix = f"Profile '{name}'"

            # Update description
            if description is not None and profile_to_update.description != description:
                profile_to_update.description = description
                logger.debug(f"{log_prefix}: Description updated.")
                updated = True

            # Update config (replace entirely if provided, after validation/fixing)
            if config is not None:
                logger.debug(f"{log_prefix}: Attempting to update config.")
                # Validate and fix the new config before applying
                temp_profile = ConfigProfile(name=name, config=config) # Temp profile for validation
                errors = temp_profile.validate()
                new_config_to_apply = None
                if errors:
                    logger.warning(f"{log_prefix}: New configuration has validation errors: {errors}")
                    logger.info(f"Attempting to fix new configuration for {log_prefix}...")
                    fixed_profile = temp_profile.fix()
                    fixed_errors = fixed_profile.validate()
                    if fixed_errors:
                        logger.error(f"{log_prefix}: Fixing new configuration failed. Config not updated. Errors: {fixed_errors}")
                        # Optionally return False here if config update must succeed
                    else:
                        logger.info(f"{log_prefix}: New configuration fixed successfully.")
                        new_config_to_apply = fixed_profile.config # Use fixed config
                else:
                    logger.debug(f"{log_prefix}: New configuration validated successfully.")
                    new_config_to_apply = copy.deepcopy(config) # Use deep copy of valid config

                # Apply the new config if validation/fixing was successful
                if new_config_to_apply is not None:
                    # Check if the config actually changed compared to the current one
                    if profile_to_update.config != new_config_to_apply:
                        profile_to_update.config = new_config_to_apply
                        logger.debug(f"{log_prefix}: Configuration updated.")
                        updated = True
                    else:
                        logger.debug(f"{log_prefix}: Provided configuration is identical to the current one. No update needed.")

            # Update default status
            if is_default is not None and profile_to_update.is_default != is_default:
                logger.debug(f"{log_prefix}: Attempting to set default status to {is_default}.")
                if is_default: # Setting this one as the new default
                    # Unset default flag on others
                    profile_to_update.is_default = True # Set first to ensure it's true even if it was the only one
                    for prof_name, prof in self.profiles.items():
                        if prof.is_default and prof_name != name:
                            prof.is_default = False
                            prof.updated_at = datetime.now()
                            logger.info(f"Unset default flag for profile '{prof_name}'.")
                    updated = True
                    # If setting as default, ensure active profile is valid (might activate this one)
                    if self.active_profile is None or self.active_profile not in self.profiles:
                        self.active_profile = name
                        logger.info(f"{log_prefix} set as default and activated as no valid profile was active.")
                    else:
                        logger.info(f"{log_prefix} set as the default profile.")
                else: # Attempting to unset default for this profile
                    # Ensure at least one default profile remains if possible
                    current_defaults = [n for n, p in self.profiles.items() if p.is_default]
                    if len(current_defaults) <= 1 and name in current_defaults:
                        logger.warning(f"Cannot unset default flag for {log_prefix}: It is the only default profile.")
                        # Do not change is_default, but maybe still mark as updated? No, only if changed.
                    else:
                        profile_to_update.is_default = False
                        logger.info(f"Unset default flag for {log_prefix}.")
                        updated = True

            # Update timestamp and save if any changes were made
            if updated:
                profile_to_update.updated_at = datetime.now()
                logger.info(f"{log_prefix} updated successfully.")
                self._save_profiles() # Persist changes
            else:
                 logger.info(f"{log_prefix}: No changes detected during update request.")

            return updated

    def delete_profile(self, name: str) -> bool:
        """
        Delete a configuration profile. Prevents deleting the last profile.
        Handles resetting active/default status if the deleted profile held them.

        Args:
            name: The name of the profile to delete.

        Returns:
            bool: True if the profile was found and deleted, False otherwise.
        """
        with self.lock:
            if name not in self.profiles:
                logger.warning(f"Cannot delete profile: Profile '{name}' not found.")
                return False

            # Prevent deleting the last profile
            if len(self.profiles) <= 1:
                logger.warning(f"Cannot delete profile '{name}': It is the only profile remaining.")
                return False

            profile_to_delete = self.profiles[name]
            was_active = (name == self.active_profile)
            was_default = profile_to_delete.is_default

            # Delete the profile
            del self.profiles[name]
            logger.info(f"Deleted profile '{name}'.")

            # If the deleted profile was active, set a new active profile
            if was_active:
                self.active_profile = None # Clear first
                self._ensure_default_profile() # This logic finds default or first and sets active
                logger.info(f"Deleted profile '{name}' was active. New active profile is '{self.active_profile}'.")

            # If the deleted profile was the default, ensure a new default exists
            if was_default:
                # Check if another default already exists
                existing_defaults = [n for n, p in self.profiles.items() if p.is_default]
                if not existing_defaults and self.profiles:
                    # If no default remains, make the current active profile the default
                    if self.active_profile and self.active_profile in self.profiles:
                        self.profiles[self.active_profile].is_default = True
                        self.profiles[self.active_profile].updated_at = datetime.now()
                        logger.info(f"Deleted profile '{name}' was default. Set current active profile '{self.active_profile}' as the new default.")
                    else:
                         # Should not happen if active profile was reset correctly, but fallback
                         first_profile_name = sorted(self.profiles.keys())[0]
                         self.profiles[first_profile_name].is_default = True
                         self.profiles[first_profile_name].updated_at = datetime.now()
                         logger.info(f"Deleted profile '{name}' was default. Set first profile '{first_profile_name}' as the new default.")

            # Save changes resulting from deletion and potential active/default updates
            self._save_profiles()
            return True

    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List summary information for all available profiles, sorted by name.
        Returns:
            list: A list of dictionaries, each containing summary info for a profile.
        """
        with self.lock:
            profile_list = []
            # Sort by name for consistent ordering
            for name in sorted(self.profiles.keys()):
                profile = self.profiles[name]
                profile_list.append({
                    "name": name,
                    "description": profile.description,
                    "is_default": profile.is_default,
                    "is_active": name == self.active_profile,
                    "created_at": profile.created_at.isoformat() if isinstance(profile.created_at, datetime) else str(profile.created_at),
                    "updated_at": profile.updated_at.isoformat() if isinstance(profile.updated_at, datetime) else str(profile.updated_at)
                    # Optionally add validation status?
                    # "is_valid": not bool(profile.validate())
                })
            return profile_list

    def clone_profile(
        self,
        source_name: str,
        target_name: str,
        description: Optional[str] = None,
        is_default: bool = False,
        set_active: bool = False
    ) -> bool:
        """
        Create a new profile by cloning an existing one's configuration.

        Args:
            source_name: The name of the profile to clone.
            target_name: The name for the new cloned profile. Must be non-empty.
            description: Optional description for the new profile. Uses "Clone of..." if None.
            is_default: If True, the new cloned profile becomes the default.
            set_active: If True, the new cloned profile becomes active.

        Returns:
            bool: True if cloned successfully, False otherwise (e.g., source not found, target exists).
        """
        if not target_name or not isinstance(target_name, str):
             logger.error("Cannot clone profile: Target profile name must be a non-empty string.")
             return False

        with self.lock:
            if source_name not in self.profiles:
                logger.warning(f"Cannot clone profile: Source '{source_name}' not found.")
                return False
            if target_name in self.profiles:
                logger.warning(f"Cannot clone profile: Target name '{target_name}' already exists.")
                return False

            source_profile = self.profiles[source_name]
            clone_desc = description if description is not None else f"Clone of '{source_name}'"

            # Use create_profile to handle validation, fixing (unlikely needed for clone), default/active flags, and saving
            return self.create_profile(
                name=target_name,
                description=clone_desc,
                config=copy.deepcopy(source_profile.config), # Deep copy config from source
                is_default=is_default,
                set_active=set_active,
                source=f"clone of '{source_name}'" # Add source info
            )


    def import_profile(
        self,
        file_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_default: bool = False,
        set_active: bool = False
    ) -> bool:
        """
        Import a profile configuration from a JSON or YAML file.
        Validates and fixes the imported configuration.

        Args:
            file_path: Path to the configuration file (.json, .yaml, .yml).
            name: Name for the imported profile. If None, derived from the filename. Must be non-empty.
            description: Optional description. Uses "Imported from..." if None.
            is_default: If True, the imported profile becomes the default.
            set_active: If True, the imported profile becomes active.

        Returns:
            bool: True if imported successfully, False otherwise.
        """
        import_path = Path(file_path)
        if not import_path.is_file():
            logger.warning(f"Cannot import profile: File not found at '{file_path}'.")
            return False

        # Determine profile name if not provided
        profile_name = name or import_path.stem # Use filename without extension
        if not profile_name or not isinstance(profile_name, str):
             logger.error(f"Cannot import profile: Invalid profile name '{profile_name}' derived from filename or provided.")
             return False

        # Check for name conflict before reading file
        with self.lock:
            if profile_name in self.profiles:
                logger.warning(f"Cannot import profile: Name '{profile_name}' already exists.")
                return False

        try:
            # Load configuration from file
            config_data = None
            ext = import_path.suffix.lower()

            if ext == ".json":
                with import_path.open("r", encoding='utf-8') as f:
                    config_data = json.load(f)
            elif ext in [".yaml", ".yml"]:
                try:
                    import yaml # Local import as PyYAML might be optional
                    with import_path.open("r", encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                except ImportError:
                    logger.error("Cannot import YAML profile: PyYAML library is not installed. Please install it (`pip install pyyaml`).")
                    return False
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML file '{file_path}': {e}")
                    return False
            else:
                logger.warning(f"Cannot import profile: Unsupported file extension '{ext}'. Use .json, .yaml, or .yml.")
                return False

            # Ensure loaded data is a dictionary
            if not isinstance(config_data, dict):
                logger.error(f"Import failed: Configuration file '{file_path}' does not contain a valid dictionary structure.")
                return False

            # Use create_profile logic to add the imported profile (handles validation, fixing, etc.)
            import_desc = description if description is not None else f"Imported from '{import_path.name}'"
            return self.create_profile(
                name=profile_name,
                description=import_desc,
                config=config_data,
                is_default=is_default,
                set_active=set_active,
                source=f"import from '{import_path.name}'"
            )

        except (IOError, json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to read or process import file '{file_path}': {e}", exc_info=True)
            return False


    def export_profile(self, name: str, file_path: str, format: str = "json") -> bool:
        """
        Export a profile's configuration to a JSON or YAML file.

        Args:
            name: The name of the profile to export.
            file_path: The path where the file should be saved. Directory will be created if needed.
            format: The output format ('json', 'yaml', or 'yml'). Defaults to 'json'.

        Returns:
            bool: True if exported successfully, False otherwise.
        """
        export_path = Path(file_path)
        export_format = format.lower()

        if export_format not in ["json", "yaml", "yml"]:
             logger.warning(f"Cannot export profile: Unsupported format '{format}'. Use 'json', 'yaml', or 'yml'.")
             return False

        with self.lock:
            if name not in self.profiles:
                logger.warning(f"Cannot export profile: Profile '{name}' not found.")
                return False
            profile_to_export = self.profiles[name]

            try:
                # Ensure the target directory exists
                export_path.parent.mkdir(parents=True, exist_ok=True)

                # Export the configuration
                if export_format == "json":
                    with export_path.open("w", encoding='utf-8') as f:
                        json.dump(profile_to_export.config, f, indent=2, ensure_ascii=False)
                else: # yaml or yml
                    try:
                        import yaml # Local import
                        with export_path.open("w", encoding='utf-8') as f:
                             yaml.dump(profile_to_export.config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                    except ImportError:
                        logger.error("Cannot export YAML profile: PyYAML library is not installed. Please install it (`pip install pyyaml`).")
                        return False
                    except yaml.YAMLError as e:
                        logger.error(f"Error writing YAML file '{file_path}': {e}")
                        return False

                logger.info(f"Exported profile '{name}' to '{file_path}' in {export_format} format.")
                return True

            except (IOError, TypeError, OSError, Exception) as e:
                logger.error(f"Failed to export profile '{name}' to '{file_path}': {e}", exc_info=True)
                return False


    def merge_profiles(
        self,
        source_names: List[str],
        target_name: str,
        description: Optional[str] = None,
        is_default: bool = False,
        set_active: bool = False
    ) -> bool:
        """
        Merge configurations from multiple source profiles into a new target profile.
        Profiles are merged in the order they appear in source_names, with later profiles
        overwriting values from earlier ones.

        Args:
            source_names: List of names of the profiles to merge. Must contain at least one name.
            target_name: Name for the new merged profile. Must be non-empty.
            description: Optional description for the merged profile.
            is_default: If True, the new merged profile becomes the default.
            set_active: If True, the new merged profile becomes active.

        Returns:
            bool: True if merged successfully, False otherwise.
        """
        if not source_names or not isinstance(source_names, list):
            logger.warning("Cannot merge profiles: Source profile names must be provided as a non-empty list.")
            return False
        if not target_name or not isinstance(target_name, str):
            logger.error("Cannot merge profiles: Target profile name must be a non-empty string.")
            return False

        with self.lock:
            # Validate source profiles exist
            missing_sources = [name for name in source_names if name not in self.profiles]
            if missing_sources:
                logger.warning(f"Cannot merge profiles: Source profiles not found: {', '.join(missing_sources)}")
                return False

            # Check if target name already exists
            if target_name in self.profiles:
                logger.warning(f"Cannot merge profiles: Target name '{target_name}' already exists.")
                return False

            # Perform the merge starting with an empty dict
            merged_config = {}
            logger.debug(f"Merging profiles in order: {source_names}")
            for name in source_names:
                source_profile = self.profiles[name]
                # merge_configs performs deep merge, source_profile.config overrides merged_config
                merged_config = merge_configs(merged_config, source_profile.config)

            # Use create_profile logic to add the merged profile
            merge_desc = description if description is not None else f"Merged from {', '.join(source_names)}"
            return self.create_profile(
                name=target_name,
                description=merge_desc,
                config=merged_config,
                is_default=is_default,
                set_active=set_active,
                source=f"merge of {source_names}"
            )

    def reset_profile(self, name: str) -> bool:
        """
        Reset a profile's configuration back to the application's DEFAULT_CONFIG.
        The profile's description, timestamps, and default/active status are preserved
        unless modified by side effects (e.g., becoming the only default).

        Args:
            name: The name of the profile to reset.

        Returns:
            bool: True if the profile was found and reset, False otherwise.
        """
        with self.lock:
            if name not in self.profiles:
                logger.warning(f"Cannot reset profile: Profile '{name}' not found.")
                return False

            profile_to_reset = self.profiles[name]
            original_config = profile_to_reset.config

            # Reset config to a deep copy of DEFAULT_CONFIG
            new_config = copy.deepcopy(DEFAULT_CONFIG)

            # Check if config actually changed
            if original_config == new_config:
                 logger.info(f"Profile '{name}' configuration is already identical to default. No reset needed.")
                 return True # Report success as it's already in the desired state

            profile_to_reset.config = new_config
            profile_to_reset.updated_at = datetime.now()

            # Validate the reset config (should pass if DEFAULT_CONFIG and schema are valid)
            errors = profile_to_reset.validate()
            if errors:
                logger.error(f"Resetting profile '{name}' resulted in validation errors (DEFAULT_CONFIG might be invalid): {errors}")
                # Decide whether to keep the potentially invalid state or revert/fail
                # Reverting might be safer:
                # profile_to_reset.config = original_config # Revert
                # profile_to_reset.updated_at = datetime.now() # Revert timestamp? Maybe keep update time.
                # logger.error(f"Reverted profile '{name}' reset due to validation errors.")
                # self._save_profiles()
                # return False
                # For now, keep the reset state but log error prominently.
            else:
                logger.info(f"Profile '{name}' configuration reset to default values.")

            # Save changes
            self._save_profiles()
            return True


    # --- NEW METHODS FOR SPECIFIC SETTING UPDATE ---

    def _get_nested_value(self, config_dict: Dict[str, Any], key_path: str) -> Tuple[bool, Any]:
        """Helper to get a value from a nested dict using dot notation.
           Returns (success_flag, value or None)."""
        keys = key_path.split('.')
        value = config_dict
        try:
            for key in keys:
                if isinstance(value, dict):
                    # Attempt case-insensitive match first for robustness
                    matched_key = None
                    key_lower = key.lower()
                    for existing_key in value:
                         if existing_key.lower() == key_lower:
                              matched_key = existing_key
                              break
                    if matched_key:
                         value = value[matched_key]
                    else:
                         # Fallback to exact match if case-insensitive failed
                         value = value[key]
                else:
                    # Path leads through a non-dictionary element before reaching the end
                    logger.warning(f"_get_nested_value: Invalid path component '{key}' in '{key_path}'. Parent is not a dictionary.")
                    return False, None
            return True, value
        except KeyError:
            logger.debug(f"_get_nested_value: Key path '{key_path}' not found in config.")
            return False, None
        except Exception as e:
            logger.error(f"Error getting nested value for '{key_path}': {e}")
            return False, None

    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: Any) -> bool:
        """Helper to set a value in a nested dict using dot notation.
           Attempts case-insensitive matching for robustness. Modifies dict in place."""
        keys = key_path.split('.')
        current_level = config_dict
        try:
            for i, key in enumerate(keys):
                # Attempt case-insensitive match for existing keys
                matched_key = None
                key_lower = key.lower()
                if isinstance(current_level, dict):
                    for existing_key in current_level:
                         if existing_key.lower() == key_lower:
                              matched_key = existing_key
                              break
                else:
                     # Trying to traverse through a non-dict element
                     logger.error(f"Cannot set value at '{key_path}': Path component '{key}' encounters non-dictionary parent.")
                     return False

                actual_key = matched_key if matched_key else key # Use matched key or original if not found/creating new

                if i == len(keys) - 1:
                    # Last key: set the value
                    if isinstance(current_level, dict):
                        current_level[actual_key] = value
                        logger.debug(f"_set_nested_value: Set '{key_path}' to {repr(value)} using key '{actual_key}'")
                        return True
                    else:
                        # Should have been caught earlier, but safety check
                        logger.error(f"Cannot set value at '{key_path}': Final path component '{actual_key}' parent is not a dictionary.")
                        return False
                else:
                    # Navigate deeper or create dict if path doesn't exist
                    if actual_key not in current_level or not isinstance(current_level.get(actual_key), dict):
                        # Path doesn't exist or isn't a dict
                        # Option 1: Fail if intermediate path doesn't exist (safer)
                        logger.error(f"Cannot set value at '{key_path}': Path component '{actual_key}' not found or not a dictionary.")
                        return False
                        # Option 2: Create intermediate dicts (use with caution)
                        # logger.debug(f"_set_nested_value: Creating intermediate dictionary for key '{actual_key}'")
                        # current_level[actual_key] = {}

                    current_level = current_level[actual_key]

            return False # Should not be reached if keys is not empty
        except Exception as e:
            logger.error(f"Error setting nested value for '{key_path}': {e}", exc_info=True)
            return False

    def _get_schema_for_path(self, schema: Dict[str, Any], key_path: str) -> Optional[Dict[str, Any]]:
        """Helper to get the schema definition (dict containing 'type') for a specific key path."""
        keys = key_path.split('.')
        current_schema = schema
        try:
            for i, key in enumerate(keys):
                 if not isinstance(current_schema, dict):
                      # Path traverses through a non-dict part of the schema definition itself
                      logger.warning(f"_get_schema_for_path: Invalid schema structure at component '{key}' for path '{key_path}'.")
                      return None

                 # Case-insensitive matching for schema keys
                 matched_key = None
                 key_lower = key.lower()
                 for schema_key in current_schema:
                      if schema_key.lower() == key_lower:
                           matched_key = schema_key
                           break

                 if matched_key:
                      current_schema = current_schema[matched_key]
                 else:
                      # Key not found in schema at this level
                      logger.warning(f"_get_schema_for_path: Path component '{key}' not found in schema for path '{key_path}'.")
                      return None

            # If the final element is a dict containing 'type', it's the field schema we want
            if isinstance(current_schema, dict) and "type" in current_schema:
                return current_schema
            else:
                 # Path points to a nested structure container or invalid schema entry
                 logger.warning(f"_get_schema_for_path: Path '{key_path}' does not resolve to a field schema with a 'type'. Found: {type(current_schema)}")
                 return None
        except Exception as e:
            logger.error(f"Error getting schema for path '{key_path}': {e}", exc_info=True)
            return None


    def update_active_profile_setting(self, key_path: str, new_value: Any) -> bool:
        """
        Update a specific setting in the *active* profile using dot notation (e.g., "search_engine.use_cache").
        Validates the new value against the schema and persists the change.

        Args:
            key_path (str): The dot-separated path to the setting. Case-insensitive matching for path components.
            new_value (Any): The new value for the setting.

        Returns:
            bool: True if the setting was successfully validated, updated, and saved, False otherwise.
        """
        with self.lock:
            if not self.active_profile:
                logger.error("Cannot update setting: No active profile set.")
                return False

            active_profile_instance = self.profiles.get(self.active_profile)
            if not active_profile_instance:
                logger.error(f"Cannot update setting: Active profile '{self.active_profile}' not found.")
                # Attempt to recover? For now, fail.
                self._ensure_default_profile() # Try to ensure active profile is valid next time
                return False

            log_prefix = f"Profile '{self.active_profile}', Setting '{key_path}'"

            # 1. Get the specific schema definition for the key_path
            field_schema = self._get_schema_for_path(CONFIG_SCHEMA, key_path)
            if not field_schema:
                # Error logged within _get_schema_for_path
                logger.error(f"{log_prefix}: Cannot update setting - schema definition not found for this path.")
                return False

            # 2. Validate the new_value against the field's schema *before* applying
            validation_errors = ConfigValidator._validate_type(new_value, field_schema, key_path)
            if validation_errors:
                logger.error(f"{log_prefix}: New value {repr(new_value)} failed validation: {validation_errors}")
                # Optionally, attempt to fix the value?
                logger.info(f"Attempting to fix invalid value for {log_prefix}...")
                fixed_value = ConfigValidator._fix_type(new_value, field_schema, key_path)
                fixed_errors = ConfigValidator._validate_type(fixed_value, field_schema, key_path)
                if fixed_errors:
                     logger.error(f"{log_prefix}: Fixing the value also failed validation ({fixed_errors}). Update rejected.")
                     return False
                else:
                     logger.info(f"{log_prefix}: Successfully fixed value to {repr(fixed_value)}. Proceeding with fixed value.")
                     new_value = fixed_value # Use the fixed value

            # 3. Check if the value actually changed
            success, current_value = self._get_nested_value(active_profile_instance.config, key_path)
            if success and current_value == new_value:
                 logger.info(f"{log_prefix}: New value is identical to the current value ({repr(new_value)}). No update performed.")
                 return True # Report success as no change needed

            # 4. Create a deep copy of the config to potentially modify
            #    This avoids partial updates if setting fails midway.
            temp_config = copy.deepcopy(active_profile_instance.config)

            # 5. Set the nested value in the temporary config copy
            if not self._set_nested_value(temp_config, key_path, new_value):
                # _set_nested_value logs the specific error
                logger.error(f"{log_prefix}: Failed to set new value in configuration structure.")
                return False

            # 6. Optional: Full validation after change (can be slow, maybe skip for single field update)
            # full_errors = ConfigValidator.validate_config(temp_config, CONFIG_SCHEMA)
            # if full_errors:
            #    logger.error(f"{log_prefix}: Updating '{key_path}' resulted in overall config validation errors: {full_errors}. Reverting change.")
            #    return False

            # 7. Apply the change to the actual profile config
            active_profile_instance.config = temp_config # Replace with the updated copy
            active_profile_instance.updated_at = datetime.now()

            # 8. Save the updated profiles
            self._save_profiles()
            logger.info(f"{log_prefix}: Successfully updated setting to {repr(new_value)}.")
            return True


# --- Helper Functions (defined outside classes) ---

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries. Values from override_config take precedence.
    Creates a new dictionary, does not modify inputs. List merging replaces lists.

    Args:
        base_config: The base dictionary.
        override_config: The dictionary with overriding values.

    Returns:
        dict: The merged dictionary.
    """
    merged = copy.deepcopy(base_config)
    if not isinstance(override_config, dict):
         # If override is not a dict, the behavior depends:
         # Option 1: Ignore override (if base is dict).
         # Option 2: Replace base with override (current behavior if base wasn't dict).
         # Let's stick to merging dicts primarily. If override isn't dict, log warning?
         if isinstance(base_config, dict):
              logger.warning(f"Attempted to merge non-dict override ({type(override_config).__name__}) onto dict base. Override ignored.")
              return merged
         else:
              # If base isn't dict either, override completely wins.
              return copy.deepcopy(override_config)


    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged.get(key), dict):
            # If both values are dicts, recurse
            merged[key] = merge_configs(merged[key], value)
        # elif isinstance(value, list) and key in merged and isinstance(merged.get(key), list):
            # List merging strategy: replace, append, or element-wise merge?
            # Simplest and most common: override list replaces base list.
            # merged[key] = copy.deepcopy(value) # Replace list
        else:
            # Otherwise, override value (use deepcopy for nested structures like lists/dicts)
            merged[key] = copy.deepcopy(value)
    return merged

def parse_value(value_str: str) -> Any:
    """
    Attempt to parse a string value from environment variables into Python types
    (JSON literals: true, false, null, numbers, strings; fallback: original string).
    More robust boolean handling.

    Args:
        value_str: The input string.

    Returns:
        Any: The parsed value or the original string.
    """
    # Handle potential None or empty strings explicitly
    if value_str is None:
        return None
    if value_str == "":
        return ""

    stripped_val = value_str.strip()
    lower_value = stripped_val.lower()

    # Explicit Boolean Check (covers 'true'/'false' and common variations)
    if lower_value in ["true", "yes", "on", "1", "y", "t"]:
        return True
    if lower_value in ["false", "no", "off", "0", "n", "f"]:
        return False

    # Explicit Null check
    if lower_value == "null":
         return None

    # Try parsing as JSON (handles numbers, JSON strings "...", lists [...], dicts {...})
    try:
        # Use loads for direct number/JSON string parsing
        json_val = json.loads(stripped_val)
        # If json.loads returns a string, it was a quoted JSON string. Return it.
        # If it returns number, bool, list, dict, null, return that parsed value.
        return json_val
    except json.JSONDecodeError:
        # Not valid JSON literal, proceed
        pass
    except Exception as e:
         # Catch other potential errors during json.loads
         logger.warning(f"Unexpected error parsing value '{value_str}' as JSON: {e}")


    # If it wasn't parsed as bool, null, or JSON literal, return the original string
    return value_str


# --- Singleton Accessor ---
# Use a lock for thread safety around singleton creation

_config_manager_singleton_lock = threading.Lock()
_config_manager_instance: Optional[ConfigManager] = None

def get_config_manager(
    config_dir: Optional[str] = None, # Allow None to use default from first call
    profiles_file: Optional[str] = None,
    env_prefix: Optional[str] = None
) -> ConfigManager:
    """
    Get THE singleton instance of the ConfigManager.
    Ensures only one manager handles the configuration files.
    Initializes with specified parameters ONLY on the very first call.
    Subsequent calls ignore parameters and return the existing instance.

    Args:
        config_dir (str, optional): Directory for configuration files. Used only on first creation. Defaults to ".config".
        profiles_file (str, optional): Filename for profiles file. Used only on first creation. Defaults to "profiles.json".
        env_prefix (str, optional): Prefix for environment variables. Used only on first creation. Defaults to "APP_".

    Returns:
        ConfigManager: The singleton ConfigManager instance.
    """
    global _config_manager_instance
    # Quick check without lock first
    if _config_manager_instance is not None:
        return _config_manager_instance

    # If instance is None, acquire lock for creation
    with _config_manager_singleton_lock:
        # Double-check inside lock (important!)
        if _config_manager_instance is None:
            logger.info("Creating ConfigManager singleton instance.")
            # Use provided args or defaults for initialization
            effective_config_dir = config_dir if config_dir is not None else ".config"
            effective_profiles_file = profiles_file if profiles_file is not None else "profiles.json"
            effective_env_prefix = env_prefix if env_prefix is not None else "APP_"

            _config_manager_instance = ConfigManager(
                config_dir=effective_config_dir,
                profiles_file=effective_profiles_file,
                env_prefix=effective_env_prefix
            )
        # else: # Instance was created by another thread while waiting for lock
              # logger.debug("ConfigManager singleton already created by another thread.")

    return _config_manager_instance


# --- Convenience Function ---

def get_config() -> Dict[str, Any]:
    """
    Convenience function to get the active, validated configuration
    from the singleton ConfigManager instance.

    Returns:
        dict: The currently active configuration dictionary.
    """
    manager = get_config_manager() # Gets singleton instance
    return manager.get_active_config()

# Example Usage (for testing if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Enable debug logging for testing
    logger.info("Testing ConfigManager...")

    # Get the singleton instance (creates if not exists)
    cm = get_config_manager(config_dir=".test_config")

    # List initial profiles
    print("Initial Profiles:", cm.list_profiles())
    print("Initial Active Config:", json.dumps(cm.get_active_config(), indent=2))

    # Test updating a setting
    print("\nUpdating search_engine.use_cache to False...")
    success = cm.update_active_profile_setting("search_engine.use_cache", False)
    print(f"Update successful: {success}")
    print("Config after update:", json.dumps(cm.get_active_config()['search_engine'], indent=2))

    print("\nUpdating appearance.theme to 'light'...")
    success = cm.update_active_profile_setting("appearance.theme", "light")
    print(f"Update successful: {success}")
    print("Config after update:", json.dumps(cm.get_active_config()['appearance'], indent=2))

    print("\nTrying to update theme to invalid value 'rainbow' (should fail or be fixed)...")
    success = cm.update_active_profile_setting("appearance.theme", "rainbow")
    print(f"Update successful: {success}") # Might be True if fixed, False if rejected
    print("Config after invalid update attempt:", json.dumps(cm.get_active_config()['appearance'], indent=2))

    print("\nTrying to update a nested provider setting...")
    success = cm.update_active_profile_setting("search_engine.providers.cohere.temperature", 0.99)
    print(f"Update successful: {success}")
    print("Config after nested update:", json.dumps(cm.get_active_config()['search_engine']['providers']['cohere'], indent=2))

    print("\nTrying to update a non-existent setting...")
    success = cm.update_active_profile_setting("search_engine.non_existent_setting", True)
    print(f"Update successful (should be False): {success}")

    print("\nResetting active profile to defaults...")
    active_prof_name = cm.active_profile
    if active_prof_name:
        reset_success = cm.reset_profile(active_prof_name)
        print(f"Reset successful: {reset_success}")
        print("Config after reset:", json.dumps(cm.get_active_config(), indent=2))

    # Clean up test config directory?
    # import shutil
    # shutil.rmtree(".test_config")
