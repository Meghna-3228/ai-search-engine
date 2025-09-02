"""
Streamlit Components Module for AI-powered Search

This module provides custom Streamlit components to:
1. Create an enhanced search interface with advanced features
2. Display search results with verification indicators
3. Visualize confidence metrics and analysis
4. Implement interactive configuration panels
5. Create responsive layouts for different devices

The goal is to provide a polished, user-friendly interface for the AI search application.
"""

import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Try to import local modules
try:
    from visualization import (
        create_confidence_gauge, 
        create_verification_breakdown, 
        create_source_diversity_chart,
        create_text_highlight_html
    )
except ImportError:
    # If visualization module is not available, define placeholder functions
    def create_confidence_gauge(confidence_score, theme="dark"):
        """Placeholder for visualization module function"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        return fig
    
    def create_verification_breakdown(analysis_result, theme="dark"):
        """Placeholder for visualization module function"""
        return go.Figure()
    
    def create_source_diversity_chart(analysis_result, theme="dark"):
        """Placeholder for visualization module function"""
        return go.Figure()
    
    def create_text_highlight_html(text, analysis_result, theme="dark"):
        """Placeholder for visualization module function"""
        return f"<div>{text}</div>"


class SearchInterface:
    """
    Custom search interface component for Streamlit
    """
    
    def __init__(
        self,
        search_function: Callable,
        default_query: str = "",
        placeholder: str = "What do you want to ask?",
        advanced_mode: bool = False,
        theme: str = "dark"
    ):
        """
        Initialize the search interface
        
        Args:
            search_function: Function to call for search execution
            default_query: Default query text
            placeholder: Placeholder text for the search box
            advanced_mode: Whether to show advanced search options
            theme: Color theme ('light', 'dark', or 'auto')
        """
        self.search_function = search_function
        self.default_query = default_query
        self.placeholder = placeholder
        self.advanced_mode = advanced_mode
        self.theme = theme
        
        # Initialize state
        if "search_executed" not in st.session_state:
            st.session_state.search_executed = False
        
        if "search_results" not in st.session_state:
            st.session_state.search_results = None
        
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        
        if "show_advanced" not in st.session_state:
            st.session_state.show_advanced = advanced_mode
    
    def _toggle_advanced_mode(self):
        """Toggle advanced mode state"""
        st.session_state.show_advanced = not st.session_state.show_advanced
    
    def _clear_results(self):
        """Clear search results"""
        st.session_state.search_executed = False
        st.session_state.search_results = None
    
    def _execute_search(self, query, **kwargs):
        """
        Execute search and store results in session state
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
        """
        try:
            with st.spinner("Searching..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Simulate incremental progress to improve user experience
                for i in range(10):
                    # Update progress bar
                    progress_bar.progress((i + 1) * 10)
                    time.sleep(0.05)
                
                # Execute the search
                results = self.search_function(query, **kwargs)
                
                # Store results in session state
                st.session_state.search_results = results
                st.session_state.search_executed = True
                
                # Add to history if not already there
                if query not in [h["query"] for h in st.session_state.search_history]:
                    history_entry = {
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "success": results.get("success", False)
                    }
                    st.session_state.search_history.append(history_entry)
                    
                    # Limit history size
                    if len(st.session_state.search_history) > 10:
                        st.session_state.search_history = st.session_state.search_history[-10:]
                
                # Complete the progress bar
                progress_bar.progress(100)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.search_results = {
                "error": str(e),
                "text": f"An error occurred: {str(e)}",
                "documents": [],
                "success": False
            }
            st.session_state.search_executed = True
    
    def render(self):
        """
        Render the search interface
        
        Returns:
            Streamlit component
        """
        # Main search bar
        with st.form(key="search_form"):
            # Search input
            query = st.text_input(
                "Search Query",
                value=self.default_query,
                placeholder=self.placeholder,
                key="search_query"
            )
            
            # Button row
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                search_button = st.form_submit_button(
                    "Search 🔍",
                    use_container_width=True
                )
            
            with col2:
                clear_button = st.form_submit_button(
                    "Clear",
                    use_container_width=True,
                    on_click=self._clear_results
                )
            
            with col3:
                # Advanced options toggle
                advanced_toggle = st.checkbox(
                    "Show Advanced Options",
                    value=st.session_state.show_advanced,
                    key="advanced_toggle"
                )
                
                if advanced_toggle != st.session_state.show_advanced:
                    self._toggle_advanced_mode()
            
            # Advanced options
            if st.session_state.show_advanced:
                st.divider()
                
                # Provider selection
                col1, col2 = st.columns(2)
                
                with col1:
                    provider = st.selectbox(
                        "Search Provider",
                        options=["cohere", "openai", "anthropic", "web_api"],
                        index=0,
                        key="search_provider"
                    )
                
                with col2:
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        key="confidence_threshold"
                    )
                
                # Additional options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    use_cache = st.checkbox(
                        "Use Cache",
                        value=True,
                        key="use_cache"
                    )
                
                with col2:
                    analyze_results = st.checkbox(
                        "Analyze Results",
                        value=True,
                        key="analyze_results"
                    )
                
                with col3:
                    use_fallback = st.checkbox(
                        "Use Fallback Providers",
                        value=True,
                        key="use_fallback"
                    )
        
        # Execute search if button is clicked
        if search_button and query:
            # Get advanced options
            kwargs = {}
            
            if st.session_state.show_advanced:
                kwargs["provider"] = provider
                kwargs["confidence_threshold"] = confidence_threshold
                kwargs["use_cache"] = use_cache
                kwargs["analyze_results"] = analyze_results
                kwargs["use_fallback"] = use_fallback
            
            self._execute_search(query, **kwargs)
        
        # Recent searches
        if st.session_state.search_history:
            with st.expander("Recent Searches", expanded=False):
                for entry in reversed(st.session_state.search_history):
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    time_str = timestamp.strftime("%H:%M:%S")
                    
                    # Format as button-like element
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        if st.button(
                            entry["query"],
                            key=f"history_{entry['query']}",
                            use_container_width=True
                        ):
                            # Set as current query and execute search
                            st.session_state.search_query = entry["query"]
                            self._execute_search(entry["query"])
                    
                    with col2:
                        st.caption(time_str)
        
        return st.session_state.search_executed, st.session_state.search_results


class ResultDisplay:
    """
    Custom result display component for Streamlit
    """
    
    def __init__(
        self,
        theme: str = "dark",
        show_visualizations: bool = True,
        show_sources: bool = True,
        expand_sources: bool = False
    ):
        """
        Initialize the result display
        
        Args:
            theme: Color theme ('light', 'dark', or 'auto')
            show_visualizations: Whether to show visualizations
            show_sources: Whether to show sources
            expand_sources: Whether to expand the sources section by default
        """
        self.theme = theme
        self.show_visualizations = show_visualizations
        self.show_sources = show_sources
        self.expand_sources = expand_sources
    
    def render(self, results: Dict[str, Any]):
        """
        Render the search results
        
        Args:
            results: Search results
            
        Returns:
            Streamlit component
        """
        if not results:
            st.warning("No results to display")
            return
        
        # Check for error
        if results.get("error") or not results.get("success", False):
            st.error(results.get("text", "An error occurred while searching"))
            return
        
        # Top metrics
        self._render_metrics(results)
        
        # Main content area
        st.markdown("### Answer")
        
        # If analysis is available, use highlighted text
        if "analysis" in results and self.show_visualizations:
            st.markdown(
                create_text_highlight_html(
                    results.get("text", ""),
                    results.get("analysis", {}),
                    self.theme
                ),
                unsafe_allow_html=True
            )
        else:
            # Otherwise, just show the text
            st.markdown(results.get("text", ""))
        
        # Bottom sections
        if self.show_sources and results.get("documents"):
            self._render_sources(results)
        
        if self.show_visualizations and "analysis" in results:
            self._render_visualizations(results)
    
    def _render_metrics(self, results: Dict[str, Any]):
        """
        Render metrics at the top of the results
        
        Args:
            results: Search results
        """
        # Create metrics row
        metrics_cols = st.columns(4)
        
        # Confidence score
        confidence_score = results.get("analysis", {}).get("confidence_score", 0)
        confidence_level = "High" if confidence_score >= 0.8 else "Medium" if confidence_score >= 0.6 else "Low"
        
        with metrics_cols[0]:
            st.metric(
                "Confidence",
                confidence_level,
                f"{confidence_score:.2f}"
            )
        
        # Provider information
        provider = results.get("provider", "Unknown")
        provider_display = provider.capitalize()
        
        with metrics_cols[1]:
            st.metric(
                "Provider",
                provider_display,
                "Fallback" if results.get("metadata", {}).get("fallback") else None
            )
        
        # Time metrics
        execution_time = results.get("execution_time", 0)
        
        with metrics_cols[2]:
            st.metric(
                "Response Time",
                f"{execution_time:.2f}s"
            )
        
        # Source count
        source_count = len(results.get("documents", []))
        
        with metrics_cols[3]:
            st.metric(
                "Sources",
                source_count
            )
    
    def _render_sources(self, results: Dict[str, Any]):
        """
        Render sources section
        
        Args:
            results: Search results
        """
        documents = results.get("documents", [])
        
        with st.expander("Sources", expanded=self.expand_sources):
            if not documents:
                st.info("No sources available")
                return
            
            for i, doc in enumerate(documents):
                # Create a card-like container for each source
                with st.container():
                    # Title and link
                    title = doc.get("title", f"Source {i+1}")
                    url = doc.get("url", "#")
                    
                    st.markdown(f"#### {i+1}. {title}")
                    st.markdown(f"[{url}]({url})")
                    
                    # Snippet or content
                    snippet = doc.get("snippet", "")
                    content = doc.get("content", "")
                    
                    if snippet:
                        st.markdown(f"**Snippet**: {snippet}")
                    elif content:
                        # Truncate long content
                        display_content = content[:500] + "..." if len(content) > 500 else content
                        st.markdown(f"**Content**: {display_content}")
                    
                    st.divider()
    
    def _render_visualizations(self, results: Dict[str, Any]):
        """
        Render visualizations section
        
        Args:
            results: Search results
        """
        with st.expander("Analysis", expanded=False):
            # Create two columns for visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence gauge
                st.plotly_chart(
                    create_confidence_gauge(
                        results.get("analysis", {}).get("confidence_score", 0),
                        self.theme
                    ),
                    use_container_width=True
                )
            
            with col2:
                # Verification breakdown
                st.plotly_chart(
                    create_verification_breakdown(
                        results.get("analysis", {}),
                        self.theme
                    ),
                    use_container_width=True
                )
            
            # Source diversity chart
            st.plotly_chart(
                create_source_diversity_chart(
                    results.get("analysis", {}),
                    self.theme
                ),
                use_container_width=True
            )
            
            # Analysis summary
            if "analysis_summary" in results:
                summary = results["analysis_summary"]
                
                st.markdown("### Analysis Summary")
                st.markdown(f"**Confidence**: {summary.get('confidence', 'Unknown')}")
                
                if summary.get("issues"):
                    st.markdown("#### Potential Issues")
                    for issue in summary["issues"]:
                        st.markdown(f"- {issue}")
                
                st.markdown(f"**Verified statements**: {summary.get('verified_count', 0)}")
                st.markdown(f"**Unverified statements**: {summary.get('potential_issues_count', 0)}")


class ConfigurationPanel:
    """
    Custom configuration panel component for Streamlit
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        save_callback: Callable[[Dict[str, Any]], None],
        theme: str = "dark"
    ):
        """
        Initialize the configuration panel
        
        Args:
            config: Current configuration
            save_callback: Function to call when configuration is saved
            theme: Color theme ('light', 'dark', or 'auto')
        """
        self.config = config
        self.save_callback = save_callback
        self.theme = theme
        
        # Initialize state
        if "current_config" not in st.session_state:
            st.session_state.current_config = self.config.copy()
        
        if "config_changed" not in st.session_state:
            st.session_state.config_changed = False
    
    def _handle_config_change(self):
        """Mark configuration as changed"""
        st.session_state.config_changed = True
    
    def _save_config(self):
        """Save configuration using callback"""
        self.save_callback(st.session_state.current_config)
        st.session_state.config_changed = False
        st.success("Configuration saved successfully!")
    
    def _reset_config(self):
        """Reset configuration to initial values"""
        st.session_state.current_config = self.config.copy()
        st.session_state.config_changed = False
    
    def render(self):
        """
        Render the configuration panel
        
        Returns:
            Streamlit component
        """
        st.markdown("## Configuration")
        
        # Create tabs for different configuration sections
        tabs = st.tabs([
            "General",
            "Search Providers",
            "Cache",
            "Appearance"
        ])
        
        # General settings
        with tabs[0]:
            self._render_general_settings()
        
        # Search provider settings
        with tabs[1]:
            self._render_provider_settings()
        
        # Cache settings
        with tabs[2]:
            self._render_cache_settings()
        
        # Appearance settings
        with tabs[3]:
            self._render_appearance_settings()
        
        # Save/reset buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.button(
                "Save Configuration",
                on_click=self._save_config,
                disabled=not st.session_state.config_changed,
                use_container_width=True
            )
        
        with col2:
            st.button(
                "Reset Changes",
                on_click=self._reset_config,
                disabled=not st.session_state.config_changed,
                use_container_width=True
            )
    
    def _render_general_settings(self):
        """Render general settings section"""
        st.markdown("### General Settings")
        
        # Get current values from session state
        current_config = st.session_state.current_config
        
        # Default provider
        default_provider = current_config.get("search_engine", {}).get("default_provider", "cohere")
        provider_options = ["cohere", "openai", "anthropic", "web_api"]
        
        selected_provider = st.selectbox(
            "Default Search Provider",
            options=provider_options,
            index=provider_options.index(default_provider) if default_provider in provider_options else 0,
            key="config_default_provider",
            on_change=self._handle_config_change
        )
        
        # Update session state
        if "search_engine" not in current_config:
            current_config["search_engine"] = {}
        
        current_config["search_engine"]["default_provider"] = selected_provider
        
        # Advanced settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Use fallback providers
            use_fallback = current_config.get("search_engine", {}).get("use_fallback", True)
            
            selected_use_fallback = st.checkbox(
                "Use Fallback Providers",
                value=use_fallback,
                key="config_use_fallback",
                on_change=self._handle_config_change
            )
            
            current_config["search_engine"]["use_fallback"] = selected_use_fallback
        
        with col2:
            # Max fallbacks
            max_fallbacks = current_config.get("search_engine", {}).get("max_fallbacks", 2)
            
            selected_max_fallbacks = st.number_input(
                "Maximum Fallback Providers",
                min_value=1,
                max_value=5,
                value=max_fallbacks,
                key="config_max_fallbacks",
                on_change=self._handle_config_change
            )
            
            current_config["search_engine"]["max_fallbacks"] = selected_max_fallbacks
        
        # Result analysis
        analyze_results = current_config.get("search_engine", {}).get("analyze_results", True)
        
        selected_analyze_results = st.checkbox(
            "Analyze Search Results",
            value=analyze_results,
            key="config_analyze_results",
            on_change=self._handle_config_change
        )
        
        current_config["search_engine"]["analyze_results"] = selected_analyze_results
    
    def _render_provider_settings(self):
        """Render search provider settings section"""
        st.markdown("### Search Provider Settings")
        
        # Get current values from session state
        current_config = st.session_state.current_config
        
        # Ensure providers exist in config
        if "search_engine" not in current_config:
            current_config["search_engine"] = {}
        
        if "providers" not in current_config["search_engine"]:
            current_config["search_engine"]["providers"] = {}
        
        providers_config = current_config["search_engine"]["providers"]
        
        # Provider tabs
        provider_tabs = st.tabs([
            "Cohere",
            "OpenAI",
            "Anthropic",
            "Web API"
        ])
        
        # Cohere settings
        with provider_tabs[0]:
            st.markdown("#### Cohere Settings")
            
            # Ensure provider exists in config
            if "cohere" not in providers_config:
                providers_config["cohere"] = {}
            
            cohere_config = providers_config["cohere"]
            
            # API key
            api_key = cohere_config.get("api_key", "")
            
            entered_api_key = st.text_input(
                "API Key",
                value=api_key,
                type="password",
                key="config_cohere_api_key",
                on_change=self._handle_config_change
            )
            
            cohere_config["api_key"] = entered_api_key
            
            # Model settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Model
                model = cohere_config.get("model", "command")
                model_options = ["command", "command-light", "command-r", "command-r-plus"]
                
                selected_model = st.selectbox(
                    "Model",
                    options=model_options,
                    index=model_options.index(model) if model in model_options else 0,
                    key="config_cohere_model",
                    on_change=self._handle_config_change
                )
                
                cohere_config["model"] = selected_model
            
            with col2:
                # Temperature
                temperature = cohere_config.get("temperature", 0.5)
                
                selected_temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=temperature,
                    step=0.1,
                    key="config_cohere_temperature",
                    on_change=self._handle_config_change
                )
                
                cohere_config["temperature"] = selected_temperature
            
            # Max tokens
            max_tokens = cohere_config.get("max_tokens", 1000)
            
            selected_max_tokens = st.slider(
                "Maximum Tokens",
                min_value=100,
                max_value=4000,
                value=max_tokens,
                step=100,
                key="config_cohere_max_tokens",
                on_change=self._handle_config_change
            )
            
            cohere_config["max_tokens"] = selected_max_tokens
            
            # Connector ID
            connector_id = cohere_config.get("connector_id", "web-search")
            
            entered_connector_id = st.text_input(
                "Connector ID",
                value=connector_id,
                key="config_cohere_connector_id",
                on_change=self._handle_config_change
            )
            
            cohere_config["connector_id"] = entered_connector_id
        
        # OpenAI settings
        with provider_tabs[1]:
            st.markdown("#### OpenAI Settings")
            
            # Ensure provider exists in config
            if "openai" not in providers_config:
                providers_config["openai"] = {}
            
            openai_config = providers_config["openai"]
            
            # API key
            api_key = openai_config.get("api_key", "")
            
            entered_api_key = st.text_input(
                "API Key",
                value=api_key,
                type="password",
                key="config_openai_api_key",
                on_change=self._handle_config_change
            )
            
            openai_config["api_key"] = entered_api_key
            
            # Model settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Model
                model = openai_config.get("model", "gpt-4-turbo")
                model_options = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
                
                selected_model = st.selectbox(
                    "Model",
                    options=model_options,
                    index=model_options.index(model) if model in model_options else 0,
                    key="config_openai_model",
                    on_change=self._handle_config_change
                )
                
                openai_config["model"] = selected_model
            
            with col2:
                # Temperature
                temperature = openai_config.get("temperature", 0.5)
                
                selected_temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=temperature,
                    step=0.1,
                    key="config_openai_temperature",
                    on_change=self._handle_config_change
                )
                
                openai_config["temperature"] = selected_temperature
            
            # Max tokens
            max_tokens = openai_config.get("max_tokens", 1000)
            
            selected_max_tokens = st.slider(
                "Maximum Tokens",
                min_value=100,
                max_value=4000,
                value=max_tokens,
                step=100,
                key="config_openai_max_tokens",
                on_change=self._handle_config_change
            )
            
            openai_config["max_tokens"] = selected_max_tokens
        
        # Anthropic settings
        with provider_tabs[2]:
            st.markdown("#### Anthropic Settings")
            
            # Ensure provider exists in config
            if "anthropic" not in providers_config:
                providers_config["anthropic"] = {}
            
            anthropic_config = providers_config["anthropic"]
            
            # API key
            api_key = anthropic_config.get("api_key", "")
            
            entered_api_key = st.text_input(
                "API Key",
                value=api_key,
                type="password",
                key="config_anthropic_api_key",
                on_change=self._handle_config_change
            )
            
            anthropic_config["api_key"] = entered_api_key
            
            # Model settings
            col1, col2 = st.columns(2)
            
            with col1:
                # Model
                model = anthropic_config.get("model", "claude-3-opus-20240229")
                model_options = [
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
                
                selected_model = st.selectbox(
                    "Model",
                    options=model_options,
                    index=model_options.index(model) if model in model_options else 0,
                    key="config_anthropic_model",
                    on_change=self._handle_config_change
                )
                
                anthropic_config["model"] = selected_model
            
            with col2:
                # Temperature
                temperature = anthropic_config.get("temperature", 0.5)
                
                selected_temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=temperature,
                    step=0.1,
                    key="config_anthropic_temperature",
                    on_change=self._handle_config_change
                )
                
                anthropic_config["temperature"] = selected_temperature
            
            # Max tokens
            max_tokens = anthropic_config.get("max_tokens", 1000)
            
            selected_max_tokens = st.slider(
                "Maximum Tokens",
                min_value=100,
                max_value=4000,
                value=max_tokens,
                step=100,
                key="config_anthropic_max_tokens",
                on_change=self._handle_config_change
            )
            
            anthropic_config["max_tokens"] = selected_max_tokens
        
        # Web API settings
        with provider_tabs[3]:
            st.markdown("#### Web API Settings")
            
            # Ensure provider exists in config
            if "web_api" not in providers_config:
                providers_config["web_api"] = {}
            
            web_api_config = providers_config["web_api"]
            
            # API URL
            api_url = web_api_config.get("api_url", "")
            
            entered_api_url = st.text_input(
                "API URL",
                value=api_url,
                key="config_web_api_url",
                on_change=self._handle_config_change
            )
            
            web_api_config["api_url"] = entered_api_url
            
            # API key
            api_key = web_api_config.get("api_key", "")
            
            entered_api_key = st.text_input(
                "API Key",
                value=api_key,
                type="password",
                key="config_web_api_key",
                on_change=self._handle_config_change
            )
            
            web_api_config["api_key"] = entered_api_key
            
            # API key header
            api_key_header = web_api_config.get("api_key_header", "Authorization")
            
            entered_api_key_header = st.text_input(
                "API Key Header",
                value=api_key_header,
                key="config_web_api_key_header",
                on_change=self._handle_config_change
            )
            
            web_api_config["api_key_header"] = entered_api_key_header
            
            # Advanced settings (timeout, headers, path configuration)
            with st.expander("Advanced Settings", expanded=False):
                # Timeout
                timeout = web_api_config.get("timeout", 10)
                
                selected_timeout = st.number_input(
                    "Timeout (seconds)",
                    min_value=1,
                    max_value=60,
                    value=timeout,
                    key="config_web_api_timeout",
                    on_change=self._handle_config_change
                )
                
                web_api_config["timeout"] = selected_timeout
                
                # Result path
                result_path = web_api_config.get("result_path", "")
                
                entered_result_path = st.text_input(
                    "Result Path (e.g., 'data.results')",
                    value=result_path,
                    key="config_web_api_result_path",
                    help="JSON path to the result data",
                    on_change=self._handle_config_change
                )
                
                web_api_config["result_path"] = entered_result_path
                
                # Text path
                text_path = web_api_config.get("text_path", "")
                
                entered_text_path = st.text_input(
                    "Text Path (e.g., 'data.results.text')",
                    value=text_path,
                    key="config_web_api_text_path",
                    help="JSON path to the response text",
                    on_change=self._handle_config_change
                )
                
                web_api_config["text_path"] = entered_text_path
                
                # Documents path
                documents_path = web_api_config.get("documents_path", "")
                
                entered_documents_path = st.text_input(
                    "Documents Path (e.g., 'data.results.documents')",
                    value=documents_path,
                    key="config_web_api_documents_path",
                    help="JSON path to the documents array",
                    on_change=self._handle_config_change
                )
                
                web_api_config["documents_path"] = entered_documents_path
                
                # Custom parameters
                st.markdown("##### Custom Parameters")
                st.markdown("Add custom parameters to include in the API request")
                
                # Initialize params if not present
                if "params" not in web_api_config:
                    web_api_config["params"] = {}
                
                params = web_api_config["params"]
                
                # Keys and values for existing parameters
                param_keys = list(params.keys()) + [""]
                param_values = list(params.values()) + [""]
                
                # Display parameter inputs
                for i in range(len(param_keys)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        key = st.text_input(
                            "Parameter Name",
                            value=param_keys[i],
                            key=f"config_web_api_param_key_{i}",
                            on_change=self._handle_config_change
                        )
                    
                    with col2:
                        value = st.text_input(
                            "Value",
                            value=param_values[i] if i < len(param_values) else "",
                            key=f"config_web_api_param_value_{i}",
                            on_change=self._handle_config_change
                        )
                    
                    # Update params dictionary
                    if key and key != "":
                        params[key] = value
                    
                    # Only show 5 parameter inputs at most
                    if i >= 4:
                        break
    
    def _render_cache_settings(self):
        """Render cache settings section"""
        st.markdown("### Cache Settings")
        
        # Get current values from session state
        current_config = st.session_state.current_config
        
        # Ensure cache config exists
        if "cache" not in current_config:
            current_config["cache"] = {}
        
        cache_config = current_config["cache"]
        
        # Enable caching
        use_cache = current_config.get("search_engine", {}).get("use_cache", True)
        
        selected_use_cache = st.checkbox(
            "Enable Result Caching",
            value=use_cache,
            key="config_use_cache",
            on_change=self._handle_config_change
        )
        
        if "search_engine" not in current_config:
            current_config["search_engine"] = {}
        
        current_config["search_engine"]["use_cache"] = selected_use_cache
        
        # Cache TTL
        cache_ttl = current_config.get("search_engine", {}).get("cache_ttl", 86400)
        
        selected_cache_ttl = st.number_input(
            "Cache Time-to-Live (seconds)",
            min_value=60,
            max_value=604800,  # 1 week
            value=cache_ttl,
            key="config_cache_ttl",
            on_change=self._handle_config_change
        )
        
        current_config["search_engine"]["cache_ttl"] = selected_cache_ttl
        
        # Cache type
        cache_type = cache_config.get("cache_type", "multi")
        cache_type_options = ["multi", "memory", "disk", "db"]
        
        selected_cache_type = st.selectbox(
            "Cache Type",
            options=cache_type_options,
            index=cache_type_options.index(cache_type) if cache_type in cache_type_options else 0,
            key="config_cache_type",
            on_change=self._handle_config_change
        )
        
        cache_config["cache_type"] = selected_cache_type
        
        # Memory cache settings
        if selected_cache_type in ["multi", "memory"]:
            with st.expander("Memory Cache Settings", expanded=False):
                # Memory size
                memory_size = cache_config.get("memory_size", 100)
                
                selected_memory_size = st.number_input(
                    "Maximum Memory Cache Size (entries)",
                    min_value=10,
                    max_value=1000,
                    value=memory_size,
                    key="config_memory_size",
                    on_change=self._handle_config_change
                )
                
                cache_config["memory_size"] = selected_memory_size
        
        # Disk cache settings
        if selected_cache_type in ["multi", "disk"]:
            with st.expander("Disk Cache Settings", expanded=False):
                # Cache directory
                disk_cache_dir = cache_config.get("disk_cache_dir", ".cache")
                
                entered_disk_cache_dir = st.text_input(
                    "Cache Directory",
                    value=disk_cache_dir,
                    key="config_disk_cache_dir",
                    on_change=self._handle_config_change
                )
                
                cache_config["disk_cache_dir"] = entered_disk_cache_dir
                
                # Max size
                disk_max_size_mb = cache_config.get("disk_max_size_mb", 100)
                
                selected_disk_max_size_mb = st.number_input(
                    "Maximum Disk Cache Size (MB)",
                    min_value=10,
                    max_value=1000,
                    value=disk_max_size_mb,
                    key="config_disk_max_size_mb",
                    on_change=self._handle_config_change
                )
                
                cache_config["disk_max_size_mb"] = selected_disk_max_size_mb
        
        # Database cache settings
        if selected_cache_type in ["multi", "db"]:
            with st.expander("Database Cache Settings", expanded=False):
                # Database path
                db_path = cache_config.get("db_path", "cache.db")
                
                entered_db_path = st.text_input(
                    "Database Path",
                    value=db_path,
                    key="config_db_path",
                    on_change=self._handle_config_change
                )
                
                cache_config["db_path"] = entered_db_path
                
                # Max entries
                db_max_entries = cache_config.get("db_max_entries", 1000)
                
                selected_db_max_entries = st.number_input(
                    "Maximum Database Cache Entries",
                    min_value=100,
                    max_value=10000,
                    value=db_max_entries,
                    key="config_db_max_entries",
                    on_change=self._handle_config_change
                )
                
                cache_config["db_max_entries"] = selected_db_max_entries
        
        # Query similarity threshold
        query_similarity_threshold = cache_config.get("query_similarity_threshold", 0.8)
        
        selected_query_similarity_threshold = st.slider(
            "Query Similarity Threshold",
            min_value=0.5,
            max_value=1.0,
            value=query_similarity_threshold,
            step=0.05,
            key="config_query_similarity_threshold",
            help="Minimum similarity score for query matching (higher = stricter matching)",
            on_change=self._handle_config_change
        )
        
        cache_config["query_similarity_threshold"] = selected_query_similarity_threshold
    
    def _render_appearance_settings(self):
        """Render appearance settings section"""
        st.markdown("### Appearance Settings")
        
        # Get current values from session state
        current_config = st.session_state.current_config
        
        # Ensure appearance config exists
        if "appearance" not in current_config:
            current_config["appearance"] = {}
        
        appearance_config = current_config["appearance"]
        
        # Theme selection
        theme = appearance_config.get("theme", self.theme)
        theme_options = ["light", "dark", "auto"]
        
        selected_theme = st.selectbox(
            "Theme",
            options=theme_options,
            index=theme_options.index(theme) if theme in theme_options else 0,
            key="config_theme",
            on_change=self._handle_config_change
        )
        
        appearance_config["theme"] = selected_theme
        
        # Visualization settings
        show_visualizations = appearance_config.get("show_visualizations", True)
        
        selected_show_visualizations = st.checkbox(
            "Show Visualizations",
            value=show_visualizations,
            key="config_show_visualizations",
            on_change=self._handle_config_change
        )
        
        appearance_config["show_visualizations"] = selected_show_visualizations
        
        # Source settings
        show_sources = appearance_config.get("show_sources", True)
        
        selected_show_sources = st.checkbox(
            "Show Sources",
            value=show_sources,
            key="config_show_sources",
            on_change=self._handle_config_change
        )
        
        appearance_config["show_sources"] = selected_show_sources
        
        # Advanced mode
        advanced_mode = appearance_config.get("advanced_mode", False)
        
        selected_advanced_mode = st.checkbox(
            "Enable Advanced Mode by Default",
            value=advanced_mode,
            key="config_advanced_mode",
            on_change=self._handle_config_change
        )
        
        appearance_config["advanced_mode"] = selected_advanced_mode
        
        # Page layout
        layout = appearance_config.get("layout", "wide")
        layout_options = ["wide", "centered"]
        
        selected_layout = st.radio(
            "Page Layout",
            options=layout_options,
            index=layout_options.index(layout) if layout in layout_options else 0,
            horizontal=True,
            key="config_layout",
            on_change=self._handle_config_change
        )
        
        appearance_config["layout"] = selected_layout


class SearchHistoryViewer:
    """
    Custom search history viewer component for Streamlit
    """
    
    def __init__(
        self,
        history: List[Dict[str, Any]],
        rerun_callback: Callable[[str], None],
        theme: str = "dark"
    ):
        """
        Initialize the search history viewer
        
        Args:
            history: Search history entries
            rerun_callback: Function to call when a search is rerun
            theme: Color theme ('light', 'dark', or 'auto')
        """
        self.history = history
        self.rerun_callback = rerun_callback
        self.theme = theme
    
    def render(self):
        """
        Render the search history viewer
        
        Returns:
            Streamlit component
        """
        st.markdown("## Search History")
        
        if not self.history:
            st.info("No search history available")
            return
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        
        # Convert timestamps to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
            df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        
        # Create history table
        history_data = []
        for i, row in df.iterrows():
            history_data.append({
                "ID": row.get("id", i),
                "Query": row.get("query", ""),
                "Time": row.get("time", ""),
                "Date": row.get("date", ""),
                "Provider": row.get("provider", ""),
                "Success": "✓" if row.get("success", False) else "✗",
                "Response Time (s)": f"{row.get('execution_time', 0):.2f}",
                "Sources": row.get("document_count", 0),
                "Confidence": row.get("confidence", "Unknown")
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Display the table
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Create history chart
        if len(df) > 1 and "execution_time" in df.columns and "timestamp" in df.columns:
            st.markdown("### Search Performance")
            
            # Create performance chart
            fig = go.Figure()
            
            # Add execution time trace
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["execution_time"],
                mode="lines+markers",
                name="Response Time",
                line=dict(color="#1E88E5", width=2),
                marker=dict(size=8)
            ))
            
            # Update layout
            fig.update_layout(
                title="Response Time History",
                xaxis_title="Time",
                yaxis_title="Response Time (s)",
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Create rerun interface
        st.markdown("### Rerun Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select query to rerun
            if len(self.history) > 0:
                queries = [entry.get("query", "") for entry in self.history]
                selected_query = st.selectbox(
                    "Select Query to Rerun",
                    options=queries,
                    index=0
                )
            else:
                selected_query = ""
                st.info("No searches available to rerun")
        
        with col2:
            # Rerun button
            st.write("")  # Spacer for alignment
            if len(self.history) > 0:
                rerun_button = st.button(
                    "Rerun Selected Search",
                    use_container_width=True
                )
                
                if rerun_button and selected_query:
                    self.rerun_callback(selected_query)


# Create a function to initialize the Streamlit page with consistent styling
def setup_page(title="AI-powered Search", layout="wide", theme="dark"):
    """
    Set up the Streamlit page with consistent styling
    
    Args:
        title: Page title
        layout: Page layout ('wide' or 'centered')
        theme: Color theme ('light', 'dark', or 'auto')
    """
    # Set page configuration
    st.set_page_config(
        page_title=title,
        page_icon="🔍",
        layout=layout,
        initial_sidebar_state="expanded"
    )
    
    # Apply theme
    if theme == "dark":
        st.markdown("""
        <style>
        :root {
            --background-color: #0e1117;
            --text-color: #fafafa;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme == "light":
        st.markdown("""
        <style>
        :root {
            --background-color: #ffffff;
            --text-color: #0e1117;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Add custom CSS for component styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: rgba(250, 250, 250, 0.7);
    }
    
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Card styling */
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 10px;
    }
    
    /* Source card styling */
    .source-card {
        border-left: 3px solid #4D9DE0;
        padding-left: 10px;
        margin-bottom: 15px;
    }
    
    /* Confidence indicators */
    .confidence-high {
        color: #3BB273;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #FFC857;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #E15554;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# Example usage in a Streamlit app
def example_search_app():
    """
    Example usage of the components in a Streamlit app
    """
    # Set up the page
    setup_page(theme="dark")
    
    # Create a dummy search function
    def search_function(query, **kwargs):
        time.sleep(1)  # Simulate search
        return {
            "text": f"Example search result for: {query}",
            "documents": [
                {"title": "Example Source 1", "url": "https://example.com/1", "snippet": "This is an example source."},
                {"title": "Example Source 2", "url": "https://example.com/2", "snippet": "This is another example."}
            ],
            "provider": kwargs.get("provider", "cohere"),
            "execution_time": 1.2,
            "success": True,
            "analysis": {
                "confidence_score": 0.85,
                "verified_facts": ["Example search result for: " + query],
                "potential_hallucinations": [],
                "source_diversity": 0.8,
                "source_credibility": 0.7,
                "factual_consistency": 0.9
            },
            "analysis_summary": {
                "confidence": "High",
                "issues": [],
                "verified_count": 1,
                "potential_issues_count": 0
            }
        }
    
    # Create app header
    st.markdown('<div class="main-header">AI-powered Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Get verified answers with advanced AI</div>', unsafe_allow_html=True)
    
    # Create search interface
    search_interface = SearchInterface(
        search_function=search_function,
        placeholder="What would you like to search for?",
        advanced_mode=False,
        theme="dark"
    )
    
    # Render search interface
    search_executed, search_results = search_interface.render()
    
    # Render results if search was executed
    if search_executed and search_results:
        st.markdown("---")
        
        # Create result display
        result_display = ResultDisplay(
            theme="dark",
            show_visualizations=True,
            show_sources=True
        )
        
        # Render results
        result_display.render(search_results)
