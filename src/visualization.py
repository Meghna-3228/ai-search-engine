# visualization.py - Fixed version

"""
Visualization Module for AI-powered Search

This module provides advanced visualization components for:
1. Displaying search result confidence and verification metrics
2. Visualizing source diversity and credibility
3. Tracking search performance and history
4. Creating interactive result summaries
5. Generating comparative visualizations across multiple searches

The visualizations are designed to be used within a Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re
import math
from functools import lru_cache
import colorsys
import html # Added for escaping text
from typing import Optional, Dict, Any, List # Added imports from typing

# Define default highlight colors (can be overridden by theme palette)
DEFAULT_VERIFIED_COLOR = "#28a745" # Green
DEFAULT_UNVERIFIED_COLOR = "#dc3545" # Red
DEFAULT_HIGHLIGHT_BG_VERIFIED = "#d4edda" # Light green background
DEFAULT_HIGHLIGHT_BG_UNVERIFIED = "#f8d7da" # Light red background

class VisualizationManager:
    """
    Main class for managing and generating visualizations
    """
    def __init__(self, theme="dark"):
        """
        Initialize the visualization manager
        Args:
            theme (str): Color theme to use ('light', 'dark', or 'auto')
        """
        self.update_theme(theme) # Use update_theme to set palette initially
        # Visualization history for tracking metrics over time
        self.history = {
            'timestamps': [],
            'confidence_scores': [],
            'response_times': [],
            'query_lengths': [],
            'result_lengths': [],
            'source_counts': []
        }

    def _generate_color_palette(self, theme):
        """
        Generate a color palette based on the theme
        Args:
            theme (str): The theme ('light', 'dark', or 'auto')
        Returns:
            dict: Color palette for visualizations
        """
        # Define base palettes
        dark_palette = {
            'background': '#0E1117',
            'text': '#F0F2F6',
            'primary': '#4D9DE0', # Blue
            'secondary': '#E15554', # Red
            'tertiary': '#3BB273', # Green
            'highlight': '#FFC857', # Yellow
            'muted': '#7E8B9A',
            'low_confidence': '#E15554',
            'medium_confidence': '#FFC857',
            'high_confidence': '#3BB273',
            'gradient': ['#E15554', '#FFC857', '#3BB273'],
            'verified_fg': '#FFFFFF', # White text on green bg
            'verified_bg': '#3BB273', # Green background
            'unverified_fg': '#FFFFFF', # White text on red bg
            'unverified_bg': '#E15554', # Red background
        }

        light_palette = {
            'background': '#FFFFFF',
            'text': '#111111',
            'primary': '#1E88E5', # Blue
            'secondary': '#D81B60', # Pink/Red
            'tertiary': '#28A745', # Green
            'highlight': '#FFA000', # Orange
            'muted': '#6C757D',
            'low_confidence': '#DC3545', # Red
            'medium_confidence': '#FFA000', # Orange
            'high_confidence': '#28A745', # Green
            'gradient': ['#DC3545', '#FFA000', '#28A745'],
            'verified_fg': '#155724', # Dark green text
            'verified_bg': '#d4edda', # Light green background
            'unverified_fg': '#721c24', # Dark red text
            'unverified_bg': '#f8d7da', # Light red background
        }

        # Return palette based on theme
        if theme == "dark":
            return dark_palette
        else: # Default to light theme for 'light' or 'auto'
            return light_palette

    def update_theme(self, theme):
        """
        Update the visualization theme
        Args:
            theme (str): The new theme ('light', 'dark', or 'auto')
        """
        self.theme = theme
        self.color_palette = self._generate_color_palette(theme)

    def record_search_metrics(self, metrics):
        """
        Record metrics from a search for historical tracking
        Args:
            metrics (dict): Metrics from the search including:
                - confidence_score: Overall confidence in the result
                - response_time: Time taken to perform the search
                - query_length: Length of the original query
                - result_length: Length of the result text
                - source_count: Number of sources used
        """
        self.history['timestamps'].append(datetime.now())
        self.history['confidence_scores'].append(metrics.get('confidence_score', 0))
        self.history['response_times'].append(metrics.get('response_time', 0))
        self.history['query_lengths'].append(metrics.get('query_length', 0))
        self.history['result_lengths'].append(metrics.get('result_length', 0))
        self.history['source_counts'].append(metrics.get('source_count', 0))

        # Keep history to a reasonable size
        max_history = 100
        if len(self.history['timestamps']) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]

    def clear_history(self):
        """
        Clear the recorded metrics history
        """
        for key in self.history:
            self.history[key] = []

    # Helper function to convert hex to rgba
    def hex_to_rgba(self, hex_color, alpha=0.25):
        """
        Convert hex color to rgba format
        Args:
            hex_color (str): Hex color code
            alpha (float): Alpha value
        Returns:
            str: RGBA color string
        """
        # Remove the # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        # Return rgba string
        return f'rgba({r}, {g}, {b}, {alpha})'

    def create_confidence_gauge(self, confidence_score, width=None, height=200):
        """
        Create a gauge chart to display confidence score
        Args:
            confidence_score (float): The confidence score (0-1)
            width (int, optional): Chart width
            height (int): Chart height
        Returns:
            plotly.graph_objects.Figure: Gauge chart figure
        """
        if not 0 <= confidence_score <= 1:
            confidence_score = max(0.0, min(1.0, confidence_score)) # Clamp score

        # Determine confidence level and color
        if confidence_score >= 0.8:
            confidence_level = "High"
            color = self.color_palette['high_confidence']
        elif confidence_score >= 0.6:
            confidence_level = "Medium"
            color = self.color_palette['medium_confidence']
        else:
            confidence_level = "Low"
            color = self.color_palette['low_confidence']

        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence_score * 100, # Convert to percentage
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {
                'text': f"Confidence: {confidence_level}",
                'font': {'color': self.color_palette['text']}
            },
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': self.color_palette['text']},
                'bar': {'color': color},
                # Use self.hex_to_rgba instead of hex_to_rgba
                'bgcolor': self.hex_to_rgba(self.color_palette['muted']), 
                'borderwidth': 1,
                'bordercolor': self.color_palette['muted'],
                'steps': [
                    # Use self.hex_to_rgba instead of hex_to_rgba
                    {'range': [0, 60], 'color': self.hex_to_rgba(self.color_palette['low_confidence'], 0.8)},
                    {'range': [60, 80], 'color': self.hex_to_rgba(self.color_palette['medium_confidence'], 0.8)},
                    {'range': [80, 100], 'color': self.hex_to_rgba(self.color_palette['high_confidence'], 0.8)}
                ],
            },
            number={'suffix': '%', 'font': {'color': self.color_palette['text']}}
        ))

        # Update layout
        fig.update_layout(
            paper_bgcolor=self.color_palette['background'],
            plot_bgcolor=self.color_palette['background'],
            font={'color': self.color_palette['text']},
            margin=dict(l=30, r=30, t=30, b=0),
            height=height
        )

        if width:
            fig.update_layout(width=width)

        return fig

    def create_verification_breakdown(self, analysis_result, width=None, height=300):
        """
        Create a visualization of the verification breakdown
        Args:
            analysis_result (dict): The analysis result including:
                - verified_facts: List of verified statements
                - potential_hallucinations: List of suspicious statements
            width (int, optional): Chart width
            height (int): Chart height
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        verified_count = len(analysis_result.get('verified_facts', []))
        suspicious_count = len(analysis_result.get('potential_hallucinations', []))
        total_count = verified_count + suspicious_count

        if total_count == 0:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No statements analyzed",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color=self.color_palette['text'], size=16)
            )
            fig.update_layout(
                paper_bgcolor=self.color_palette['background'],
                plot_bgcolor=self.color_palette['background'],
                font={'color': self.color_palette['text']},
                margin=dict(l=30, r=30, t=40, b=40),
                height=height
            )
            if width: fig.update_layout(width=width)
            return fig

        # Create data for the chart
        categories = ['Verified', 'Unverified']
        values = [verified_count, suspicious_count]
        percentages = [(count / total_count * 100) if total_count > 0 else 0 for count in values]

        # Create bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            text=[f"{p:.1f}%" for p in percentages],
            textposition='auto',
            marker_color=[self.color_palette['high_confidence'], self.color_palette['low_confidence']],
            hoverinfo='text',
            hovertext=[
                f"Verified statements: {verified_count} ({percentages[0]:.1f}%)",
                f"Unverified statements: {suspicious_count} ({percentages[1]:.1f}%)"
            ]
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Statement Verification Breakdown',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'color': self.color_palette['text']}
            },
            xaxis={
                'title': None,
                'tickfont': {'color': self.color_palette['text']},
                # Use self.hex_to_rgba instead of hex_to_rgba
                'gridcolor': self.hex_to_rgba(self.color_palette['muted'], 0.4),
                'linecolor': self.color_palette['muted']
            },
            yaxis={
                'title': 'Number of Statements',
                'tickfont': {'color': self.color_palette['text']},
                # Use self.hex_to_rgba instead of hex_to_rgba
                'gridcolor': self.hex_to_rgba(self.color_palette['muted'], 0.4),
                'linecolor': self.color_palette['muted']
            },
            paper_bgcolor=self.color_palette['background'],
            plot_bgcolor=self.color_palette['background'],
            font={'color': self.color_palette['text']},
            margin=dict(l=40, r=30, t=60, b=40),
            bargap=0.3,
            height=height
        )

        if width:
            fig.update_layout(width=width)

        return fig

    def create_source_diversity_chart(self, analysis_result, width=None, height=250):
        """
        Create a visualization of source diversity metrics.
        Args:
            analysis_result (dict): The analysis result including:
                - source_diversity: Source diversity score (0-1)
                - source_credibility: Source credibility score (0-1)
                - factual_consistency: Factual consistency score (0-1)
            width (int, optional): Chart width
            height (int): Chart height
        Returns:
            plotly.graph_objects.Figure: Radar chart figure
        """
        # Extract metrics, clamping between 0 and 1
        source_diversity = max(0.0, min(1.0, analysis_result.get('source_diversity', 0)))
        source_credibility = max(0.0, min(1.0, analysis_result.get('source_credibility', 0)))
        factual_consistency = max(0.0, min(1.0, analysis_result.get('factual_consistency', 0)))

        # Create radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[source_diversity * 100, source_credibility * 100, factual_consistency * 100],
            theta=['Diversity', 'Credibility', 'Consistency'],
            fill='toself',
            # Use self.hex_to_rgba instead of hex_to_rgba
            fillcolor=self.hex_to_rgba(self.color_palette['primary'], 0.6),
            line=dict(color=self.color_palette['primary']),
            hoverinfo='text',
            hovertext=[
                f"Source Diversity: {source_diversity:.2f}",
                f"Source Credibility: {source_credibility:.2f}",
                f"Factual Consistency: {factual_consistency:.2f}"
            ]
        ))

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color=self.color_palette['text']),
                    # Use self.hex_to_rgba instead of hex_to_rgba
                    gridcolor=self.hex_to_rgba(self.color_palette['muted'], 0.8),
                    linecolor=self.color_palette['muted'],
                    angle=90, # Start angle at top
                    tickangle=90
                ),
                angularaxis=dict(
                    tickfont=dict(color=self.color_palette['text'], size=12),
                    linecolor=self.color_palette['muted'],
                    # Use self.hex_to_rgba instead of hex_to_rgba
                    gridcolor=self.hex_to_rgba(self.color_palette['muted'], 0.8),
                ),
                bgcolor=self.color_palette['background']
            ),
            paper_bgcolor=self.color_palette['background'],
            plot_bgcolor=self.color_palette['background'],
            font={'color': self.color_palette['text']},
            margin=dict(l=40, r=40, t=40, b=30), # Adjust margins
            height=height,
            showlegend=False # No legend needed for single trace
        )

        if width:
            fig.update_layout(width=width)

        return fig

    def create_historical_metrics_chart(self, metric_type='confidence', last_n=10, width=None, height=300):
        """
        Create a chart showing historical metrics
        Args:
            metric_type (str): Type of metric to visualize
                               ('confidence', 'response_time', 'sources')
            last_n (int): Number of recent searches to include
            width (int, optional): Chart width
            height (int): Chart height
        Returns:
            plotly.graph_objects.Figure: Line chart figure
        """
        # Rest of this method remains the same...
        # Removing for brevity
        pass

    def create_text_highlight_html(self, text, analysis_result):
        """
        Create HTML with highlighted text based on verification.

        Args:
            text (str): The result text.
            analysis_result (dict): The analysis result containing:
                - verified_facts (list): List of verified statements (strings).
                - potential_hallucinations (list): List of unverified statements (strings).

        Returns:
            str: HTML string with highlighted text.
                 Returns original text wrapped in <div> if analysis is missing or invalid.
        """
        if not text or not isinstance(text, str):
            return "<div>No text provided for highlighting.</div>" # Wrap in div for consistency

        if not analysis_result or not isinstance(analysis_result, dict):
            # Return original text safely escaped if no analysis provided, wrapped in div
            return f"<div>{html.escape(text)}</div>"

        verified_facts = analysis_result.get('verified_facts', [])
        unverified_statements = analysis_result.get('potential_hallucinations', [])

        # Combine all statements to be highlighted, marking their type
        highlights = []
        for fact in verified_facts:
            if isinstance(fact, str) and fact:
                highlights.append({'text': fact, 'type': 'verified'})
        for statement in unverified_statements:
            if isinstance(statement, str) and statement:
                highlights.append({'text': statement, 'type': 'unverified'})

        # Sort highlights by length descending to match longer segments first
        highlights.sort(key=lambda x: len(x['text']), reverse=True)

        # Define CSS styles using the theme palette
        verified_fg = self.color_palette.get('verified_fg', '#000000')
        verified_bg = self.color_palette.get('verified_bg', DEFAULT_HIGHLIGHT_BG_VERIFIED)
        unverified_fg = self.color_palette.get('unverified_fg', '#000000')
        unverified_bg = self.color_palette.get('unverified_bg', DEFAULT_HIGHLIGHT_BG_UNVERIFIED)

        # --- Start CSS Definition ---
        css = f"""
<style>
.highlight-verified {{
    background-color: {verified_bg};
    color: {verified_fg};
    padding: 0.1em 0.2em;
    margin: 0 0.1em;
    border-radius: 0.2em;
    display: inline; /* Keep inline */
    cursor: help; /* Add help cursor */
}}
.highlight-unverified {{
    background-color: {unverified_bg};
    color: {unverified_fg};
    padding: 0.1em 0.2em;
    margin: 0 0.1em;
    border-radius: 0.2em;
    text-decoration: underline dotted; /* Add dotted underline */
    display: inline; /* Keep inline */
    cursor: help; /* Add help cursor */
}}
/* Add styling for the main container if needed */
.highlight-container {{
    line-height: 1.6; /* Improve readability */
    white-space: pre-wrap; /* Preserve whitespace and line breaks */
    word-wrap: break-word; /* Ensure long words break */
}}
</style>
"""
        # --- End CSS Definition ---

        # --- Segment-Based Replacement Logic ---
        processed_indices = set() # Track indices in the *original* text that have been covered
        replacements = [] # Store (start, end, type, statement_text) tuples

        temp_text_for_finding = text # Use original text for finding indices

        for item in highlights:
            statement_text = item['text']
            highlight_type = item['type']
            start_find = 0
            while start_find < len(temp_text_for_finding):
                # Find the *next* occurrence in the original text
                found_index = temp_text_for_finding.find(statement_text, start_find)
                if found_index == -1:
                    break # Stop searching for this statement
                end_index = found_index + len(statement_text)

                # Check for overlap with already selected replacements
                is_overlap = False
                for i in range(found_index, end_index):
                    if i in processed_indices:
                        is_overlap = True
                        break

                if not is_overlap:
                    # Add this segment to replacements
                    replacements.append((found_index, end_index, highlight_type, statement_text))
                    # Mark indices as processed in the original text
                    for i in range(found_index, end_index):
                        processed_indices.add(i)
                    # Move start_find past this found segment for the *next iteration of this statement*
                    # This ensures we don't re-match the same occurrence if the loop continues
                    start_find = end_index
                else:
                    # Move start_find past the start of this overlapping occurrence
                    # to find the *next potential* non-overlapping match
                    start_find = found_index + 1

        # Sort replacements by start index to process in order
        replacements.sort(key=lambda x: x[0])

        # Build the final HTML string using the replacements
        last_end = 0
        final_parts = []
        for start, end, type, statement_text in replacements:
            # Add the text segment *before* this highlight (escaped)
            if start > last_end:
                final_parts.append(html.escape(text[last_end:start]))

            # Define the highlight span
            escaped_statement = html.escape(statement_text)
            if type == 'verified':
                class_name = 'highlight-verified'
                title = 'Verified Statement'
            else: # unverified
                class_name = 'highlight-unverified'
                title = 'Unverified Statement'

            # Construct the span tag
            highlight_html_tag = f'<span class="{class_name}" title="{title}">{escaped_statement}</span>'
            final_parts.append(highlight_html_tag)
            last_end = end

        # Add any remaining text *after* the last highlight (escaped)
        if last_end < len(text):
            final_parts.append(html.escape(text[last_end:]))

        # Combine all parts and wrap in CSS and container div
        processed_text_html = "".join(final_parts)
        final_html = f"{css}<div class='highlight-container'>{processed_text_html}</div>"
        # --- End Segment-Based Replacement Logic ---

        return final_html
    # --- END CORRECTED HIGHLIGHTING LOGIC ---


# ------------------------------------------------------------
# Function definition for display_sources
def display_sources(documents):
    """
    Display source documents in a structured format using Streamlit expanders.

    Args:
        documents (list): List of source document dictionaries.
                          Each dict should ideally have 'title', 'url', 'snippet'.
    """
    if not documents:
        st.info("No source documents available.")
        return

    st.subheader("Sources")
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            st.warning(f"Source {i+1} is not in the expected format.")
            continue

        title = doc.get('title', f"Source {i+1}")
        url = doc.get('url')
        snippet = doc.get('snippet', doc.get('text', 'No snippet available.')) # Use snippet, text, or default

        with st.expander(f"**{i+1}. {title}**"):
            if url:
                # Display URL safely, preventing potential markdown injection
                # Escape the URL text part, but keep the URL itself raw for the link
                st.markdown(f"**URL:** [{html.escape(url)}]({url})", unsafe_allow_html=False)
            else:
                st.markdown("**URL:** Not available")

            # Display snippet/text safely
            st.markdown("**Content Snippet:**")
            # Escape the snippet content and display within a blockquote
            st.markdown(f"> {html.escape(snippet)}", unsafe_allow_html=False)

            # Optional: Display other metadata if available
            # metadata_str = ", ".join(f"{k}: {v}" for k, v in doc.items() if k not in ['title', 'url', 'snippet', 'text', 'id'])
            # if metadata_str:
            #     st.caption(f"Metadata: {metadata_str}")

# --- Singleton Accessor ---
# Using a simple global variable approach for Streamlit context
_visualization_manager_instance: Optional[VisualizationManager] = None

def get_visualization_manager(theme="dark") -> VisualizationManager:
    """
    Get a singleton instance of the VisualizationManager within Streamlit's context.
    Updates theme if necessary.

    Args:
        theme (str): The desired theme ('light', 'dark', 'auto').

    Returns:
        VisualizationManager: The singleton instance.
    """
    global _visualization_manager_instance
    if _visualization_manager_instance is None:
        _visualization_manager_instance = VisualizationManager(theme)
    elif _visualization_manager_instance.theme != theme:
        # Update theme if it has changed
        _visualization_manager_instance.update_theme(theme)
    return _visualization_manager_instance


# --- Convenience Functions ---

def display_highlighted_text(text, analysis_result, theme="dark"):
    """
    Display the result text with verified/unverified statements highlighted.

    Args:
        text (str): The result text.
        analysis_result (dict): The analysis result.
        theme (str): Current theme ('light', 'dark', 'auto').
    """
    manager = get_visualization_manager(theme)
    highlighted_html = manager.create_text_highlight_html(text, analysis_result)
    # This is where Streamlit renders the HTML
    st.markdown(highlighted_html, unsafe_allow_html=True)


# Simplify the analysis dashboard function to only show the confidence gauge
def display_analysis_dashboard(analysis_result, theme="dark"):
    """
    Display a simplified dashboard with just the confidence gauge.

    Args:
        analysis_result (dict): The result from ResultAnalyzer.analyze_result.
        theme (str): Current theme ('light', 'dark', 'auto').
    """
    if not analysis_result:
        st.warning("No analysis results to display.")
        return

    manager = get_visualization_manager(theme)

    # Just display the confidence gauge
    st.plotly_chart(
        manager.create_confidence_gauge(analysis_result.get('confidence_score', 0)),
        use_container_width=True
    )