# app.py
"""
Streamlit Application for AI-powered Search Engine.
Integrates various backend modules for a user-friendly search experience,
including standard search and proactive exploration.
"""

import streamlit as st
import time
import asyncio
import logging
from datetime import datetime # Added for history timestamp
from typing import Dict, Any, List
import html # Import html for escaping
import json # For previewing exploration results
# +++ Import copy +++
import copy

# Import singleton accessors and helper functions from other modules
from config_manager import get_config_manager, ConfigManager
from search_engine import get_search_engine, run_search, run_exploration, SearchEngine
from visualization import display_sources
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.models.widgets import Button
from bokeh.models import CustomJS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("app")

# --- Helper Functions ---

def load_app_config(config_manager: ConfigManager) -> Dict[str, Any]:
    """Load application config using the provided config manager instance."""
    logger.debug("Loading application config...")
    try:
        config = config_manager.get_active_config()
        logger.debug(f"Loaded theme: {config.get('appearance', {}).get('theme')}")
        return config
    except Exception as e:
        st.error(f"Fatal Error: Failed to load configuration. Please check logs. Error: {e}")
        logger.critical(f"Failed to load configuration: {e}", exc_info=True)
        return {
            "appearance": {"theme": "light", "layout": "centered"},
            "search_engine": {"default_provider": "unknown"}
        }

async def perform_search_async(query: str) -> Dict[str, Any]:
    """Wrapper to run the async search function."""
    try:
        # Pass relevant settings if run_search accepts them
        results = await run_search(
            query,
            # provider=st.session_state.get("search_engine.default_provider"), # Example
            # use_cache=st.session_state.get("search_engine.use_cache"),       # Example
            # use_fallback=st.session_state.get("search_engine.use_fallback")  # Example
        )
        return results
    except Exception as e:
        logger.error(f"Exception during run_search for query '{query}': {e}", exc_info=True)
        return {
            "original_query": query,
            "processed_query_data": None,
            "search_result": {
                'provider': None, 'success': False, 'text': f"Search execution failed: {str(e)}",
                'documents': [], 'error': f"{type(e).__name__}: {str(e)}",
                'raw_response': None, 'execution_time': 0
            },
            "analysis": None, "providers_tried": [], "total_execution_time": 0,
            "cache_hit": False, "error": f"Search execution failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

async def perform_exploration_async(topic: str) -> Dict[str, Any]:
    """Wrapper to run the async exploration function."""
    try:
        # Pass relevant settings if run_exploration accepts them
        suggestions = await run_exploration(
             topic
            # provider=st.session_state.get("search_engine.default_provider") # Example
        )
        return suggestions
    except Exception as e:
        logger.error(f"Exception during run_exploration for topic '{topic}': {e}", exc_info=True)
        return {
            'success': False, 'topic': topic, 'error': f"Exploration execution failed: {str(e)}",
            'provider': None, 'suggestions': None, 'execution_time': 0.0
        }

def add_copy_button(text_to_copy, button_text="📋 Copy"):
    """ Add a copy to clipboard button for the provided text. """
    base_key = f"copy_{hash(text_to_copy)}_{int(time.time())}"
    button_key = f"{base_key}_button"
    message_id = f"copy_message_{base_key}"

    copy_js = CustomJS(args=dict(text=text_to_copy, message_id=message_id), code="""
        navigator.clipboard.writeText(text).then(function() {
            var messageElement = document.getElementById(message_id);
            if (messageElement) {
                messageElement.style.display = 'inline';
                setTimeout(function() { messageElement.style.display = 'none'; }, 1500);
            } else { console.error('Copy message element not found for ID:', message_id); }
        }).catch(err => { console.error('Failed to copy text: ', err); });
    """)

    copy_button = Button(label=button_text, width=80, name=button_key)
    copy_button.js_on_event("button_click", copy_js)

    col1, col2 = st.columns([1, 5])
    with col1:
        streamlit_bokeh_events(
            copy_button, events="button_click", key=button_key,
            refresh_on_update=False, override_height=35, debounce_time=0,
        )
    with col2:
        st.markdown(
            f"<span id='{message_id}' style='display: none; color: green; margin-left: 5px;'>Copied!</span>",
            unsafe_allow_html=True
        )

# --- UI Components ---

def display_sidebar(search_engine: SearchEngine):
    """Create the sidebar with settings and interactive search history."""
    st.sidebar.title("AI Search Settings")

    # --- Search Engine Settings ---
    st.sidebar.subheader("Search Engine")
    available_providers = search_engine.get_available_providers()
    default_provider_value = None

    if not available_providers:
        st.sidebar.warning("No search providers available!")
        if "search_engine.default_provider" not in st.session_state:
             st.session_state["search_engine.default_provider"] = None
        selectbox_disabled = True
        default_index = 0
    else:
        selectbox_disabled = False
        preferred_provider = "cohere"
        if "search_engine.default_provider" not in st.session_state:
            st.session_state["search_engine.default_provider"] = preferred_provider if preferred_provider in available_providers else available_providers[0]

        default_provider_value = st.session_state["search_engine.default_provider"]
        try:
            default_index = available_providers.index(default_provider_value)
        except ValueError:
             logger.warning(f"Invalid provider '{default_provider_value}' in session state. Resetting.")
             default_index = 0
             st.session_state["search_engine.default_provider"] = available_providers[0]

    st.sidebar.selectbox(
        "Primary Search Provider", options=available_providers, index=default_index,
        key="search_engine.default_provider", disabled=selectbox_disabled
    )
    st.sidebar.toggle("Enable Caching", key="search_engine.use_cache")
    st.sidebar.toggle("Enable Provider Fallback", key="search_engine.use_fallback")
    st.sidebar.divider()

    # --- History Management ---
    st.sidebar.subheader("History")
    history_list = st.session_state.get('search_history', [])

    with st.sidebar.expander("View Search History", expanded=False):
        if not history_list:
            st.caption("No searches recorded in this session.")
        else:
            # Iterate using index for safe deletion
            indices_to_delete = []
            for i in range(len(history_list)):
                entry = history_list[i] # Get entry by current index
                entry_key_base = f"history_{i}_{entry.get('timestamp', 'notime')}"
                query_display = entry.get('query', 'Missing Query')
                timestamp_display = entry.get('timestamp', 'Missing Timestamp')
                preview_display = entry.get('preview', '') # Get the saved preview

                st.markdown(f"**{html.escape(query_display)}**")
                st.caption(f"Searched: {timestamp_display}")

                # --- Display Preview ---
                if preview_display:
                     # Use markdown for italics and escape potential html
                     st.markdown(f"> *{html.escape(preview_display)}*", unsafe_allow_html=True)

                # --- Action Buttons ---
                col1, col2 = st.columns(2) # Buttons side-by-side
                with col1:
                    rerun_key = f"{entry_key_base}_rerun"
                    if st.button("🔄 Re-run", key=rerun_key, help="Load this query into the search bar", use_container_width=True):
                        # Set the main query input value state
                        st.session_state.search_query_input = entry.get('query', '') # Use .get for safety
                        # Optionally clear results when re-running?
                        st.session_state.results = None
                        st.session_state.exploration_results = None
                        st.session_state.search_performed = False
                        st.rerun()

                with col2:
                    delete_key = f"{entry_key_base}_delete"
                    if st.button("🗑️ Delete", key=delete_key, help="Delete this entry", use_container_width=True):
                        # Mark index for deletion after the loop
                        indices_to_delete.append(i)

                if i < len(history_list) - 1:
                    st.divider()

            # --- Perform Deletion After Loop ---
            if indices_to_delete:
                 # Delete indices in reverse order to avoid index shifting issues
                 for index in sorted(indices_to_delete, reverse=True):
                     try:
                         del st.session_state.search_history[index]
                         logger.info(f"Deleted history entry at index {index}")
                     except IndexError:
                          logger.warning(f"Attempted to delete history index {index}, but it was out of bounds.")
                     except Exception as e:
                          logger.error(f"Error deleting history item at index {index}: {e}")
                 st.rerun() # Rerun once after all deletions are done

    # Clear All History Button
    if history_list:
         if st.sidebar.button("Clear All History", key="clear_history_button"):
            st.session_state.search_history = []
            st.sidebar.success("History cleared!")
            time.sleep(1)
            st.rerun()

def display_results(results: Dict[str, Any], app_config: Dict[str, Any]):
    """Display the standard search results and sources with copy button."""
    search_result_data = results.get("search_result", {})
    search_success = search_result_data.get("success", False)
    show_sources_flag = True

    st.subheader("Search Result")
    result_text = search_result_data.get("text", "").strip()

    if search_success and result_text:
        st.markdown(result_text)
        add_copy_button(result_text)
    elif search_success and not result_text:
        st.info("Search successful, but no text summary was generated.")
    else:
        error_msg = search_result_data.get('error', 'N/A')
        st.warning(f"Search did not produce text result. Error (if any): {html.escape(error_msg)}")

    if search_success:
        col1, col2, col3 = st.columns(3)
        with col1: st.caption(f"Time: {results.get('total_execution_time', 0):.2f}s")
        # --- FIXED: Use if/else block for st.success/st.info ---
        with col2:
            if results.get('cache_hit', False):
                 st.success("⚡ Cache Hit")
            else:
                 st.info("☁️ Live Result")
        # --- End Fix ---
        with col3:
            provider = search_result_data.get("provider")
            if provider: st.caption(f"Provider: {provider}")

    if show_sources_flag:
        documents = search_result_data.get("documents", [])
        if documents:
            with st.container(border=True): display_sources(documents)
        elif search_success and not results.get('cache_hit'):
            st.info("No source documents were returned by the search provider.")


def display_exploration_suggestions(suggestions_data: Dict[str, Any]):
    """Display the generated exploration suggestions."""
    topic = suggestions_data.get('topic', 'N/A')
    st.subheader(f"Exploration Suggestions for: {html.escape(topic)}")
    success = suggestions_data.get('success', False)
    provider = suggestions_data.get('provider', 'N/A')
    exec_time = suggestions_data.get('execution_time', 0.0)

    if success:
        st.caption(f"Generated by: {provider} | Time: {exec_time:.2f}s")
        suggestions = suggestions_data.get('suggestions')
        if not suggestions:
            st.warning("The AI provider returned a response, but no suggestions could be parsed.")
            return

        categories = {
            'sub_topics': "Related Sub-topics", 'key_questions': "Key Open Questions",
            'influential_works': "Influential Researchers/Works",
            'opposing_viewpoints': "Potential Controversies/Opposing Viewpoints"
        }
        any_suggestions_found = False
        for key, title in categories.items():
            items = suggestions.get(key)
            if items and isinstance(items, list):
                any_suggestions_found = True
                with st.expander(title, expanded=True):
                    for item in items: st.markdown(f"- {html.escape(item)}")
        if not any_suggestions_found:
             st.info("No specific suggestions were generated in the defined categories.")
    else:
        error_msg = suggestions_data.get('error', 'An unknown error occurred.')
        st.error(f"Failed to generate exploration suggestions.\nError: {html.escape(error_msg)}")
        st.caption(f"Provider Attempted: {provider} | Time: {exec_time:.2f}s")

# --- Main Application Logic ---

# +++ Define Callback Function for Clearing Results +++
def clear_results_callback():
    """Callback function to clear search results and input."""
    st.session_state.search_performed = False
    st.session_state.results = None
    st.session_state.exploration_results = None
    st.session_state.search_query_input = "" # Clear state linked to text_input key
    # st.session_state.current_query = "" # Also clear if necessary
    logger.debug("Clear results callback executed.")

def main():
    """Main application entry point."""
    try:
        config_manager = get_config_manager()
        search_engine = get_search_engine()
    except Exception as manager_init_error:
        st.error(f"Fatal Error: Failed to initialize core managers: {manager_init_error}")
        logger.critical(f"Failed to initialize core managers: {manager_init_error}", exc_info=True)
        return

    app_config = load_app_config(config_manager)

    st.set_page_config(
        page_title="AI-Powered Search", page_icon="🧭",
        layout="wide", initial_sidebar_state="expanded"
    )

    # --- Initialize Session State Variables ---
    if 'search_performed' not in st.session_state: st.session_state.search_performed = False
    if 'results' not in st.session_state: st.session_state.results = None
    if 'exploration_results' not in st.session_state: st.session_state.exploration_results = None
    if 'current_query' not in st.session_state: st.session_state.current_query = ""
    if 'search_history' not in st.session_state: st.session_state.search_history = []
    if "search_engine.default_provider" not in st.session_state: pass
    if "search_engine.use_cache" not in st.session_state: st.session_state["search_engine.use_cache"] = True
    if "search_engine.use_fallback" not in st.session_state: st.session_state["search_engine.use_fallback"] = True
    if "search_query_input" not in st.session_state: st.session_state.search_query_input = ""

    # Display sidebar
    display_sidebar(search_engine)

    st.title("AI-Powered Search & Exploration")
    st.write("Enter your question for a direct answer, or a topic to explore related ideas.")

    # Search Form
    with st.form(key='search_form'):
        # Text input relies on its key
        query_from_input_widget = st.text_input( # Renamed variable to avoid confusion
            "Enter Question or Topic:",
            key="search_query_input",
        )
        explore_mode = st.checkbox(
            "Explore this topic", key="explore_mode_checkbox"
        )
        submit_button = st.form_submit_button(label='Search / Explore 🚀')

        # This block executes ONLY when the form is submitted
        if submit_button and st.session_state.search_query_input: # Check state linked to key
            current_search_query = st.session_state.search_query_input # Get value from state
            st.session_state.current_query = current_search_query
            st.session_state.search_performed = True
            st.session_state.results = None
            st.session_state.exploration_results = None

            action_successful = False
            result_preview = ""
            action_result = None # Store the dict returned by search/explore

            with st.spinner("Processing..."):
                try:
                    if explore_mode:
                        logger.info(f"Running EXPLORATION for topic: '{current_search_query}'")
                        action_result = asyncio.run(perform_exploration_async(current_search_query))
                        st.session_state.exploration_results = action_result
                        if action_result and action_result.get('success'):
                             action_successful = True
                             suggestions = action_result.get('suggestions')
                             if suggestions:
                                 preview_parts = []
                                 if suggestions.get('sub_topics'): preview_parts.append(f"Sub-topics: {', '.join(suggestions['sub_topics'][:2])}")
                                 if suggestions.get('key_questions'): preview_parts.append(f"Questions: {suggestions['key_questions'][0]}")
                                 result_preview = " | ".join(preview_parts)
                                 if len(result_preview) > 150: result_preview = result_preview[:147] + "..."
                             else: result_preview = "Exploration successful, no suggestions parsed."

                    else:
                        logger.info(f"Running SEARCH for query: '{current_search_query}'")
                        action_result = asyncio.run(perform_search_async(current_search_query))
                        st.session_state.results = action_result
                        if action_result and action_result.get("search_result", {}).get('success'):
                             action_successful = True
                             result_text = action_result.get("search_result", {}).get("text", "")
                             if result_text:
                                 result_preview = result_text[:150].replace('\n', ' ') + ("..." if len(result_text) > 150 else "")
                             else: result_preview = "Search successful, no text summary."

                    # Save to History
                    if action_successful:
                        history_entry = {
                            "query": current_search_query,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "mode": "explore" if explore_mode else "search",
                            "preview": result_preview
                        }
                        current_history = st.session_state.get('search_history', [])
                        current_history.insert(0, history_entry)
                        max_history_size = 25
                        st.session_state.search_history = current_history[:max_history_size]
                        logger.info(f"Saved query '{current_search_query}' with preview to history.")

                except Exception as e:
                    logger.error(f"Error running async task from form: {e}", exc_info=True)
                    error_payload = {
                        "original_query": current_search_query, "error": f"Execution failed: {e}",
                        "search_result": {'success': False, 'text': f"Execution failed: {e}"} if not explore_mode else None,
                        "suggestions": None if explore_mode else None, "success": False
                    }
                    # Assign error payload to the correct state variable
                    if explore_mode: st.session_state.exploration_results = error_payload
                    else: st.session_state.results = error_payload
                    action_result = error_payload # Ensure action_result has the error payload

            # Rerun after processing to update display
            st.rerun()
        elif submit_button and not st.session_state.search_query_input:
            # Handle case where submit is clicked with empty input
            st.warning("Please enter a query or topic.")


    # --- MODIFIED Clear Results Button: Uses on_click ---
    # Display this button OUTSIDE the form, only if results are shown
    if st.session_state.get('search_performed', False): # Use .get for safety
        col1, col2 = st.columns([1, 5]) # Adjust layout as needed
        with col1:
            st.button(
                "Clear Results",
                key="clear_results_button_main", # Unique key
                use_container_width=True,
                on_click=clear_results_callback, # Attach the callback
                help="Clear the current search results and input field."
                # The callback handles the state changes and rerun
            )
    # --- End Modification ---


    # Display Results Area (conditionally based on search_performed state)
    if st.session_state.get('search_performed', False):
        st.divider()
        # Prioritize exploration results if they exist
        if st.session_state.get('exploration_results'):
            display_exploration_suggestions(st.session_state.exploration_results)
        # Otherwise, display standard search results if they exist
        elif st.session_state.get('results'):
            display_results(st.session_state.results, app_config)
        # If search_performed is True but no results yet (e.g., during processing before rerun)
        # else:
            # st.caption("Processing or results cleared.") # Optional placeholder


if __name__ == "__main__":
    try: main()
    except Exception as main_app_error:
        # Check for specific Streamlit error to avoid recursive st.error calls
        is_streamlit_command_error = isinstance(main_app_error, (AttributeError, st.errors.StreamlitAPIException)) and "_repr_html_" in str(main_app_error)

        if is_streamlit_command_error:
             logger.error(f"Caught Streamlit command error: {main_app_error}", exc_info=True)
             print(f"CRITICAL STREAMLIT COMMAND ERROR: {main_app_error}") # Log to console instead of st.error
        else:
             logger.critical(f"Critical error in Streamlit app main execution: {main_app_error}", exc_info=True)
             try: st.error(f"A critical application error occurred: {main_app_error}")
             except Exception as display_error: print(f"CRITICAL ERROR: {main_app_error}. Error displaying message: {display_error}")

