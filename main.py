"""Main Streamlit application for Data Science Agent."""

import traceback
from pathlib import Path

import streamlit as st

from agents.base import AgentContext, UserFeedback
from agents.data_cleaning import DataCleaningAgent
from agents.data_ingestion import DataIngestionAgent
from agents.data_profiling import DataProfilingAgent
from agents.evaluation import EvaluationAgent
from agents.feature_engineering import FeatureEngineeringAgent
from agents.feature_selection import FeatureSelectionAgent
from agents.modeling import ModelingAgent
from agents.orchestration import OrchestrationAgent, PipelinePhase
from agents.report_generation import ReportGenAgent
from config import settings
from memory.memory_client import memory_client
from ui.components import (
    render_agent_result,
    render_approval_interface,
    render_cleaning_summary,
    render_data_preview,
    render_error_display,
    render_file_uploader,
    render_memory_stats,
    render_pipeline_status,
    render_profile_summary,
    render_target_selector,
)
from utils.logging import get_agent_logger

# Configure Streamlit page
st.set_page_config(
    page_title="Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logger
logger = get_agent_logger("MainApp")


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OrchestrationAgent()
    
    if "context" not in st.session_state:
        st.session_state.context = AgentContext()
    
    if "agents_registered" not in st.session_state:
        st.session_state.agents_registered = False
    
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    
    if "target_variable" not in st.session_state:
        st.session_state.target_variable = None


def register_agents():
    """Register all agents with the orchestrator."""
    if not st.session_state.agents_registered:
        orchestrator = st.session_state.orchestrator
        
        # Register all available agents
        orchestrator.register_agent(PipelinePhase.DATA_INGESTION, DataIngestionAgent())
        orchestrator.register_agent(PipelinePhase.DATA_PROFILING, DataProfilingAgent())
        orchestrator.register_agent(PipelinePhase.DATA_CLEANING, DataCleaningAgent())
        orchestrator.register_agent(PipelinePhase.FEATURE_SELECTION, FeatureSelectionAgent())
        orchestrator.register_agent(PipelinePhase.FEATURE_ENGINEERING, FeatureEngineeringAgent())
        orchestrator.register_agent(PipelinePhase.MODELING, ModelingAgent())
        orchestrator.register_agent(PipelinePhase.EVALUATION, EvaluationAgent())
        orchestrator.register_agent(PipelinePhase.REPORT_GENERATION, ReportGenAgent())
        
        st.session_state.agents_registered = True
        logger.info("All available agents registered with orchestrator")


def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.title("🤖 Data Science Agent")
    
    # Pipeline status
    st.sidebar.subheader("📊 Pipeline Status")
    orchestrator = st.session_state.orchestrator
    status = orchestrator.get_pipeline_status()
    
    st.sidebar.write(f"**Current Phase:** {status['current_phase'].replace('_', ' ').title()}")
    st.sidebar.write(f"**Status:** {status['pipeline_state'].replace('_', ' ').title()}")
    st.sidebar.progress(status["progress_percentage"] / 100)
    
    # Controls
    st.sidebar.subheader("🎛️ Controls")
    
    if st.sidebar.button("🔄 Reset Pipeline"):
        orchestrator.reset_pipeline()
        st.session_state.context = AgentContext()
        st.session_state.current_result = None
        st.session_state.uploaded_file_path = None
        st.session_state.target_variable = None
        st.sidebar.success("Pipeline reset successfully!")
        st.rerun()
    
    # Memory stats
    st.sidebar.subheader("🧠 Memory Stats")
    try:
        memory_stats = memory_client.get_memory_stats()
        vector_entries = memory_stats.get("vector_store", {}).get("total_entries", 0)
        st.sidebar.write(f"Vector entries: {vector_entries}")
        
        symbolic_entries = sum(
            stats.get("total_entries", 0) 
            for stats in memory_stats.get("symbolic_store", {}).values()
            if isinstance(stats, dict)
        )
        st.sidebar.write(f"Symbolic entries: {symbolic_entries}")
        
    except Exception as e:
        st.sidebar.error(f"Memory stats error: {e}")
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    
    debug_mode = st.sidebar.checkbox("Debug Mode", value=settings.debug)
    if debug_mode != settings.debug:
        settings.debug = debug_mode
    
    project_id = st.sidebar.text_input(
        "Project ID",
        value=st.session_state.context.user_preferences.get("project_id", "default"),
        help="Identifier for this project session"
    )
    
    if project_id:
        st.session_state.context.user_preferences["project_id"] = project_id


def render_main_content():
    """Render the main content area."""
    st.title("🤖 AI Data Science Pipeline")
    st.markdown("Automated end-to-end data science with intelligent agents and human oversight.")
    
    # Initialize components
    initialize_session_state()
    register_agents()
    
    orchestrator = st.session_state.orchestrator
    context = st.session_state.context
    
    # Pipeline status overview
    render_pipeline_status(orchestrator)
    
    # Handle current pipeline state
    try:
        # Step 1: File upload
        if not st.session_state.uploaded_file_path:
            st.subheader("1️⃣ Upload Your Dataset")
            uploaded_path = render_file_uploader()
            
            if uploaded_path:
                st.session_state.uploaded_file_path = uploaded_path
                context.metadata["uploaded_file"] = uploaded_path
                st.rerun()
            else:
                st.info("👆 Please upload a dataset to begin the pipeline.")
                return
        
        # Step 2: Target selection (if data is loaded)
        if st.session_state.uploaded_file_path and context.data is not None:
            if not st.session_state.target_variable:
                st.subheader("2️⃣ Select Target Variable")
                selected_target = render_target_selector(context.data)
                
                if selected_target:
                    st.session_state.target_variable = selected_target
                    context.metadata["target_variable"] = selected_target
                    context.target = selected_target
                    st.rerun()
        
        # Step 3: Handle user approval if needed
        current_result = st.session_state.current_result
        if current_result and current_result.requires_approval:
            st.subheader("🤔 Review Required")
            
            # Display the result
            render_agent_result(current_result)
            
            # Approval interface
            user_feedback = render_approval_interface(current_result)
            
            if user_feedback:
                # Check if this is natural language feedback that needs LLM processing
                if user_feedback.get('natural_language_feedback') and user_feedback.get('use_llm_processing'):
                    from utils.feedback_processor import feedback_processor
                    
                    try:
                        with st.spinner("🤖 Processing your feedback with AI..."):
                            # Process natural language feedback
                            processed_feedback = feedback_processor.process_natural_language_feedback(
                                feedback_text=user_feedback['feedback_text'],
                                agent_result=current_result,
                                context=context
                            )
                            
                            # Display the processed result
                            if hasattr(processed_feedback, 'additional_context') and processed_feedback.additional_context:
                                explanation = processed_feedback.additional_context.get('llm_explanation', '')
                                intent = processed_feedback.additional_context.get('intent', 'unknown')
                                
                                st.success(f"✅ **AI Understanding:** {intent.title()}")
                                if explanation:
                                    st.info(f"💭 **Explanation:** {explanation}")
                                
                                # Show generated code if any
                                if processed_feedback.overrides.get('custom_engineering_code'):
                                    with st.expander("🔍 Generated Code"):
                                        st.code(processed_feedback.overrides['custom_engineering_code'], language='python')
                                
                                # Show what will be applied
                                if processed_feedback.overrides:
                                    with st.expander("📋 Actions to Apply"):
                                        for key, value in processed_feedback.overrides.items():
                                            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                            
                            feedback = processed_feedback
                            
                            # Only execute if not preview mode
                            if not user_feedback.get('preview_only', False):
                                # Automatically execute the processed feedback
                                with st.spinner("⚙️ Applying AI-processed feedback and re-running..."):
                                    result = orchestrator.run(context, feedback)
                                    st.session_state.current_result = result
                                    st.session_state.context = result.context
                                    st.success("✅ AI feedback applied and pipeline re-executed!")
                                    st.rerun()
                            else:
                                # Just show the preview without executing
                                st.info("👀 Preview mode - click 'Process & Apply with AI' to execute these changes")
                                return
                    
                    except Exception as e:
                        st.error(f"Error processing natural language feedback: {e}")
                        logger.error(f"Natural language feedback processing error: {e}")
                        return
                
                else:
                    # Regular feedback processing
                    feedback = UserFeedback(**user_feedback)
                    
                    # Execute orchestrator with feedback
                    try:
                        with st.spinner("⚙️ Applying feedback and re-running..."):
                            result = orchestrator.run(context, feedback)
                            st.session_state.current_result = result
                            st.session_state.context = result.context
                            st.success("✅ Feedback applied successfully!")
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing feedback: {e}")
                        logger.error(f"Feedback processing error: {e}")
            
            return
        
        # Step 4: Execute next pipeline phase
        if (st.session_state.uploaded_file_path and 
            (not context.target or st.session_state.target_variable)):
            
            # Auto-execute if no approval needed, or manual execution
            if st.button("▶️ Execute Next Phase", type="primary") or not current_result:
                try:
                    # Update context with current metadata
                    if st.session_state.uploaded_file_path:
                        context.metadata["uploaded_file"] = st.session_state.uploaded_file_path
                    if st.session_state.target_variable:
                        context.metadata["target_variable"] = st.session_state.target_variable
                        context.target = st.session_state.target_variable
                    
                    # Run orchestrator
                    result = orchestrator.run(context)
                    st.session_state.current_result = result
                    st.session_state.context = result.context
                    
                    # Display result
                    render_agent_result(result)
                    
                    # Render phase-specific content
                    render_phase_specific_content(result)
                    
                    if not result.requires_approval and result.success:
                        st.rerun()
                
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Pipeline execution error: {error_msg}")
                    logger.error(f"Pipeline error: {error_msg}\n{traceback.format_exc()}")
            
            # Display current result if available
            elif current_result:
                render_agent_result(current_result)
                render_phase_specific_content(current_result)
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"Application error: {error_msg}")
        logger.error(f"App error: {error_msg}\n{traceback.format_exc()}")


def render_phase_specific_content(result):
    """Render content specific to the current phase.
    
    Args:
        result: AgentResult from the current phase
    """
    if not result.success:
        return
    
    agent_name = result.agent_name
    context = result.context
    
    # Data Ingestion specific content
    if agent_name == "DataIngestionAgent":
        if context.data is not None:
            render_data_preview(context.data, "📁 Ingested Data")
    
    # Data Profiling specific content
    elif agent_name == "DataProfilingAgent":
        if context.profile_summary:
            render_profile_summary(context.profile_summary)
        
        if context.data is not None:
            render_data_preview(context.data, "📊 Profiled Data")
    
    # Data Cleaning specific content
    elif agent_name == "DataCleaningAgent":
        if "cleaning_result" in result.artifacts:
            render_cleaning_summary(result.artifacts["cleaning_result"])
        
        if context.cleaned_data is not None:
            render_data_preview(context.cleaned_data, "🧹 Cleaned Data")
        
        # Show before/after comparison
        if context.data is not None and "cleaning_result" in result.artifacts:
            cleaning_result = result.artifacts["cleaning_result"]
            
            st.subheader("📈 Before vs After Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Original Rows", cleaning_result["original_shape"][0])
                st.metric("Original Columns", cleaning_result["original_shape"][1])
            
            with col2:
                st.metric("Final Rows", cleaning_result["final_shape"][0])
                st.metric("Final Columns", cleaning_result["final_shape"][1])


def main():
    """Main application entry point."""
    try:

        initialize_session_state()
        # Render sidebar
        render_sidebar()
        
        # Render main content
        render_main_content()
        
    except Exception as e:
        st.error(f"Critical application error: {e}")
        logger.error(f"Critical error: {e}\n{traceback.format_exc()}")
        
        # Offer reset option
        if st.button("🔄 Reset Application"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()