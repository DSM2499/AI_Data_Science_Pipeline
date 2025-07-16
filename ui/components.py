"""Reusable UI components for Streamlit interface."""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from agents.base import AgentResult
from agents.orchestration import PipelinePhase, PipelineState


def render_pipeline_status(orchestrator) -> None:
    """Render the pipeline status overview.
    
    Args:
        orchestrator: OrchestrationAgent instance
    """
    status = orchestrator.get_pipeline_status()
    
    # Create columns for status display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Phase", status["current_phase"].replace("_", " ").title())
    
    with col2:
        st.metric("Progress", f"{status['progress_percentage']:.1f}%")
    
    with col3:
        st.metric("Completed", f"{len(status['completed_phases'])}/{status['total_phases']}")
    
    with col4:
        st.metric("Status", status["pipeline_state"].replace("_", " ").title())
    
    # Progress bar
    progress = status["progress_percentage"] / 100
    st.progress(progress)
    
    # Phase details
    if status["completed_phases"]:
        st.success(f"✅ Completed: {', '.join([p.replace('_', ' ').title() for p in status['completed_phases']])}")
    
    if status["failed_phases"]:
        st.error(f"❌ Failed: {', '.join([p.replace('_', ' ').title() for p in status['failed_phases']])}")


def render_agent_result(result: AgentResult) -> None:
    """Render the result of an agent execution.
    
    Args:
        result: AgentResult to display
    """
    # Agent header
    if result.success:
        st.success(f"✅ {result.agent_name} completed successfully")
    else:
        st.error(f"❌ {result.agent_name} failed")
    
    # Metrics display
    if result.metrics:
        # Filter metrics to only show scalar values for st.metric
        scalar_metrics = {}
        list_metrics = {}
        
        for key, value in result.metrics.items():
            if isinstance(value, (int, float, str)) or value is None:
                scalar_metrics[key] = value
            elif isinstance(value, list):
                list_metrics[key] = value
            else:
                # Convert other types to string
                scalar_metrics[key] = str(value)
        
        # Display scalar metrics
        if scalar_metrics:
            cols = st.columns(len(scalar_metrics))
            for i, (key, value) in enumerate(scalar_metrics.items()):
                with cols[i]:
                    st.metric(key.replace("_", " ").title(), value)
        
        # Display list metrics separately
        if list_metrics:
            for key, value in list_metrics.items():
                st.subheader(key.replace("_", " ").title())
                if isinstance(value, list) and len(value) > 0:
                    # Display as bullet points
                    for item in value:
                        st.write(f"• {item}")
                else:
                    st.write("No items")
    
    # Logs
    if result.logs:
        with st.expander("📋 Execution Logs"):
            for log in result.logs:
                st.text(log)
    
    # Warnings
    if result.warnings:
        with st.expander("⚠️ Warnings", expanded=True):
            for warning in result.warnings:
                st.warning(warning)
    
    # Suggestions
    if result.suggestions:
        with st.expander("💡 Suggestions"):
            for suggestion in result.suggestions:
                st.info(suggestion)
    
    # Error details
    if result.error:
        st.error(f"Error: {result.error}")


def render_data_preview(data: pd.DataFrame, title: str = "Data Preview") -> None:
    """Render a data preview with basic statistics.
    
    Args:
        data: DataFrame to preview
        title: Title for the preview section
    """
    st.subheader(title)
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", data.shape[0])
    
    with col2:
        st.metric("Columns", data.shape[1])
    
    with col3:
        st.metric("Memory (MB)", f"{data.memory_usage(deep=True).sum() / (1024*1024):.2f}")
    
    with col4:
        null_pct = (data.isnull().sum().sum() / data.size) * 100
        st.metric("Null %", f"{null_pct:.1f}")
    
    # Data sample
    st.dataframe(data.head(100))
    
    # Column info
    with st.expander("📊 Column Information"):
        col_info = pd.DataFrame({
            "Column": data.columns,
            "Type": data.dtypes.astype(str),
            "Non-Null": data.count(),
            "Null %": (data.isnull().sum() / len(data) * 100).round(2),
            "Unique": data.nunique(),
        })
        st.dataframe(col_info)


def render_profile_summary(profile_summary: Dict[str, Any]) -> None:
    """Render data profiling summary with visualizations.
    
    Args:
        profile_summary: Profile summary dictionary
    """
    st.subheader("📊 Data Profile Summary")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Shape", f"{profile_summary['dataset_shape']}")
    
    with col2:
        st.metric("Null Percentage", f"{profile_summary['overall_null_percentage']:.1f}%")
    
    with col3:
        st.metric("Duplicates", profile_summary['duplicate_rows'])
    
    with col4:
        st.metric("Memory (MB)", f"{profile_summary['memory_usage_mb']:.1f}")
    
    # Column type distribution
    col_types = {
        "Numeric": len(profile_summary.get('numeric_columns', [])),
        "Categorical": len(profile_summary.get('categorical_columns', [])),
        "DateTime": len(profile_summary.get('datetime_columns', [])),
        "Boolean": len(profile_summary.get('boolean_columns', [])),
    }
    
    if sum(col_types.values()) > 0:
        fig = px.pie(
            values=list(col_types.values()),
            names=list(col_types.keys()),
            title="Column Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data quality issues
    quality_issues = [
        ("High Null Columns", len(profile_summary.get('high_null_columns', []))),
        ("Constant Columns", len(profile_summary.get('constant_columns', []))),
        ("High Cardinality", len(profile_summary.get('high_cardinality_columns', []))),
        ("Potential IDs", len(profile_summary.get('potential_identifiers', []))),
    ]
    
    with st.expander("🔍 Data Quality Issues"):
        for issue, count in quality_issues:
            if count > 0:
                st.warning(f"{issue}: {count}")
    
    # Null statistics visualization
    null_stats = profile_summary.get('null_statistics', {})
    if null_stats:
        with st.expander("📈 Null Values by Column"):
            null_df = pd.DataFrame(list(null_stats.items()), columns=['Column', 'Null_Percentage'])
            null_df = null_df[null_df['Null_Percentage'] > 0].sort_values('Null_Percentage', ascending=False)
            
            if not null_df.empty:
                fig = px.bar(
                    null_df.head(20),
                    x='Null_Percentage',
                    y='Column',
                    orientation='h',
                    title="Top 20 Columns by Null Percentage"
                )
                st.plotly_chart(fig, use_container_width=True)


def render_cleaning_summary(cleaning_result: Dict[str, Any]) -> None:
    """Render data cleaning summary.
    
    Args:
        cleaning_result: Cleaning result dictionary
    """
    st.subheader("🧹 Data Cleaning Summary")
    
    # Before/after comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Original Shape", f"{cleaning_result['original_shape']}")
    
    with col2:
        st.metric("Final Shape", f"{cleaning_result['final_shape']}")
    
    # Operations summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calculate operations count from operations_applied list
        operations_count = len(cleaning_result.get('operations_applied', []))
        st.metric("Operations", operations_count)
    
    with col2:
        st.metric("Rows Removed", cleaning_result['rows_removed'])
    
    with col3:
        st.metric("Columns Removed", cleaning_result['columns_removed'])
    
    with col4:
        st.metric("Nulls Filled", cleaning_result['null_values_filled'])
    
    # Operations details
    operations = cleaning_result.get('operations', [])
    if operations:
        with st.expander("🔧 Applied Operations"):
            for i, op in enumerate(operations, 1):
                st.write(f"{i}. **{op['operation_type']}** on `{op['column']}`: {op['reason']}")
    
    # Generated cleaning code
    if cleaning_result.get('cleaning_code'):
        with st.expander("💻 Generated Cleaning Code"):
            st.code(cleaning_result['cleaning_code'], language='python')


def render_approval_interface(result: AgentResult) -> Optional[Dict[str, Any]]:
    """Render user approval interface for agent results.
    
    Args:
        result: AgentResult requiring approval
        
    Returns:
        Dictionary with user feedback or None if no action taken
    """
    st.subheader("🤔 User Approval Required")
    
    if result.user_message:
        st.info(result.user_message)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approve & Continue", key=f"approve_{result.execution_id}"):
            return {"approved": True}
    
    with col2:
        if st.button("🔄 Request Changes", key=f"changes_{result.execution_id}"):
            return {
                "approved": False, 
                "feedback_text": "User requested changes to this phase",
                "overrides": {"request_changes": True}
            }
    
    with col3:
        if st.button("⏪ Rollback", key=f"rollback_{result.execution_id}"):
            return {
                "approved": False,
                "feedback_text": "User requested rollback", 
                "overrides": {"rollback": True}
            }
    
    # Feedback form
    with st.expander("💬 Provide Feedback", expanded=True):
        st.write("**Natural Language Feedback:**")
        
        # Help section with examples
        with st.expander("💡 How to give natural language feedback", expanded=False):
            st.markdown("""
            **You can describe what you want in plain English!**
            
            **Feature Engineering Examples:**
            - "Create a ratio of income to age"
            - "Add interaction between education and experience" 
            - "Skip the polynomial features, they're too complex"
            - "Create log transformation of the salary column"
            - "Make a new feature that bins age into groups"
            
            **Data Cleaning Examples:**
            - "Don't fill missing values in the salary column"
            - "Remove outliers more aggressively"
            - "Keep the duplicate rows, don't remove them"
            
            **Parameter Adjustments:**
            - "Make the correlation threshold stricter (0.1 instead of 0.05)"
            - "Create more features, increase the ratio to 1.0"
            - "Be more conservative with feature creation"
            
            **Complex Requests:**
            - "Create a feature that combines age and salary as a ratio, then create buckets for different income levels"
            - "Skip any features that involve complex mathematical operations, I prefer simple ratios and differences"
            """)
        
        st.info("🤖 The AI will understand your request and generate the appropriate code and settings!")
        
        feedback_text = st.text_area(
            "Describe what you'd like to change or add:",
            placeholder="Example: I want to create a new feature that combines age and income as a ratio, and skip any polynomial features because they make the model too complex.",
            height=100,
            key=f"feedback_text_{result.execution_id}"
        )
        
        # Add buttons for natural language feedback processing
        if feedback_text:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Process & Apply with AI", key=f"process_apply_{result.execution_id}", type="primary"):
                    return {
                        "approved": False,
                        "natural_language_feedback": True,
                        "feedback_text": feedback_text,
                        "use_llm_processing": True
                    }
            with col2:
                if st.button("👀 Preview AI Understanding", key=f"preview_nl_{result.execution_id}"):
                    # Just show processing without executing
                    st.info("💡 Use 'Process & Apply with AI' to execute the changes")
                    return {
                        "approved": False,
                        "natural_language_feedback": True,
                        "feedback_text": feedback_text,
                        "use_llm_processing": True,
                        "preview_only": True
                    }
        
        # Custom overrides
        st.write("**Custom Overrides:**")
        
        # Agent-specific override options
        overrides = {}
        
        if result.agent_name == "DataIngestionAgent":
            new_target = st.selectbox(
                "Change target variable:",
                options=["Keep current"] + list(result.context.data.columns),
                key=f"target_override_{result.execution_id}"
            )
            if new_target != "Keep current":
                overrides["target_variable"] = new_target
        
        elif result.agent_name == "DataCleaningAgent":
            skip_operations = st.multiselect(
                "Skip these operations:",
                options=[op["operation_type"] for op in result.artifacts.get("cleaning_result", {}).get("operations", [])],
                key=f"skip_ops_{result.execution_id}"
            )
            if skip_operations:
                overrides["skip_operations"] = skip_operations
            
            custom_code = st.text_area(
                "Custom cleaning code:",
                placeholder="# Additional cleaning operations\n# data = your_custom_cleaning(data)",
                key=f"custom_code_{result.execution_id}"
            )
            if custom_code:
                overrides["custom_cleaning_code"] = custom_code
        
        elif result.agent_name == "FeatureEngineeringAgent":
            # Feature engineering specific overrides
            engineering_result = result.artifacts.get("engineering_result", {})
            operations_applied = engineering_result.get("operations_applied", [])
            
            if operations_applied:
                skip_features = st.multiselect(
                    "Skip these feature operations:",
                    options=[f"{op.get('operation_type', 'unknown')}: {op.get('feature_name', 'unnamed')}" 
                            for op in operations_applied if isinstance(op, dict)],
                    key=f"skip_features_{result.execution_id}"
                )
                if skip_features:
                    overrides["skip_feature_operations"] = skip_features
            
            # Custom feature engineering code
            custom_code = st.text_area(
                "Custom feature engineering code:",
                placeholder="# Additional feature engineering\n# Example: data['new_feature'] = data['col1'] * data['col2']",
                key=f"custom_feature_code_{result.execution_id}"
            )
            if custom_code:
                overrides["custom_engineering_code"] = custom_code
            
            # Feature engineering parameters
            st.write("**Adjust Parameters:**")
            col1, col2 = st.columns(2)
            
            with col1:
                max_features_ratio = st.slider(
                    "Max new features ratio:",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key=f"max_features_{result.execution_id}"
                )
                if max_features_ratio != 0.5:
                    overrides["max_new_features_ratio"] = max_features_ratio
            
            with col2:
                correlation_threshold = st.slider(
                    "Min correlation threshold:",
                    min_value=0.01,
                    max_value=0.2,
                    value=0.05,
                    step=0.01,
                    key=f"correlation_threshold_{result.execution_id}"
                )
                if correlation_threshold != 0.05:
                    overrides["correlation_threshold"] = correlation_threshold
        
        if st.button("📝 Submit Feedback", key=f"submit_feedback_{result.execution_id}"):
            return {
                "approved": False,
                "feedback_text": feedback_text,
                "overrides": overrides,
            }
    
    return None


def render_feedback_processing_result(processed_feedback: Dict[str, Any]) -> None:
    """Render the results of natural language feedback processing.
    
    Args:
        processed_feedback: Results from LLM feedback processing
    """
    st.subheader("🤖 AI Feedback Processing Results")
    
    intent = processed_feedback.get('intent', 'unknown')
    explanation = processed_feedback.get('explanation', '')
    actions = processed_feedback.get('actions', [])
    code = processed_feedback.get('code', '')
    parameters = processed_feedback.get('parameters', {})
    
    # Display intent and explanation
    col1, col2 = st.columns(2)
    
    with col1:
        intent_color = {
            'modify': '🔧',
            'add': '➕', 
            'create': '🆕',
            'skip': '⏭️',
            'rollback': '⏪',
            'unknown': '❓'
        }
        st.metric("Detected Intent", f"{intent_color.get(intent, '❓')} {intent.title()}")
    
    with col2:
        st.metric("Actions Found", len(actions))
    
    if explanation:
        st.info(f"💭 **AI Understanding:** {explanation}")
    
    # Display actions
    if actions:
        st.write("**Actions to Execute:**")
        for i, action in enumerate(actions, 1):
            st.write(f"{i}. {action}")
    
    # Display generated code
    if code and code.strip():
        with st.expander("🔍 Generated Code", expanded=True):
            st.code(code, language='python')
    
    # Display parameter adjustments
    if parameters:
        with st.expander("⚙️ Parameter Adjustments"):
            for param, value in parameters.items():
                st.write(f"• **{param.replace('_', ' ').title()}:** {value}")


def render_memory_stats(memory_stats: Dict[str, Any]) -> None:
    """Render memory system statistics.
    
    Args:
        memory_stats: Memory statistics dictionary
    """
    st.subheader("🧠 Memory System Status")
    
    # Vector store stats
    vector_stats = memory_stats.get("vector_store", {})
    if vector_stats:
        st.write("**Vector Store:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Entries", vector_stats.get("total_entries", 0))
        
        with col2:
            st.metric("Collection", vector_stats.get("collection_name", "Unknown"))
    
    # Symbolic store stats
    symbolic_stats = memory_stats.get("symbolic_store", {})
    if symbolic_stats:
        st.write("**Symbolic Store:**")
        
        for table_name, stats in symbolic_stats.items():
            if isinstance(stats, dict) and "total_entries" in stats:
                st.write(f"- {table_name.replace('_', ' ').title()}: {stats['total_entries']} entries")


def render_file_uploader() -> Optional[str]:
    """Render file upload interface.
    
    Returns:
        Path to uploaded file or None
    """
    st.subheader("📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or Parquet file",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Upload your dataset to begin the data science pipeline"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = f"project_output/data/{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        return file_path
    
    return None


def render_target_selector(data: pd.DataFrame) -> Optional[str]:
    """Render target variable selection interface.
    
    Args:
        data: DataFrame for target selection
        
    Returns:
        Selected target column name or None
    """
    if data is not None and not data.empty:
        st.subheader("🎯 Select Target Variable")
        
        target_options = ["No target (unsupervised)"] + list(data.columns)
        
        selected_target = st.selectbox(
            "Choose the target variable for prediction:",
            options=target_options,
            help="Select the column you want to predict"
        )
        
        if selected_target != "No target (unsupervised)":
            # Show target variable preview
            with st.expander("🔍 Target Variable Preview"):
                st.write(f"**Column:** {selected_target}")
                st.write(f"**Type:** {data[selected_target].dtype}")
                st.write(f"**Unique Values:** {data[selected_target].nunique()}")
                st.write(f"**Null Values:** {data[selected_target].isnull().sum()}")
                
                # Show value distribution
                if data[selected_target].nunique() < 20:
                    value_counts = data[selected_target].value_counts().head(10)
                    fig = px.bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        title=f"Distribution of {selected_target}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(
                        data,
                        x=selected_target,
                        title=f"Distribution of {selected_target}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            return selected_target
    
    return None


def render_error_display(error_msg: str, suggestions: Optional[List[str]] = None) -> None:
    """Render error message with suggestions.
    
    Args:
        error_msg: Error message to display
        suggestions: Optional list of suggestions
    """
    st.error(f"❌ Error: {error_msg}")
    
    if suggestions:
        st.subheader("💡 Suggestions:")
        for suggestion in suggestions:
            st.info(suggestion)


def render_phase_card(phase: PipelinePhase, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Render a phase status card.
    
    Args:
        phase: Pipeline phase
        status: Phase status (completed, running, pending, failed)
        details: Optional phase details
    """
    phase_name = phase.value.replace("_", " ").title()
    
    # Status emoji mapping
    status_emoji = {
        "completed": "✅",
        "running": "🔄",
        "pending": "⏳",
        "failed": "❌",
        "waiting": "🤔",
    }
    
    emoji = status_emoji.get(status, "❓")
    
    with st.container():
        st.write(f"{emoji} **{phase_name}**")
        
        if details:
            for key, value in details.items():
                st.write(f"  - {key}: {value}")
        
        st.write("---")