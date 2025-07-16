"""Data Cleaning Agent - Identifies and applies data cleaning operations."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from agents.base import AgentContext, AgentResult, LLMCapableAgent, UserFeedback
from memory.memory_client import memory_client
from utils.exceptions import DataCleaningError, ValidationError


class CleaningOperation(BaseModel):
    """Represents a single cleaning operation."""
    
    operation_type: str
    column: str
    parameters: Dict[str, Any]
    reason: str
    impact_estimate: Optional[str] = None


class CleaningResult(BaseModel):
    """Result of data cleaning process."""
    
    operations_applied: List[CleaningOperation]
    rows_removed: int
    columns_removed: int
    null_values_filled: int
    duplicates_removed: int
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    cleaning_code: str


class DataCleaningAgent(LLMCapableAgent):
    """Agent responsible for identifying and applying data cleaning operations."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the data cleaning agent.
        
        Args:
            use_llm: Whether to use LLM for generating insights and explanations
        """
        super().__init__("DataCleaningAgent", use_llm)
        
        # Cleaning thresholds
        self.null_threshold = 90.0  # Remove columns with >90% nulls
        self.duplicate_threshold = 0.1  # Remove duplicates if >10% of data
        self.constant_threshold = 0.99  # Remove if >99% same value
        self.outlier_threshold = 3.0  # Z-score threshold for outlier detection
    
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute data cleaning process.
        
        Args:
            context: Pipeline context containing data and profile
            feedback: Optional user feedback for refinement
            
        Returns:
            AgentResult with cleaned data and cleaning log
        """
        try:
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            # Handle user feedback
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            data = context.data.copy()
            profile_summary = context.profile_summary
            target = context.target
            
            self.logger.info(f"Starting data cleaning for dataset with shape: {data.shape}")
            
            # Plan cleaning operations
            planned_operations = self._plan_cleaning_operations(data, profile_summary, target)
            
            # Apply cleaning operations
            cleaned_data, cleaning_result = self._apply_cleaning_operations(data, planned_operations)
            
            # Generate LLM insights if enabled
            llm_insights = ""
            if self.use_llm:
                llm_insights = self._generate_insight(context, feedback)
            
            # Update context
            context.cleaned_data = cleaned_data
            context.data = cleaned_data  # Update main data reference
            context.llm_insights[self.name] = llm_insights
            
            # Store cleaning results in memory
            self._store_cleaning_memory(cleaning_result, context)
            
            # Generate suggestions for user review
            suggestions = self._generate_suggestions(cleaning_result, profile_summary)
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Applied {len(cleaning_result.operations_applied)} cleaning operations",
                    f"Shape changed from {cleaning_result.original_shape} to {cleaning_result.final_shape}",
                    f"Removed {cleaning_result.rows_removed} rows and {cleaning_result.columns_removed} columns",
                    f"Filled {cleaning_result.null_values_filled} null values",
                ],
                metrics={
                    "operations_count": len(cleaning_result.operations_applied),
                    "rows_removed": cleaning_result.rows_removed,
                    "columns_removed": cleaning_result.columns_removed,
                    "null_values_filled": cleaning_result.null_values_filled,
                    "shape_reduction": (
                        (cleaning_result.original_shape[0] * cleaning_result.original_shape[1] - 
                         cleaning_result.final_shape[0] * cleaning_result.final_shape[1]) /
                        (cleaning_result.original_shape[0] * cleaning_result.original_shape[1])
                    ) * 100,
                },
                artifacts={
                    "cleaning_result": cleaning_result.dict(),
                    "cleaning_code": cleaning_result.cleaning_code,
                    "llm_insights": llm_insights,
                },
                suggestions=suggestions,
                requires_approval=len(cleaning_result.operations_applied) > 5 or cleaning_result.rows_removed > data.shape[0] * 0.2,
            )
            
            # Add warnings for significant changes
            warnings = self._generate_warnings(cleaning_result)
            if warnings:
                result.warnings = warnings
                result.requires_approval = True
                result.user_message = (
                    f"Data cleaning completed with {len(warnings)} significant changes. "
                    f"Please review before proceeding."
                )
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Data cleaning failed: {e}"],
                user_message="Data cleaning encountered an error. Please review the data and settings.",
            )
            
            self._log_execution(result)
            return result
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate that the context contains required data and profile.
        
        Args:
            context: Pipeline context to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValidationError: If validation fails
        """
        if context.data is None:
            raise ValidationError("No data found in context")
        
        if context.profile_summary is None:
            raise ValidationError("No profile summary found in context")
        
        if not isinstance(context.data, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame")
        
        return True
    
    def _plan_cleaning_operations(
        self,
        data: pd.DataFrame,
        profile_summary: Dict[str, Any],
        target: Optional[str],
    ) -> List[CleaningOperation]:
        """Plan cleaning operations based on data profile.
        
        Args:
            data: DataFrame to clean
            profile_summary: Data profiling results
            target: Target column name
            
        Returns:
            List of planned cleaning operations
        """
        operations = []
        
        # 1. Remove constant columns
        for col in profile_summary.get("constant_columns", []):
            if col != target:  # Never remove target
                operations.append(CleaningOperation(
                    operation_type="remove_column",
                    column=col,
                    parameters={},
                    reason=f"Column '{col}' has constant values",
                ))
        
        # 2. Remove high null columns
        null_stats = profile_summary.get("null_statistics", {})
        for col, null_pct in null_stats.items():
            if null_pct > self.null_threshold and col != target:
                operations.append(CleaningOperation(
                    operation_type="remove_column",
                    column=col,
                    parameters={"null_percentage": null_pct},
                    reason=f"Column '{col}' has {null_pct:.1f}% null values (>{self.null_threshold}%)",
                ))
        
        # 3. Remove potential identifier columns
        for col in profile_summary.get("potential_identifiers", []):
            if col != target:
                operations.append(CleaningOperation(
                    operation_type="remove_column",
                    column=col,
                    parameters={},
                    reason=f"Column '{col}' appears to be an identifier",
                ))
        
        # 4. Plan imputation for remaining columns with moderate nulls
        for col in data.columns:
            if col in [op.column for op in operations]:  # Skip if already marked for removal
                continue
            
            null_pct = null_stats.get(col, 0)
            if 0 < null_pct <= self.null_threshold:
                if col in profile_summary.get("numeric_columns", []):
                    operations.append(CleaningOperation(
                        operation_type="impute_numeric",
                        column=col,
                        parameters={"strategy": "median", "null_percentage": null_pct},
                        reason=f"Fill {null_pct:.1f}% null values in numeric column '{col}' with median",
                    ))
                elif col in profile_summary.get("categorical_columns", []):
                    operations.append(CleaningOperation(
                        operation_type="impute_categorical",
                        column=col,
                        parameters={"strategy": "most_frequent", "null_percentage": null_pct},
                        reason=f"Fill {null_pct:.1f}% null values in categorical column '{col}' with mode",
                    ))
        
        # 5. Plan duplicate removal
        if profile_summary.get("duplicate_rows", 0) > 0:
            operations.append(CleaningOperation(
                operation_type="remove_duplicates",
                column="all",
                parameters={"duplicate_count": profile_summary["duplicate_rows"]},
                reason=f"Remove {profile_summary['duplicate_rows']} duplicate rows",
            ))
        
        # 6. Plan outlier handling for numeric columns
        numeric_stats = profile_summary.get("numeric_stats", {})
        for col, stats in numeric_stats.items():
            if col != target and stats.get("outliers", 0) > 0:
                outlier_pct = (stats["outliers"] / stats["count"]) * 100
                if outlier_pct > 5:  # Only handle if >5% outliers
                    operations.append(CleaningOperation(
                        operation_type="handle_outliers",
                        column=col,
                        parameters={
                            "method": "clip",
                            "outlier_count": stats["outliers"],
                            "outlier_percentage": outlier_pct,
                        },
                        reason=f"Clip {stats['outliers']} outliers in '{col}' ({outlier_pct:.1f}%)",
                    ))
        
        # 7. Plan type corrections
        for col in data.columns:
            if col in [op.column for op in operations if op.operation_type == "remove_column"]:
                continue
            
            # Check for numeric columns stored as strings
            if col in profile_summary.get("categorical_columns", []):
                sample_values = data[col].dropna().head(100)
                if self._is_numeric_string(sample_values):
                    operations.append(CleaningOperation(
                        operation_type="convert_type",
                        column=col,
                        parameters={"target_type": "numeric"},
                        reason=f"Convert '{col}' from string to numeric",
                    ))
        
        return operations
    
    def _apply_cleaning_operations(
        self,
        data: pd.DataFrame,
        operations: List[CleaningOperation],
    ) -> Tuple[pd.DataFrame, CleaningResult]:
        """Apply the planned cleaning operations.
        
        Args:
            data: DataFrame to clean
            operations: List of operations to apply
            
        Returns:
            Tuple of (cleaned_data, cleaning_result)
        """
        original_shape = data.shape
        cleaned_data = data.copy()
        applied_operations = []
        rows_removed = 0
        columns_removed = 0
        null_values_filled = 0
        duplicates_removed = 0
        cleaning_code_lines = ["# Data Cleaning Code", "import pandas as pd", "import numpy as np", ""]
        
        for operation in operations:
            try:
                if operation.operation_type == "remove_column":
                    if operation.column in cleaned_data.columns:
                        cleaned_data = cleaned_data.drop(columns=[operation.column])
                        columns_removed += 1
                        cleaning_code_lines.append(f"data = data.drop(columns=['{operation.column}'])  # {operation.reason}")
                        applied_operations.append(operation)
                
                elif operation.operation_type == "remove_duplicates":
                    initial_rows = len(cleaned_data)
                    cleaned_data = cleaned_data.drop_duplicates()
                    duplicates_removed = initial_rows - len(cleaned_data)
                    rows_removed += duplicates_removed
                    cleaning_code_lines.append(f"data = data.drop_duplicates()  # {operation.reason}")
                    applied_operations.append(operation)
                
                elif operation.operation_type == "impute_numeric":
                    if operation.column in cleaned_data.columns:
                        null_count_before = cleaned_data[operation.column].isnull().sum()
                        strategy = operation.parameters.get("strategy", "median")
                        
                        if strategy == "median":
                            fill_value = cleaned_data[operation.column].median()
                        elif strategy == "mean":
                            fill_value = cleaned_data[operation.column].mean()
                        else:
                            fill_value = 0
                        
                        cleaned_data[operation.column] = cleaned_data[operation.column].fillna(fill_value)
                        null_values_filled += null_count_before
                        cleaning_code_lines.append(f"data['{operation.column}'] = data['{operation.column}'].fillna(data['{operation.column}'].{strategy}())  # {operation.reason}")
                        applied_operations.append(operation)
                
                elif operation.operation_type == "impute_categorical":
                    if operation.column in cleaned_data.columns:
                        null_count_before = cleaned_data[operation.column].isnull().sum()
                        mode_value = cleaned_data[operation.column].mode()
                        
                        if len(mode_value) > 0:
                            fill_value = mode_value.iloc[0]
                            cleaned_data[operation.column] = cleaned_data[operation.column].fillna(fill_value)
                            null_values_filled += null_count_before
                            cleaning_code_lines.append(f"data['{operation.column}'] = data['{operation.column}'].fillna(data['{operation.column}'].mode().iloc[0])  # {operation.reason}")
                            applied_operations.append(operation)
                
                elif operation.operation_type == "handle_outliers":
                    if operation.column in cleaned_data.columns:
                        method = operation.parameters.get("method", "clip")
                        
                        if method == "clip":
                            Q1 = cleaned_data[operation.column].quantile(0.25)
                            Q3 = cleaned_data[operation.column].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            cleaned_data[operation.column] = cleaned_data[operation.column].clip(lower_bound, upper_bound)
                            cleaning_code_lines.append(f"# Clip outliers in '{operation.column}'")
                            cleaning_code_lines.append(f"Q1 = data['{operation.column}'].quantile(0.25)")
                            cleaning_code_lines.append(f"Q3 = data['{operation.column}'].quantile(0.75)")
                            cleaning_code_lines.append(f"IQR = Q3 - Q1")
                            cleaning_code_lines.append(f"data['{operation.column}'] = data['{operation.column}'].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)")
                            applied_operations.append(operation)
                
                elif operation.operation_type == "convert_type":
                    if operation.column in cleaned_data.columns:
                        target_type = operation.parameters.get("target_type")
                        
                        if target_type == "numeric":
                            cleaned_data[operation.column] = pd.to_numeric(cleaned_data[operation.column], errors='coerce')
                            cleaning_code_lines.append(f"data['{operation.column}'] = pd.to_numeric(data['{operation.column}'], errors='coerce')  # {operation.reason}")
                            applied_operations.append(operation)
                
            except Exception as e:
                self.logger.warning(f"Failed to apply operation {operation.operation_type} on {operation.column}: {e}")
                continue
        
        final_shape = cleaned_data.shape
        cleaning_code = "\n".join(cleaning_code_lines)
        
        cleaning_result = CleaningResult(
            operations_applied=applied_operations,
            rows_removed=rows_removed,
            columns_removed=columns_removed,
            null_values_filled=null_values_filled,
            duplicates_removed=duplicates_removed,
            original_shape=original_shape,
            final_shape=final_shape,
            cleaning_code=cleaning_code,
        )
        
        return cleaned_data, cleaning_result
    
    def _is_numeric_string(self, series: pd.Series) -> bool:
        """Check if a series contains numeric values stored as strings.
        
        Args:
            series: Series to check
            
        Returns:
            True if series appears to contain numeric strings
        """
        try:
            # Try to convert a sample to numeric
            sample = series.head(50).astype(str)
            numeric_count = 0
            
            for value in sample:
                try:
                    float(value.replace(',', '').replace('$', '').replace('%', ''))
                    numeric_count += 1
                except (ValueError, AttributeError):
                    continue
            
            # If more than 80% of values can be converted, consider it numeric
            return (numeric_count / len(sample)) > 0.8
            
        except Exception:
            return False
    
    def _generate_suggestions(
        self,
        cleaning_result: CleaningResult,
        profile_summary: Dict[str, Any],
    ) -> List[str]:
        """Generate suggestions based on cleaning results.
        
        Args:
            cleaning_result: Results of cleaning operations
            profile_summary: Original data profile
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if cleaning_result.columns_removed > 0:
            suggestions.append(
                f"Removed {cleaning_result.columns_removed} columns. "
                f"Consider reviewing if any important features were lost."
            )
        
        if cleaning_result.rows_removed > 0:
            suggestions.append(
                f"Removed {cleaning_result.rows_removed} rows. "
                f"Ensure remaining data is still representative."
            )
        
        if cleaning_result.null_values_filled > 0:
            suggestions.append(
                f"Filled {cleaning_result.null_values_filled} null values. "
                f"Consider impact on model performance."
            )
        
        # Check remaining data quality
        final_size = cleaning_result.final_shape[0] * cleaning_result.final_shape[1]
        original_size = cleaning_result.original_shape[0] * cleaning_result.original_shape[1]
        reduction_pct = ((original_size - final_size) / original_size) * 100
        
        if reduction_pct > 30:
            suggestions.append(
                f"Significant data reduction ({reduction_pct:.1f}%). "
                f"Consider more conservative cleaning approaches."
            )
        
        return suggestions
    
    def _generate_warnings(self, cleaning_result: CleaningResult) -> List[str]:
        """Generate warnings for significant cleaning changes.
        
        Args:
            cleaning_result: Results of cleaning operations
            
        Returns:
            List of warning strings
        """
        warnings = []
        
        if cleaning_result.rows_removed > cleaning_result.original_shape[0] * 0.2:
            warnings.append(f"Removed {cleaning_result.rows_removed} rows (>20% of data)")
        
        if cleaning_result.columns_removed > 5:
            warnings.append(f"Removed {cleaning_result.columns_removed} columns")
        
        if cleaning_result.null_values_filled > cleaning_result.original_shape[0] * 0.1:
            warnings.append(f"Filled {cleaning_result.null_values_filled} null values (>10% of rows)")
        
        return warnings
    
    def _store_cleaning_memory(self, cleaning_result: CleaningResult, context: AgentContext) -> None:
        """Store cleaning results in memory for future reference.
        
        Args:
            cleaning_result: Cleaning results to store
            context: Current pipeline context
        """
        try:
            # Store detailed cleaning log
            cleaning_data = {
                "original_shape": cleaning_result.original_shape,
                "final_shape": cleaning_result.final_shape,
                "operations_count": len(cleaning_result.operations_applied),
                "rows_removed": cleaning_result.rows_removed,
                "columns_removed": cleaning_result.columns_removed,
                "null_values_filled": cleaning_result.null_values_filled,
                "operations": [op.dict() for op in cleaning_result.operations_applied],
                "cleaning_code": cleaning_result.cleaning_code,
            }
            
            memory_client.store_symbolic(
                table_name="data_cleaning_log",
                data=cleaning_data,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
            # Create semantic description
            description = f"""
            Data cleaning completed:
            - Shape changed from {cleaning_result.original_shape} to {cleaning_result.final_shape}
            - Applied {len(cleaning_result.operations_applied)} operations
            - Removed {cleaning_result.rows_removed} rows and {cleaning_result.columns_removed} columns
            - Filled {cleaning_result.null_values_filled} null values
            - Operations: {', '.join([op.operation_type for op in cleaning_result.operations_applied[:5]])}
            """
            
            memory_client.store_vector(
                content=description.strip(),
                tags=["data_cleaning", "preprocessing"],
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
                metadata={"operations_count": len(cleaning_result.operations_applied)},
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store cleaning memory: {e}")
    
    def _get_llm_prompt(self, context: AgentContext, feedback: Optional[UserFeedback] = None) -> str:
        """Generate LLM prompt for data cleaning insights.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            Formatted prompt string
        """
        profile_summary = context.profile_summary
        target = context.target
        
        prompt = f"""
        You are a data scientist reviewing data cleaning operations. Provide insights and recommendations.

        Original Dataset:
        - Shape: {profile_summary['dataset_shape']}
        - Target: {target or 'Not specified'}
        - Null percentage: {profile_summary['overall_null_percentage']:.1f}%

        Data Quality Issues Found:
        - High null columns: {len(profile_summary.get('high_null_columns', []))}
        - Constant columns: {len(profile_summary.get('constant_columns', []))}
        - Duplicate rows: {profile_summary.get('duplicate_rows', 0)}
        - Potential identifiers: {len(profile_summary.get('potential_identifiers', []))}

        Please provide:
        1. Assessment of the cleaning approach
        2. Potential risks or concerns with the cleaning operations
        3. Alternative cleaning strategies to consider
        4. Impact on model performance expectations
        5. Recommendations for next steps

        Focus on practical, actionable advice.
        """
        
        return prompt.strip()
    
    def _apply_overrides(
        self,
        context: AgentContext,
        overrides: Dict[str, Any],
    ) -> AgentContext:
        """Apply user overrides to the cleaning process.
        
        Args:
            context: Current context
            overrides: User overrides to apply
            
        Returns:
            Updated context
        """
        # User can override specific cleaning decisions
        if "skip_operations" in overrides:
            skip_ops = overrides["skip_operations"]
            self.logger.info(f"User requested to skip operations: {skip_ops}")
            # This would be handled in the planning phase
        
        if "custom_cleaning_code" in overrides:
            custom_code = overrides["custom_cleaning_code"]
            if custom_code and context.data is not None:
                try:
                    # Execute custom cleaning code (in a safe environment)
                    # For security, this should be sandboxed in production
                    exec(custom_code, {"data": context.data, "pd": pd, "np": np})
                    self.logger.info("Applied custom cleaning code")
                except Exception as e:
                    self.logger.error(f"Failed to execute custom cleaning code: {e}")
        
        return context