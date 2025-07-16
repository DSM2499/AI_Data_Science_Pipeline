"""Feature Engineering Agent - Creates new features using statistical logic and LLM insights."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from agents.base import AgentContext, AgentResult, LLMCapableAgent, UserFeedback
from memory.memory_client import memory_client
from utils.exceptions import FeatureEngineeringError, ValidationError


class EngineeringOperation(BaseModel):
    """Represents a single feature engineering operation."""
    
    operation_type: str
    feature_name: str
    source_features: List[str]
    parameters: Dict[str, Any]
    rationale: str
    success: bool = False
    error_message: Optional[str] = None


class FeatureEngineeringResult(BaseModel):
    """Result of feature engineering process."""
    
    operations_applied: List[EngineeringOperation]
    new_features: List[str]
    original_feature_count: int
    final_feature_count: int
    features_added: int
    engineering_code: str


class FeatureEngineeringAgent(LLMCapableAgent):
    """Agent responsible for creating new features using AI assistance and statistical methods."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the feature engineering agent.
        
        Args:
            use_llm: Whether to use LLM for generating feature suggestions
        """
        super().__init__("FeatureEngineeringAgent", use_llm)
        
        # Engineering parameters
        self.max_new_features_ratio = 0.5  # Don't create more than 50% new features
        self.correlation_threshold = 0.05  # Minimum correlation with target for new features
        self.polynomial_degree = 2  # Degree for polynomial features
    
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute feature engineering process.
        
        Args:
            context: Pipeline context containing selected features
            feedback: Optional user feedback for refinement
            
        Returns:
            AgentResult with engineered features and explanations
        """
        try:
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            # Log data types for debugging
            if context.data is not None:
                numeric_cols = context.data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = context.data.select_dtypes(include=['object', 'category']).columns.tolist()
                self.logger.info(f"Data type analysis - Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
                if categorical_cols:
                    self.logger.info(f"Categorical columns detected: {categorical_cols[:5]}...")  # Log first 5
            
            # Store pre-engineering data for potential rollback
            if context.pre_engineering_data is None:
                context.pre_engineering_data = context.data.copy()
                self.logger.info("Stored pre-engineering data snapshot")
            
            # Handle user feedback
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            data = context.data.copy()
            target = context.target
            
            self.logger.info(f"Starting feature engineering for dataset with {data.shape[1]} features")
            
            # Preprocess categorical data by encoding it
            data = self._preprocess_categorical_data(data, target)
            
            # Get LLM suggestions for feature engineering (include user feedback in prompt)
            llm_suggestions = []
            if self.use_llm:
                llm_suggestions = self._get_llm_feature_suggestions(data, target, context, feedback)
            
            # Plan engineering operations (considering user overrides)
            planned_operations = self._plan_engineering_operations(data, target, llm_suggestions, context)
            
            # Apply engineering operations (with user overrides)
            engineered_data, engineering_result = self._apply_engineering_operations(
                data, planned_operations, target, context
            )
            
            # Apply custom engineering code if provided
            if context.user_overrides.get('custom_engineering_code'):
                custom_code = context.user_overrides['custom_engineering_code']
                try:
                    # Execute custom feature engineering code safely
                    local_vars = {"data": engineered_data, "pd": pd, "np": np}
                    exec(custom_code, {"__builtins__": {}}, local_vars)
                    engineered_data = local_vars.get("data", engineered_data)
                    self.logger.info("Applied custom feature engineering code")
                    
                    # Add a note about custom code in the engineering result
                    engineering_result.engineering_code += f"\n\n# User Custom Code:\n{custom_code}"
                    
                except Exception as e:
                    self.logger.error(f"Failed to execute custom engineering code: {e}")
                    # Continue without the custom code
            
            # Generate LLM insights about engineered features
            llm_insights = ""
            if self.use_llm:
                llm_insights = self._generate_engineering_insights(engineering_result, context)
            
            # Update context
            context.enriched_data = engineered_data
            context.data = engineered_data  # Update main data reference
            context.llm_insights[self.name] = llm_insights
            
            # Store engineering results in memory
            self._store_engineering_memory(engineering_result, context)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(engineering_result)
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Applied {len(engineering_result.operations_applied)} engineering operations",
                    f"Created {engineering_result.features_added} new features",
                    f"Final dataset shape: {engineered_data.shape}",
                    f"New features: {', '.join(engineering_result.new_features[:5])}{'...' if len(engineering_result.new_features) > 5 else ''}",
                ],
                metrics={
                    "original_features": engineering_result.original_feature_count,
                    "final_features": engineering_result.final_feature_count,
                    "features_added": engineering_result.features_added,
                    "operations_applied": len(engineering_result.operations_applied),
                    "success_rate": sum(1 for op in engineering_result.operations_applied if op.success) / len(engineering_result.operations_applied) if engineering_result.operations_applied else 0,
                },
                artifacts={
                    "engineering_result": engineering_result.dict(),
                    "engineering_code": engineering_result.engineering_code,
                    "llm_insights": llm_insights,
                },
                suggestions=suggestions,
                requires_approval=engineering_result.features_added > 10,
            )
            
            # Add warnings for excessive feature creation
            if engineering_result.features_added > engineering_result.original_feature_count * 0.3:
                result.warnings = [
                    f"Created many new features ({engineering_result.features_added}). "
                    f"Consider feature selection to prevent overfitting."
                ]
                result.requires_approval = True
                result.user_message = (
                    f"Feature engineering created {engineering_result.features_added} new features. "
                    f"Please review before proceeding."
                )
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            error_message = f"Feature engineering failed: {e}"
            
            # Check for specific categorical data conversion errors
            if "Could not convert" in str(e) and "to numeric" in str(e):
                error_message = (
                    "Feature engineering failed due to categorical data conversion. "
                    "The system attempted to perform mathematical operations on non-numeric data. "
                    "This usually happens when the dataset contains categorical columns (like country names, text, etc.) "
                    "that cannot be used in mathematical calculations. "
                    "Try providing feedback to skip operations that don't work with your data type."
                )
                self.logger.error(f"Categorical data conversion error: {e}")
                
                # Add suggestions for user
                suggestions = [
                    "Skip polynomial features if you have categorical data",
                    "Consider using frequency encoding for categorical columns", 
                    "Focus on numeric columns for ratio and mathematical operations",
                    "Use natural language feedback like 'skip polynomial features' to avoid this error"
                ]
            else:
                self.logger.error(error_message)
                suggestions = []
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=error_message,
                logs=[f"Feature engineering failed: {e}"],
                suggestions=suggestions,
                user_message="Feature engineering encountered an error. Please review the suggestions below or provide feedback to modify the approach.",
                requires_approval=True  # Allow user to provide feedback to fix the issue
            )
            
            self._log_execution(result)
            return result
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate that the context contains required data.
        
        Args:
            context: Pipeline context to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValidationError: If validation fails
        """
        if context.data is None:
            raise ValidationError("No data found in context")
        
        if not isinstance(context.data, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame")
        
        if context.data.empty:
            raise ValidationError("DataFrame is empty")
        
        return True
    
    def _get_llm_feature_suggestions(
        self,
        data: pd.DataFrame,
        target: Optional[str],
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> List[str]:
        """Get LLM suggestions for feature engineering.
        
        Args:
            data: Current dataset
            target: Target variable name
            context: Pipeline context
            
        Returns:
            List of feature engineering suggestions
        """
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return []
            
            # Prepare column information
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            columns_info = {
                "numeric_columns": numeric_cols[:10],  # Limit for prompt size
                "categorical_columns": categorical_cols[:10],
                "total_features": len(data.columns),
                "data_shape": data.shape,
            }
            
            target_info = None
            task_type = None
            
            if target and target in data.columns:
                target_info = {
                    "name": target,
                    "type": str(data[target].dtype),
                    "unique_values": int(data[target].nunique()),
                    "null_percentage": float((data[target].isnull().sum() / len(data)) * 100),
                }
                
                task_type = "classification" if (data[target].dtype == 'object' or data[target].nunique() < 20) else "regression"
            
            # Include user feedback in the suggestions if provided
            feedback_context = ""
            if feedback and feedback.feedback_text:
                feedback_context = f"\n\nUser Feedback: {feedback.feedback_text}\n"
                if hasattr(feedback, 'additional_context') and feedback.additional_context:
                    intent = feedback.additional_context.get('intent', '')
                    if intent:
                        feedback_context += f"User Intent: {intent}\n"
            
            suggestions_text = llm_service.generate_feature_suggestions(
                columns_info=columns_info,
                target_info=target_info,
                task_type=task_type,
                max_tokens=800,
                temperature=0.7,
            )
            
            # If we have feedback, modify suggestions to incorporate it
            if feedback_context:
                enhanced_prompt = f"""
                {suggestions_text}
                
                {feedback_context}
                
                Please revise the above suggestions to specifically address the user feedback.
                Focus on what the user requested and adjust the feature engineering approach accordingly.
                """
                
                suggestions_text = llm_service.generate_simple_response(
                    prompt=enhanced_prompt,
                    system_message="You are a feature engineering expert. Adjust your suggestions based on user feedback.",
                    temperature=0.4
                )
            
            # Parse suggestions (simple implementation)
            suggestions = [
                line.strip() for line in suggestions_text.split('\n')
                if line.strip() and not line.startswith('#')
            ]
            
            return suggestions[:10]  # Limit number of suggestions
            
        except Exception as e:
            self.logger.warning(f"Failed to get LLM feature suggestions: {e}")
            return []
    
    def _plan_engineering_operations(
        self,
        data: pd.DataFrame,
        target: Optional[str],
        llm_suggestions: List[str],
        context: Optional[AgentContext] = None,
    ) -> List[EngineeringOperation]:
        """Plan feature engineering operations.
        
        Args:
            data: Current dataset
            target: Target variable name
            llm_suggestions: LLM-generated suggestions
            
        Returns:
            List of planned engineering operations
        """
        operations = []
        
        # Exclude target from feature engineering
        feature_columns = [col for col in data.columns if col != target]
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_columns:
            numeric_columns.remove(target)
        
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Ratio features for numeric columns
        if len(numeric_columns) >= 2:
            for i, col1 in enumerate(numeric_columns[:5]):  # Limit combinations
                for col2 in numeric_columns[i+1:6]:
                    if col1 != col2:
                        operations.append(EngineeringOperation(
                            operation_type="ratio",
                            feature_name=f"{col1}_to_{col2}_ratio",
                            source_features=[col1, col2],
                            parameters={"numerator": col1, "denominator": col2},
                            rationale=f"Ratio of {col1} to {col2} may capture important relationships",
                        ))
        
        # 2. Binning for numeric features
        for col in numeric_columns[:5]:  # Limit number of features
            operations.append(EngineeringOperation(
                operation_type="binning",
                feature_name=f"{col}_binned",
                source_features=[col],
                parameters={"bins": 5, "strategy": "quantile"},
                rationale=f"Binning {col} to capture non-linear patterns",
            ))
        
        # 3. Log transformation for skewed numeric features
        for col in numeric_columns[:5]:
            if data[col].min() > 0:  # Only for positive values
                skewness = abs(data[col].skew())
                if skewness > 1:  # Significantly skewed
                    operations.append(EngineeringOperation(
                        operation_type="log_transform",
                        feature_name=f"{col}_log",
                        source_features=[col],
                        parameters={"base": "natural"},
                        rationale=f"Log transform {col} to reduce skewness ({skewness:.2f})",
                    ))
        
        # 4. Polynomial features (limited)
        if len(numeric_columns) >= 2:
            # Select top 3 numeric features for polynomial combinations
            selected_numeric = numeric_columns[:3]
            operations.append(EngineeringOperation(
                operation_type="polynomial",
                feature_name="polynomial_features",
                source_features=selected_numeric,
                parameters={"degree": 2, "interaction_only": True},
                rationale=f"Create interaction terms for top numeric features",
            ))
        
        # 5. Frequency encoding for categorical features
        for col in categorical_columns:
            if col != target and data[col].nunique() > 2:
                operations.append(EngineeringOperation(
                    operation_type="frequency_encoding",
                    feature_name=f"{col}_frequency",
                    source_features=[col],
                    parameters={"normalize": True},
                    rationale=f"Frequency encoding for categorical {col}",
                ))
        
        # 6. Aggregation features
        if len(numeric_columns) >= 3:
            operations.append(EngineeringOperation(
                operation_type="aggregation",
                feature_name="numeric_sum",
                source_features=numeric_columns[:5],
                parameters={"operation": "sum"},
                rationale="Sum of key numeric features",
            ))
            
            operations.append(EngineeringOperation(
                operation_type="aggregation",
                feature_name="numeric_mean",
                source_features=numeric_columns[:5],
                parameters={"operation": "mean"},
                rationale="Mean of key numeric features",
            ))
        
        # Limit total operations to prevent feature explosion
        max_operations = max(5, int(len(feature_columns) * self.max_new_features_ratio))
        return operations[:max_operations]
    
    def _apply_engineering_operations(
        self,
        data: pd.DataFrame,
        operations: List[EngineeringOperation],
        target: Optional[str],
        context: Optional[AgentContext] = None,
    ) -> Tuple[pd.DataFrame, FeatureEngineeringResult]:
        """Apply feature engineering operations.
        
        Args:
            data: Original dataset
            operations: Operations to apply
            target: Target variable name
            context: Pipeline context (for user overrides)
            
        Returns:
            Tuple of (engineered_data, engineering_result)
        """
        # Filter out operations that user wants to skip
        skip_operations = []
        if context:
            skip_operations = context.user_overrides.get('skip_feature_operations', [])
            if skip_operations:
                self.logger.info(f"User requested to skip: {skip_operations}")
        
        # Filter operations based on user preferences
        filtered_operations = []
        for operation in operations:
            operation_label = f"{operation.operation_type}: {operation.feature_name}"
            if operation_label not in skip_operations:
                filtered_operations.append(operation)
            else:
                self.logger.info(f"Skipping operation: {operation_label}")
        
        engineered_data = data.copy()
        applied_operations = []
        new_features = []
        original_feature_count = len(data.columns)
        code_lines = ["# Feature Engineering Code", "import pandas as pd", "import numpy as np", ""]
        
        for operation in filtered_operations:
            try:
                if operation.operation_type == "ratio":
                    col1, col2 = operation.source_features
                    if col1 in engineered_data.columns and col2 in engineered_data.columns:
                        # Only create ratios for numeric columns - with extra validation
                        col1_numeric = pd.api.types.is_numeric_dtype(engineered_data[col1])
                        col2_numeric = pd.api.types.is_numeric_dtype(engineered_data[col2])
                        
                        if not (col1_numeric and col2_numeric):
                            operation.error_message = f"Cannot create ratio from non-numeric columns: {col1} ({engineered_data[col1].dtype}), {col2} ({engineered_data[col2].dtype})"
                            continue
                        
                        # Extra safety: try mathematical operations
                        try:
                            _ = engineered_data[col1] + 1
                            _ = engineered_data[col2] + 1
                        except (TypeError, ValueError) as e:
                            operation.error_message = f"Columns appear numeric but fail math test for ratio {col1}/{col2}: {e}"
                            continue
                        
                        try:
                            # Ensure both columns are actually numeric
                            num_col1 = pd.to_numeric(engineered_data[col1], errors='coerce').fillna(0)
                            num_col2 = pd.to_numeric(engineered_data[col2], errors='coerce').fillna(1)  # avoid division by 0
                            
                            # Avoid division by zero
                            denominator = num_col2.replace(0, 1e-8)  # Use small number instead of NaN
                            new_feature = num_col1 / denominator
                            
                            # Handle any remaining NaN or infinite values
                            new_feature = np.nan_to_num(new_feature, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            engineered_data[operation.feature_name] = new_feature
                            new_features.append(operation.feature_name)
                            operation.success = True
                        except (ValueError, TypeError) as e:
                            operation.error_message = f"Ratio calculation failed for {col1}/{col2}: {e}"
                        
                        code_lines.append(f"# Ratio calculation with safety checks")
                        code_lines.append(f"data['{operation.feature_name}'] = pd.to_numeric(data['{col1}'], errors='coerce').fillna(0) / pd.to_numeric(data['{col2}'], errors='coerce').fillna(1).replace(0, 1e-8)")
                        code_lines.append(f"data['{operation.feature_name}'] = np.nan_to_num(data['{operation.feature_name}'], nan=0.0, posinf=0.0, neginf=0.0)")
                
                elif operation.operation_type == "binning":
                    col = operation.source_features[0]
                    if col in engineered_data.columns:
                        # Only bin numeric columns
                        if not pd.api.types.is_numeric_dtype(engineered_data[col]):
                            operation.error_message = f"Cannot bin non-numeric column: {col} (dtype: {engineered_data[col].dtype})"
                            continue
                        
                        bins = operation.parameters.get("bins", 5)
                        try:
                            # Check if column has enough unique values for binning
                            unique_values = engineered_data[col].nunique()
                            if unique_values < bins:
                                bins = max(2, unique_values)
                            
                            binned = pd.cut(engineered_data[col], bins=bins, labels=False, duplicates='drop')
                            engineered_data[operation.feature_name] = binned.fillna(-1).astype(int)
                            new_features.append(operation.feature_name)
                            operation.success = True
                            
                            code_lines.append(f"# Binning {col} (ensuring numeric type)")
                            code_lines.append(f"if pd.api.types.is_numeric_dtype(data['{col}']):")
                            code_lines.append(f"    data['{operation.feature_name}'] = pd.cut(data['{col}'], bins={bins}, labels=False, duplicates='drop')")
                            code_lines.append(f"    data['{operation.feature_name}'] = data['{operation.feature_name}'].fillna(-1).astype(int)")
                        except (ValueError, TypeError) as e:
                            operation.error_message = f"Binning failed for {col}: {e}"
                
                elif operation.operation_type == "log_transform":
                    col = operation.source_features[0]
                    if col in engineered_data.columns:
                        # Only log transform numeric columns
                        if not pd.api.types.is_numeric_dtype(engineered_data[col]):
                            operation.error_message = f"Cannot log transform non-numeric column: {col} (dtype: {engineered_data[col].dtype})"
                            continue
                        
                        try:
                            # Add small constant to handle zeros and negatives
                            log_values = np.log1p(engineered_data[col].clip(lower=0))
                            engineered_data[operation.feature_name] = log_values
                            new_features.append(operation.feature_name)
                            operation.success = True
                            
                            code_lines.append(f"# Log transform {col} (ensuring numeric type)")
                            code_lines.append(f"if pd.api.types.is_numeric_dtype(data['{col}']):")
                            code_lines.append(f"    data['{operation.feature_name}'] = np.log1p(data['{col}'].clip(lower=0))")
                        except (ValueError, TypeError) as e:
                            operation.error_message = f"Log transform failed for {col}: {e}"
                
                elif operation.operation_type == "polynomial":
                    source_cols = [col for col in operation.source_features if col in engineered_data.columns]
                    # Only use numeric columns for polynomial features and double-check they're actually numeric
                    numeric_source_cols = []
                    for col in source_cols:
                        if pd.api.types.is_numeric_dtype(engineered_data[col]):
                            # Extra validation - try to perform a mathematical operation
                            try:
                                _ = engineered_data[col] + 1  # Test if we can do math
                                numeric_source_cols.append(col)
                            except (TypeError, ValueError) as e:
                                self.logger.warning(f"Column {col} appears numeric but fails math test: {e}")
                                # Try to force conversion one more time
                                try:
                                    engineered_data[col] = pd.to_numeric(engineered_data[col], errors='coerce').fillna(0)
                                    _ = engineered_data[col] + 1  # Test again
                                    numeric_source_cols.append(col)
                                except:
                                    self.logger.warning(f"Could not salvage column {col} for polynomial features")
                    
                    if len(numeric_source_cols) >= 2:
                        try:
                            poly = PolynomialFeatures(
                                degree=operation.parameters.get("degree", 2),
                                interaction_only=operation.parameters.get("interaction_only", True),
                                include_bias=False
                            )
                            
                            # Extra safety: ensure all data is float and handle any infinities
                            X_subset = engineered_data[numeric_source_cols].copy()
                            for col in numeric_source_cols:
                                X_subset[col] = pd.to_numeric(X_subset[col], errors='coerce').fillna(0)
                                # Replace infinities
                                X_subset[col] = X_subset[col].replace([np.inf, -np.inf], 0)
                            
                            poly_features = poly.fit_transform(X_subset)
                            feature_names = poly.get_feature_names_out(numeric_source_cols)
                            
                            # Add only interaction terms (skip original features)
                            for i, name in enumerate(feature_names):
                                if len(name.split(' ')) > 1:  # Interaction term
                                    clean_name = name.replace(' ', '_').replace('^', '_pow_')
                                    # Extra safety for new features
                                    new_feature = poly_features[:, i]
                                    new_feature = np.nan_to_num(new_feature, nan=0.0, posinf=0.0, neginf=0.0)
                                    engineered_data[f"poly_{clean_name}"] = new_feature
                                    new_features.append(f"poly_{clean_name}")
                            
                            operation.success = True
                            code_lines.append(f"# Polynomial features for {numeric_source_cols}")
                            code_lines.append(f"from sklearn.preprocessing import PolynomialFeatures")
                            
                        except Exception as e:
                            operation.error_message = f"Polynomial features failed: {e}"
                
                elif operation.operation_type == "frequency_encoding":
                    col = operation.source_features[0]
                    if col in engineered_data.columns:
                        freq_map = engineered_data[col].value_counts().to_dict()
                        engineered_data[operation.feature_name] = engineered_data[col].map(freq_map).fillna(0)
                        
                        if operation.parameters.get("normalize", False):
                            total_count = len(engineered_data)
                            engineered_data[operation.feature_name] = engineered_data[operation.feature_name] / total_count
                        
                        new_features.append(operation.feature_name)
                        operation.success = True
                        
                        code_lines.append(f"freq_map = data['{col}'].value_counts().to_dict()")
                        code_lines.append(f"data['{operation.feature_name}'] = data['{col}'].map(freq_map).fillna(0)")
                
                elif operation.operation_type == "aggregation":
                    source_cols = [col for col in operation.source_features if col in engineered_data.columns]
                    # Only use numeric columns for aggregation
                    numeric_source_cols = [col for col in source_cols if pd.api.types.is_numeric_dtype(engineered_data[col])]
                    if numeric_source_cols:
                        agg_op = operation.parameters.get("operation", "sum")
                        
                        try:
                            if agg_op == "sum":
                                engineered_data[operation.feature_name] = engineered_data[numeric_source_cols].sum(axis=1)
                            elif agg_op == "mean":
                                engineered_data[operation.feature_name] = engineered_data[numeric_source_cols].mean(axis=1)
                            elif agg_op == "std":
                                engineered_data[operation.feature_name] = engineered_data[numeric_source_cols].std(axis=1).fillna(0)
                            
                            new_features.append(operation.feature_name)
                            operation.success = True
                            
                            code_lines.append(f"# Aggregation of numeric columns: {numeric_source_cols}")
                            code_lines.append(f"data['{operation.feature_name}'] = data[{numeric_source_cols}].{agg_op}(axis=1)")
                        except (ValueError, TypeError) as e:
                            operation.error_message = f"Aggregation failed: {e}"
                
                applied_operations.append(operation)
                
            except Exception as e:
                operation.error_message = str(e)
                applied_operations.append(operation)
                self.logger.warning(f"Feature engineering operation failed: {operation.operation_type} - {e}")
        
        final_feature_count = len(engineered_data.columns)
        features_added = final_feature_count - original_feature_count
        engineering_code = "\n".join(code_lines)
        
        engineering_result = FeatureEngineeringResult(
            operations_applied=applied_operations,
            new_features=new_features,
            original_feature_count=original_feature_count,
            final_feature_count=final_feature_count,
            features_added=features_added,
            engineering_code=engineering_code,
        )
        
        return engineered_data, engineering_result
    
    def _generate_engineering_insights(
        self,
        engineering_result: FeatureEngineeringResult,
        context: AgentContext,
    ) -> str:
        """Generate LLM insights about feature engineering results.
        
        Args:
            engineering_result: Feature engineering results
            context: Pipeline context
            
        Returns:
            Generated insights text
        """
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "LLM service not available for feature engineering insights"
            
            successful_ops = [op for op in engineering_result.operations_applied if op.success]
            failed_ops = [op for op in engineering_result.operations_applied if not op.success]
            
            prompt = f"""
            Analyze these feature engineering results and provide insights:
            
            Original features: {engineering_result.original_feature_count}
            Features added: {engineering_result.features_added}
            Final features: {engineering_result.final_feature_count}
            
            Successful operations: {len(successful_ops)}
            Failed operations: {len(failed_ops)}
            
            New features created:
            {engineering_result.new_features[:10]}
            
            Operation types used:
            {list(set(op.operation_type for op in successful_ops))}
            
            Target variable: {context.target or 'None (unsupervised)'}
            
            Please provide:
            1. Assessment of the feature engineering approach
            2. Potential value of the new features
            3. Recommendations for next steps
            4. Any concerns about feature quality or quantity
            
            Be specific and actionable.
            """
            
            insights = llm_service.generate_simple_response(
                prompt=prompt,
                system_message="You are a feature engineering expert analyzing feature creation results.",
                max_tokens=600,
                temperature=0.7,
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate engineering insights: {e}")
            return f"Could not generate insights: {e}"
    
    def _generate_suggestions(self, engineering_result: FeatureEngineeringResult) -> List[str]:
        """Generate suggestions based on feature engineering results.
        
        Args:
            engineering_result: Engineering results
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        successful_ops = [op for op in engineering_result.operations_applied if op.success]
        failed_ops = [op for op in engineering_result.operations_applied if not op.success]
        
        if len(failed_ops) > 0:
            suggestions.append(
                f"{len(failed_ops)} feature engineering operations failed. "
                f"Review data quality and operation parameters."
            )
        
        if engineering_result.features_added > 20:
            suggestions.append(
                f"Many features created ({engineering_result.features_added}). "
                f"Consider feature selection to prevent overfitting."
            )
        
        if engineering_result.features_added == 0:
            suggestions.append(
                "No new features were created. Consider more aggressive feature engineering or domain expertise."
            )
        
        if len(set(op.operation_type for op in successful_ops)) < 3:
            suggestions.append(
                "Limited variety in feature engineering operations. Consider exploring more techniques."
            )
        
        return suggestions
    
    def _store_engineering_memory(
        self,
        engineering_result: FeatureEngineeringResult,
        context: AgentContext,
    ) -> None:
        """Store feature engineering results in memory.
        
        Args:
            engineering_result: Engineering results to store
            context: Pipeline context
        """
        try:
            # Store each successful feature for future reference
            for operation in engineering_result.operations_applied:
                if operation.success:
                    performance_metrics = {"created": True, "operation_type": operation.operation_type}
                    dataset_context = {
                        "feature_count": engineering_result.original_feature_count,
                        "target": context.target,
                        "task_type": "classification" if context.target and context.data[context.target].nunique() < 20 else "regression",
                    }
                    
                    memory_client.store_feature_success(
                        feature_name=operation.feature_name,
                        performance_metrics=performance_metrics,
                        dataset_context=dataset_context,
                        source_agent=self.name,
                        project_id=context.user_preferences.get("project_id", "default"),
                    )
            
            # Store overall engineering summary
            engineering_data = {
                "original_feature_count": engineering_result.original_feature_count,
                "final_feature_count": engineering_result.final_feature_count,
                "features_added": engineering_result.features_added,
                "operations_applied": len(engineering_result.operations_applied),
                "success_rate": sum(1 for op in engineering_result.operations_applied if op.success) / len(engineering_result.operations_applied) if engineering_result.operations_applied else 0,
                "new_features": engineering_result.new_features,
                "engineering_code": engineering_result.engineering_code,
            }
            
            memory_client.store_symbolic(
                table_name="feature_engineering_log",
                data=engineering_data,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store engineering memory: {e}")
    
    def _get_llm_prompt(self, context: AgentContext, feedback: Optional[UserFeedback] = None) -> str:
        """Generate LLM prompt for feature engineering insights.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            Formatted prompt string
        """
        data = context.data
        target = context.target
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        prompt = f"""
        You are a feature engineering expert working with this dataset:

        Dataset Information:
        - Shape: {data.shape}
        - Target variable: {target or 'Not specified (unsupervised)'}
        - Numeric features: {len(numeric_cols)} ({numeric_cols[:5]}...)
        - Categorical features: {len(categorical_cols)} ({categorical_cols[:5]}...)

        Please suggest specific feature engineering techniques:
        1. Mathematical transformations (ratios, logs, polynomials)
        2. Binning and discretization strategies
        3. Interaction features that might be valuable
        4. Domain-specific feature ideas (if you can infer the domain)
        5. Aggregation features

        Focus on techniques that are likely to improve model performance for this type of data.
        Provide specific implementation suggestions.
        """
        
        return prompt.strip()
    
    def _apply_overrides(
        self,
        context: AgentContext,
        overrides: Dict[str, Any],
    ) -> AgentContext:
        """Apply user overrides to feature engineering.
        
        Args:
            context: Current context
            overrides: User overrides to apply
            
        Returns:
            Updated context
        """
        # Store user preferences in context for use during re-run
        context.user_overrides.update(overrides)
        
        # Apply parameter overrides
        if "max_new_features_ratio" in overrides:
            self.max_new_features_ratio = overrides["max_new_features_ratio"]
            self.logger.info(f"Updated max_new_features_ratio to {self.max_new_features_ratio}")
        
        if "correlation_threshold" in overrides:
            self.correlation_threshold = overrides["correlation_threshold"]
            self.logger.info(f"Updated correlation_threshold to {self.correlation_threshold}")
        
        # Handle skipped feature operations
        if "skip_feature_operations" in overrides:
            skip_ops = overrides["skip_feature_operations"]
            context.user_overrides["skip_feature_operations"] = skip_ops
            self.logger.info(f"User requested to skip feature operations: {skip_ops}")
        
        # Handle custom feature engineering code
        if "custom_engineering_code" in overrides:
            custom_code = overrides["custom_engineering_code"]
            if custom_code and context.data is not None:
                try:
                    # Store the custom code for execution during re-run
                    context.user_overrides["custom_engineering_code"] = custom_code
                    self.logger.info("Stored custom feature engineering code for re-run")
                except Exception as e:
                    self.logger.error(f"Error processing custom engineering code: {e}")
        
        # Reset the data to the state before feature engineering if we're re-running
        if context.pre_engineering_data is not None:
            context.data = context.pre_engineering_data.copy()
            self.logger.info("Reset data to pre-engineering state for re-run with feedback")
        
        return context
    
    def _preprocess_categorical_data(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None
    ) -> pd.DataFrame:
        """Preprocess categorical data by encoding it to numeric format.
        
        Args:
            data: Original dataset
            target: Target variable name
            
        Returns:
            Dataset with categorical columns encoded
        """
        processed_data = data.copy()
        
        # Get categorical columns (excluding target)
        categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also check for columns with string/mixed data types that might not be detected
        for col in processed_data.columns:
            if col != target and col not in categorical_columns:
                # Check if column contains strings that can't be converted to numeric
                try:
                    pd.to_numeric(processed_data[col], errors='raise')
                except (ValueError, TypeError):
                    # This column has non-numeric data, treat as categorical
                    categorical_columns.append(col)
                    self.logger.info(f"Detected hidden categorical column: {col} (dtype: {processed_data[col].dtype})")
        
        if target and target in categorical_columns:
            categorical_columns.remove(target)  # Don't encode target here
        
        if not categorical_columns:
            self.logger.info("No categorical columns to encode")
            return processed_data
        
        self.logger.info(f"Encoding {len(categorical_columns)} categorical columns: {categorical_columns}")
        
        for col in categorical_columns:
            try:
                # Handle missing values first
                if processed_data[col].isnull().any():
                    processed_data[col] = processed_data[col].fillna('__MISSING__')
                    self.logger.info(f"Filled missing values in {col} with '__MISSING__'")
                
                # Convert to string to ensure consistent handling
                processed_data[col] = processed_data[col].astype(str)
                
                # Get unique value count
                unique_count = processed_data[col].nunique()
                
                if unique_count <= 1:
                    # Constant column, can be dropped or converted to 0
                    processed_data[col] = 0
                    self.logger.info(f"Converted constant categorical column {col} to 0")
                    
                elif unique_count == 2:
                    # Binary categorical - use simple binary encoding
                    categories = processed_data[col].unique()
                    processed_data[col] = processed_data[col].map({categories[0]: 0, categories[1]: 1})
                    self.logger.info(f"Binary encoded column {col}: {categories[0]}->0, {categories[1]}->1")
                    
                elif unique_count <= 10:
                    # Low cardinality - use label encoding
                    processed_data[col] = pd.Categorical(processed_data[col]).codes
                    self.logger.info(f"Label encoded column {col} ({unique_count} categories)")
                    
                else:
                    # High cardinality - use frequency encoding
                    freq_map = processed_data[col].value_counts().to_dict()
                    processed_data[col] = processed_data[col].map(freq_map).fillna(0)
                    self.logger.info(f"Frequency encoded column {col} ({unique_count} categories)")
                
                # Ensure the result is numeric and handle any remaining issues
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
                
                # Final validation - make sure it's actually numeric
                if not pd.api.types.is_numeric_dtype(processed_data[col]):
                    self.logger.warning(f"Column {col} still not numeric after encoding, forcing to int")
                    processed_data[col] = processed_data[col].astype(str).str.extract('(\d+)').fillna(0).astype(int)
                
            except Exception as e:
                self.logger.warning(f"Failed to encode categorical column {col}: {e}")
                # Fall back to hash encoding for problematic columns
                try:
                    # Convert to hash codes as last resort
                    processed_data[col] = processed_data[col].astype(str).apply(lambda x: abs(hash(x)) % 10000)
                    self.logger.info(f"Applied hash encoding to problematic column {col}")
                except Exception as e2:
                    self.logger.error(f"Failed to encode {col} with fallback method: {e2}")
                    # Last resort: drop the column
                    processed_data = processed_data.drop(columns=[col])
                    self.logger.warning(f"Dropped problematic column {col}")
        
        # Final verification - ensure all columns are numeric (except target)
        non_numeric_cols = processed_data.select_dtypes(exclude=[np.number]).columns.tolist()
        if target and target in non_numeric_cols:
            non_numeric_cols.remove(target)  # Target can remain categorical
            
        if non_numeric_cols:
            self.logger.warning(f"Some columns still not numeric after encoding: {non_numeric_cols}")
            # Force convert remaining non-numeric columns
            for col in non_numeric_cols:
                if col != target:
                    try:
                        processed_data[col] = processed_data[col].astype(str).str.extract('(\d+)').fillna(0).astype(int)
                        self.logger.info(f"Force converted {col} to numeric")
                    except:
                        processed_data = processed_data.drop(columns=[col])
                        self.logger.warning(f"Dropped unconvertible column {col}")
        
        self.logger.info(f"Categorical preprocessing complete. Data shape: {processed_data.shape}")
        self.logger.info(f"All columns now numeric (except target): {processed_data.select_dtypes(include=[np.number]).shape[1]} numeric columns")
        return processed_data