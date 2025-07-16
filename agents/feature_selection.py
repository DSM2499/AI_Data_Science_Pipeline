"""Feature Selection Agent - Reduces feature space by removing low-signal and redundant features."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder

from agents.base import AgentContext, AgentResult, LLMCapableAgent, UserFeedback
from memory.memory_client import memory_client
from utils.exceptions import FeatureSelectionError, ValidationError


class FeatureSelectionResult(BaseModel):
    """Result of feature selection process."""
    
    selected_features: List[str]
    removed_features: List[str]
    selection_methods: List[str]
    feature_scores: Dict[str, float]
    selection_rationale: Dict[str, str]
    original_feature_count: int
    final_feature_count: int
    reduction_percentage: float


class FeatureSelectionAgent(LLMCapableAgent):
    """Agent responsible for intelligent feature selection and dimensionality reduction."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the feature selection agent.
        
        Args:
            use_llm: Whether to use LLM for generating insights and rationale
        """
        super().__init__("FeatureSelectionAgent", use_llm)
        
        # Selection thresholds
        self.variance_threshold = 0.01  # Remove low-variance features
        self.correlation_threshold = 0.95  # Remove highly correlated features
        self.importance_percentile = 70  # Keep top 70% of features by importance
        self.max_features_ratio = 0.8  # Keep at most 80% of original features
    
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute feature selection process.
        
        Args:
            context: Pipeline context containing cleaned data
            feedback: Optional user feedback for refinement
            
        Returns:
            AgentResult with selected features and selection rationale
        """
        try:
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            # Handle user feedback
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            data = context.data.copy()
            target = context.target
            
            self.logger.info(f"Starting feature selection for dataset with {data.shape[1]} features")
            
            # Separate features and target
            if target and target in data.columns:
                X = data.drop(columns=[target])
                y = data[target]
            else:
                X = data
                y = None
                self.logger.warning("No target variable specified - using unsupervised methods")
            
            # Perform feature selection
            selection_result = self._perform_feature_selection(X, y, context)
            
            # Apply selection to data
            selected_data = data[selection_result.selected_features + ([target] if target else [])]
            
            # Generate LLM insights about selection
            llm_insights = ""
            if self.use_llm:
                llm_insights = self._generate_selection_insights(selection_result, context)
            
            # Update context
            context.selected_features = selection_result.selected_features
            context.data = selected_data
            context.llm_insights[self.name] = llm_insights
            
            # Store selection results in memory
            self._store_selection_memory(selection_result, context)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(selection_result)
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Selected {selection_result.final_feature_count} features from {selection_result.original_feature_count}",
                    f"Reduction: {selection_result.reduction_percentage:.1f}%",
                    f"Selection methods: {', '.join(selection_result.selection_methods)}",
                    f"Removed features: {', '.join(selection_result.removed_features[:5])}{'...' if len(selection_result.removed_features) > 5 else ''}",
                ],
                metrics={
                    "original_features": selection_result.original_feature_count,
                    "selected_features": selection_result.final_feature_count,
                    "reduction_percentage": selection_result.reduction_percentage,
                    "methods_used": len(selection_result.selection_methods),
                },
                artifacts={
                    "selection_result": selection_result.dict(),
                    "llm_insights": llm_insights,
                },
                suggestions=suggestions,
                requires_approval=selection_result.reduction_percentage > 50,
            )
            
            # Add warnings for significant feature reduction
            if selection_result.reduction_percentage > 70:
                result.warnings = [
                    f"Significant feature reduction ({selection_result.reduction_percentage:.1f}%) - "
                    f"verify important features weren't removed"
                ]
                result.requires_approval = True
                result.user_message = (
                    f"Feature selection removed {selection_result.reduction_percentage:.1f}% of features. "
                    f"Please review the selection before proceeding."
                )
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Feature selection failed: {e}"],
                user_message="Feature selection encountered an error. Please review the data and settings.",
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
        
        if context.data.shape[1] < 2:
            raise ValidationError("Dataset must have at least 2 features for selection")
        
        return True
    
    def _perform_feature_selection(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        context: AgentContext,
    ) -> FeatureSelectionResult:
        """Perform comprehensive feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            context: Pipeline context
            
        Returns:
            FeatureSelectionResult with selection details
        """
        original_features = list(X.columns)
        selected_features = original_features.copy()
        removed_features = []
        selection_methods = []
        feature_scores = {}
        selection_rationale = {}
        
        # 1. Remove low-variance features
        if len(selected_features) > 1:
            variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            
            # Handle non-numeric features
            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 0:
                numeric_mask = variance_selector.fit_transform(X[numeric_features]).shape[1]
                low_var_features = [
                    col for i, col in enumerate(numeric_features)
                    if not variance_selector.get_support()[i]
                ]
                
                if low_var_features:
                    selected_features = [f for f in selected_features if f not in low_var_features]
                    removed_features.extend(low_var_features)
                    selection_methods.append("variance_threshold")
                    
                    for feature in low_var_features:
                        feature_scores[feature] = 0.0
                        selection_rationale[feature] = f"Removed due to low variance (<{self.variance_threshold})"
        
        # 2. Remove highly correlated features
        if len(selected_features) > 1:
            numeric_selected = [f for f in selected_features if f in X.select_dtypes(include=[np.number]).columns]
            
            if len(numeric_selected) > 1:
                corr_matrix = X[numeric_selected].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                high_corr_features = [
                    column for column in upper_triangle.columns
                    if any(upper_triangle[column] > self.correlation_threshold)
                ]
                
                if high_corr_features:
                    # Keep features with highest correlation to target if available
                    if y is not None:
                        target_corrs = {}
                        for feature in high_corr_features:
                            if feature in X.columns and X[feature].dtype in [np.number]:
                                try:
                                    target_corrs[feature] = abs(X[feature].corr(y))
                                except:
                                    target_corrs[feature] = 0.0
                        
                        # Remove features with lower target correlation
                        features_to_remove = []
                        for feature in high_corr_features:
                            corr_partners = upper_triangle[feature].dropna()
                            corr_partners = corr_partners[corr_partners > self.correlation_threshold]
                            
                            for partner in corr_partners.index:
                                if (partner in selected_features and 
                                    target_corrs.get(feature, 0) < target_corrs.get(partner, 0)):
                                    features_to_remove.append(feature)
                                    break
                        
                        features_to_remove = list(set(features_to_remove))
                    else:
                        # Without target, remove arbitrary correlated features
                        features_to_remove = high_corr_features[:len(high_corr_features)//2]
                    
                    if features_to_remove:
                        selected_features = [f for f in selected_features if f not in features_to_remove]
                        removed_features.extend(features_to_remove)
                        selection_methods.append("correlation_filter")
                        
                        for feature in features_to_remove:
                            selection_rationale[feature] = f"Removed due to high correlation (>{self.correlation_threshold})"
        
        # 3. Statistical feature selection (if target available)
        if y is not None and len(selected_features) > 1:
            X_selected = X[selected_features]
            
            # Determine task type
            is_classification = (y.dtype == 'object' or y.nunique() < 20)
            
            try:
                if is_classification:
                    # Encode target for classification
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y.fillna('missing'))
                    
                    # Use chi2 for categorical features, f_classif for numeric
                    numeric_features = X_selected.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_features) > 0:
                        # F-test for numeric features
                        f_selector = SelectPercentile(f_classif, percentile=self.importance_percentile)
                        X_numeric_selected = f_selector.fit_transform(
                            X_selected[numeric_features].fillna(0), y_encoded
                        )
                        
                        selected_numeric = [
                            numeric_features[i] for i in range(len(numeric_features))
                            if f_selector.get_support()[i]
                        ]
                        
                        # Get scores
                        scores = f_selector.scores_
                        for i, feature in enumerate(numeric_features):
                            feature_scores[feature] = float(scores[i])
                        
                        # Remove unselected numeric features
                        removed_numeric = [f for f in numeric_features if f not in selected_numeric]
                        if removed_numeric:
                            selected_features = [f for f in selected_features if f not in removed_numeric]
                            removed_features.extend(removed_numeric)
                            selection_methods.append("f_test_classification")
                            
                            for feature in removed_numeric:
                                selection_rationale[feature] = f"Removed by F-test (low classification relevance)"
                
                else:
                    # Regression task
                    numeric_features = X_selected.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_features) > 0:
                        f_selector = SelectPercentile(f_regression, percentile=self.importance_percentile)
                        X_numeric_selected = f_selector.fit_transform(
                            X_selected[numeric_features].fillna(0), y.fillna(y.mean())
                        )
                        
                        selected_numeric = [
                            numeric_features[i] for i in range(len(numeric_features))
                            if f_selector.get_support()[i]
                        ]
                        
                        # Get scores
                        scores = f_selector.scores_
                        for i, feature in enumerate(numeric_features):
                            feature_scores[feature] = float(scores[i])
                        
                        # Remove unselected numeric features
                        removed_numeric = [f for f in numeric_features if f not in selected_numeric]
                        if removed_numeric:
                            selected_features = [f for f in selected_features if f not in removed_numeric]
                            removed_features.extend(removed_numeric)
                            selection_methods.append("f_test_regression")
                            
                            for feature in removed_numeric:
                                selection_rationale[feature] = f"Removed by F-test (low regression relevance)"
            
            except Exception as e:
                self.logger.warning(f"Statistical feature selection failed: {e}")
        
        # 4. Ensure we don't remove too many features
        max_features = max(1, int(len(original_features) * self.max_features_ratio))
        if len(selected_features) < max_features:
            # Add back some features if we removed too many
            features_to_add_back = [
                f for f in removed_features 
                if f in feature_scores and feature_scores[f] > 0
            ]
            
            # Sort by score and add back top features
            features_to_add_back.sort(key=lambda x: feature_scores.get(x, 0), reverse=True)
            needed_features = max_features - len(selected_features)
            
            for feature in features_to_add_back[:needed_features]:
                selected_features.append(feature)
                removed_features.remove(feature)
                selection_rationale[feature] = "Added back to maintain feature count"
        
        # Calculate reduction percentage
        reduction_percentage = ((len(original_features) - len(selected_features)) / len(original_features)) * 100
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            removed_features=removed_features,
            selection_methods=selection_methods,
            feature_scores=feature_scores,
            selection_rationale=selection_rationale,
            original_feature_count=len(original_features),
            final_feature_count=len(selected_features),
            reduction_percentage=reduction_percentage,
        )
    
    def _generate_selection_insights(
        self,
        selection_result: FeatureSelectionResult,
        context: AgentContext,
    ) -> str:
        """Generate LLM insights about feature selection.
        
        Args:
            selection_result: Feature selection results
            context: Pipeline context
            
        Returns:
            Generated insights text
        """
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "LLM service not available for feature selection insights"
            
            prompt = f"""
            Analyze this feature selection result and provide insights:
            
            Original features: {selection_result.original_feature_count}
            Selected features: {selection_result.final_feature_count}
            Reduction: {selection_result.reduction_percentage:.1f}%
            
            Selection methods used: {', '.join(selection_result.selection_methods)}
            
            Top removed features by score:
            {dict(list(sorted(selection_result.feature_scores.items(), key=lambda x: x[1], reverse=True))[:5])}
            
            Target variable: {context.target or 'None (unsupervised)'}
            
            Please provide:
            1. Assessment of the feature selection approach
            2. Potential impact on model performance
            3. Recommendations for feature engineering
            4. Any concerns about removed features
            
            Be concise and actionable.
            """
            
            insights = llm_service.generate_simple_response(
                prompt=prompt,
                system_message="You are a feature selection expert providing analysis of feature selection results.",
                max_tokens=600,
                temperature=0.7,
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate selection insights: {e}")
            return f"Could not generate insights: {e}"
    
    def _generate_suggestions(self, selection_result: FeatureSelectionResult) -> List[str]:
        """Generate suggestions based on feature selection results.
        
        Args:
            selection_result: Feature selection results
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if selection_result.reduction_percentage > 60:
            suggestions.append(
                f"Significant feature reduction ({selection_result.reduction_percentage:.1f}%). "
                f"Consider if domain-important features were removed."
            )
        
        if len(selection_result.selection_methods) == 1:
            suggestions.append(
                "Only one selection method was used. Consider ensemble feature selection for better results."
            )
        
        if selection_result.final_feature_count < 5:
            suggestions.append(
                f"Very few features selected ({selection_result.final_feature_count}). "
                f"Consider feature engineering to create more informative features."
            )
        
        if "correlation_filter" in selection_result.selection_methods:
            suggestions.append(
                "High correlation features were removed. Consider creating interaction terms."
            )
        
        return suggestions
    
    def _store_selection_memory(
        self,
        selection_result: FeatureSelectionResult,
        context: AgentContext,
    ) -> None:
        """Store feature selection results in memory.
        
        Args:
            selection_result: Selection results to store
            context: Pipeline context
        """
        try:
            # Store detailed selection log
            selection_data = {
                "original_feature_count": selection_result.original_feature_count,
                "final_feature_count": selection_result.final_feature_count,
                "reduction_percentage": selection_result.reduction_percentage,
                "selection_methods": selection_result.selection_methods,
                "selected_features": selection_result.selected_features,
                "removed_features": selection_result.removed_features,
                "feature_scores": selection_result.feature_scores,
                "target_variable": context.target,
            }
            
            memory_client.store_symbolic(
                table_name="feature_selection_log",
                data=selection_data,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
            # Create semantic description
            description = f"""
            Feature selection completed:
            - Reduced from {selection_result.original_feature_count} to {selection_result.final_feature_count} features
            - Reduction: {selection_result.reduction_percentage:.1f}%
            - Methods: {', '.join(selection_result.selection_methods)}
            - Target: {context.target or 'unsupervised'}
            - Top selected features: {', '.join(selection_result.selected_features[:5])}
            """
            
            memory_client.store_vector(
                content=description.strip(),
                tags=["feature_selection", "dimensionality_reduction"],
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
                metadata={"reduction_percentage": selection_result.reduction_percentage},
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store selection memory: {e}")
    
    def _get_llm_prompt(self, context: AgentContext, feedback: Optional[UserFeedback] = None) -> str:
        """Generate LLM prompt for feature selection insights.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            Formatted prompt string
        """
        data = context.data
        target = context.target
        
        prompt = f"""
        You are a feature selection expert analyzing a dataset for feature reduction.

        Dataset Information:
        - Shape: {data.shape}
        - Target variable: {target or 'Not specified (unsupervised)'}
        - Feature types: {dict(data.dtypes.value_counts())}

        Current Features:
        {list(data.columns[:20])}{'...' if len(data.columns) > 20 else ''}

        Please provide recommendations for:
        1. Which features are likely to be most important
        2. Feature selection strategies for this data type
        3. Potential feature interactions to consider
        4. Risk factors in feature removal

        Focus on actionable advice for this specific dataset.
        """
        
        return prompt.strip()
    
    def _apply_overrides(
        self,
        context: AgentContext,
        overrides: Dict[str, Any],
    ) -> AgentContext:
        """Apply user overrides to feature selection.
        
        Args:
            context: Current context
            overrides: User overrides to apply
            
        Returns:
            Updated context
        """
        if "force_include_features" in overrides:
            force_include = overrides["force_include_features"]
            if isinstance(force_include, list):
                self.logger.info(f"User forced inclusion of features: {force_include}")
                # This would be handled in the selection logic
        
        if "force_exclude_features" in overrides:
            force_exclude = overrides["force_exclude_features"]
            if isinstance(force_exclude, list):
                self.logger.info(f"User forced exclusion of features: {force_exclude}")
                # This would be handled in the selection logic
        
        if "selection_method" in overrides:
            method = overrides["selection_method"]
            self.logger.info(f"User specified selection method: {method}")
        
        return context