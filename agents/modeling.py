"""Modeling Agent - Trains multiple models and selects the best performer."""

import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from agents.base import AgentContext, AgentResult, LLMCapableAgent, UserFeedback
from config import modeling_config, settings
from memory.memory_client import memory_client
from utils.exceptions import ModelingError, ValidationError


class ModelResult(BaseModel):
    """Result of training a single model."""
    
    model_name: str
    model_type: str  # classification or regression
    training_time: float
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    cross_val_scores: Optional[List[float]] = None


class ModelingResult(BaseModel):
    """Result of the modeling process."""
    
    task_type: str
    models_trained: List[ModelResult]
    best_model: ModelResult
    model_comparison: Dict[str, Dict[str, float]]
    dataset_split: Dict[str, int]
    total_training_time: float
    recommendations: List[str]


class ModelingAgent(LLMCapableAgent):
    """Agent responsible for training and evaluating multiple ML models."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the modeling agent.
        
        Args:
            use_llm: Whether to use LLM for generating insights and recommendations
        """
        super().__init__("ModelingAgent", use_llm)
        
        # Training configuration
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        self.max_training_time = settings.max_time_per_model
        self.max_models = settings.max_models_per_run
    
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute model training and selection process.
        
        Args:
            context: Pipeline context containing engineered features
            feedback: Optional user feedback for refinement
            
        Returns:
            AgentResult with trained models and performance comparison
        """
        try:
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            # Handle user feedback
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            data = context.data.copy()
            target = context.target
            
            self.logger.info(f"Starting model training for dataset with shape: {data.shape}")
            
            # Prepare data for modeling
            X, y, task_type = self._prepare_data(data, target)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y if task_type == 'classification' else None
            )
            
            # Get model recommendations from LLM
            llm_recommendations = []
            if self.use_llm:
                llm_recommendations = self._get_llm_model_recommendations(data, target, task_type, context)
            
            # Train models
            modeling_result = self._train_models(X_train, y_train, X_test, y_test, task_type, llm_recommendations)
            
            # Save best model
            best_model_path = self._save_best_model(modeling_result.best_model, context)
            modeling_result.best_model.model_path = best_model_path
            
            # Generate LLM insights about modeling results
            llm_insights = ""
            if self.use_llm:
                llm_insights = self._generate_modeling_insights(modeling_result, context)
            
            # Update context
            context.model = modeling_result.best_model
            context.llm_insights[self.name] = llm_insights
            
            # Store modeling results in memory
            self._store_modeling_memory(modeling_result, context)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(modeling_result)
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Trained {len(modeling_result.models_trained)} models for {task_type} task",
                    f"Best model: {modeling_result.best_model.model_name}",
                    f"Best score: {self._get_primary_metric(modeling_result.best_model, task_type):.4f}",
                    f"Total training time: {modeling_result.total_training_time:.2f} seconds",
                ],
                metrics={
                    "task_type": task_type,
                    "models_trained": len(modeling_result.models_trained),
                    "best_model": modeling_result.best_model.model_name,
                    "best_score": self._get_primary_metric(modeling_result.best_model, task_type),
                    "training_time": modeling_result.total_training_time,
                },
                artifacts={
                    "modeling_result": modeling_result.dict(),
                    "best_model_path": best_model_path,
                    "llm_insights": llm_insights,
                },
                suggestions=suggestions,
                requires_approval=False,  # Models can be reviewed in evaluation phase
            )
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Model training failed: {e}"],
                user_message="Model training encountered an error. Please review the data and configuration.",
            )
            
            self._log_execution(result)
            return result
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate that the context contains required data and target.
        
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
        
        if not context.target:
            raise ValidationError("Target variable must be specified for modeling")
        
        if context.target not in context.data.columns:
            raise ValidationError(f"Target variable '{context.target}' not found in data")
        
        # Check for sufficient data
        if len(context.data) < 50:
            raise ValidationError("Insufficient data for reliable model training (minimum 50 rows)")
        
        return True
    
    def _prepare_data(self, data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Prepare data for modeling.
        
        Args:
            data: Dataset with features and target
            target: Target variable name
            
        Returns:
            Tuple of (features, target, task_type)
        """
        # Separate features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Determine task type
        if y.dtype == 'object' or y.nunique() < 20:
            task_type = 'classification'
            # Encode categorical target
            if y.dtype == 'object':
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.fillna('missing')), index=y.index)
        else:
            task_type = 'regression'
        
        # Handle missing values in features
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        y = y.fillna(y.mean() if task_type == 'regression' else y.mode().iloc[0] if len(y.mode()) > 0 else 0)
        
        # Encode categorical features
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        self.logger.info(f"Prepared data for {task_type}: X shape {X.shape}, y shape {y.shape}")
        
        return X, y, task_type
    
    def _get_llm_model_recommendations(
        self,
        data: pd.DataFrame,
        target: str,
        task_type: str,
        context: AgentContext,
    ) -> List[str]:
        """Get LLM recommendations for model selection.
        
        Args:
            data: Dataset
            target: Target variable
            task_type: Classification or regression
            context: Pipeline context
            
        Returns:
            List of model recommendations
        """
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return []
            
            dataset_characteristics = {
                "shape": data.shape,
                "task_type": task_type,
                "feature_count": data.shape[1] - 1,
                "sample_size": len(data),
                "target_distribution": {
                    "unique_values": int(data[target].nunique()),
                    "null_percentage": float((data[target].isnull().sum() / len(data)) * 100),
                },
                "feature_types": dict(data.dtypes.value_counts()),
            }
            
            recommendations_text = llm_service.generate_model_recommendations(
                dataset_characteristics=dataset_characteristics,
                task_type=task_type,
                max_tokens=600,
                temperature=0.7,
            )
            
            # Parse recommendations (simple implementation)
            recommendations = [
                line.strip() for line in recommendations_text.split('\n')
                if line.strip() and not line.startswith('#')
            ]
            
            return recommendations[:5]  # Limit recommendations
            
        except Exception as e:
            self.logger.warning(f"Failed to get LLM model recommendations: {e}")
            return []
    
    def _train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        llm_recommendations: List[str],
    ) -> ModelingResult:
        """Train multiple models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            task_type: Classification or regression
            llm_recommendations: LLM recommendations
            
        Returns:
            ModelingResult with all trained models
        """
        models_to_train = self._select_models(task_type, llm_recommendations)
        model_results = []
        total_start_time = time.time()
        
        for model_name, model_config in models_to_train.items():
            try:
                self.logger.info(f"Training {model_name}...")
                start_time = time.time()
                
                # Create model instance
                model = self._create_model(model_name, model_config, task_type)
                
                # Train model
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                else:
                    raise ModelingError(f"Model {model_name} does not have fit method")
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, task_type, model, X_test)
                
                # Get feature importance if available
                feature_importance = self._get_feature_importance(model, X_train.columns)
                
                training_time = time.time() - start_time
                
                model_result = ModelResult(
                    model_name=model_name,
                    model_type=task_type,
                    training_time=training_time,
                    hyperparameters=model.get_params(),
                    metrics=metrics,
                    feature_importance=feature_importance,
                )
                
                model_results.append(model_result)
                
                self.logger.info(f"{model_name} completed in {training_time:.2f}s")
                
                # Store model temporarily for best model selection
                model_result._model_instance = model
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        if not model_results:
            raise ModelingError("No models were successfully trained")
        
        # Select best model
        best_model = self._select_best_model(model_results, task_type)
        
        # Create model comparison
        model_comparison = {}
        primary_metric = 'roc_auc' if task_type == 'classification' else 'r2'
        
        for result in model_results:
            model_comparison[result.model_name] = {
                'primary_metric': result.metrics.get(primary_metric, 0),
                'training_time': result.training_time,
                **result.metrics
            }
        
        total_training_time = time.time() - total_start_time
        
        # Generate recommendations
        recommendations = self._generate_model_recommendations(model_results, task_type)
        
        return ModelingResult(
            task_type=task_type,
            models_trained=model_results,
            best_model=best_model,
            model_comparison=model_comparison,
            dataset_split={
                'train_size': len(X_train),
                'test_size': len(X_test),
                'total_size': len(X_train) + len(X_test),
            },
            total_training_time=total_training_time,
            recommendations=recommendations,
        )
    
    def _select_models(self, task_type: str, llm_recommendations: List[str]) -> Dict[str, Dict[str, Any]]:
        """Select models to train based on task type and recommendations.
        
        Args:
            task_type: Classification or regression
            llm_recommendations: LLM recommendations
            
        Returns:
            Dictionary of models to train
        """
        if task_type == 'classification':
            base_models = modeling_config.CLASSIFICATION_MODELS
        else:
            base_models = modeling_config.REGRESSION_MODELS
        
        # Select top models (limited by max_models setting)
        selected_models = dict(list(base_models.items())[:self.max_models])
        
        return selected_models
    
    def _create_model(self, model_name: str, model_config: Dict[str, Any], task_type: str):
        """Create a model instance.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            task_type: Task type
            
        Returns:
            Model instance
        """
        params = model_config.get('params', {})
        
        if task_type == 'classification':
            if model_name == 'logistic_regression':
                return LogisticRegression(**params)
            elif model_name == 'random_forest':
                return RandomForestClassifier(**params)
            elif model_name == 'xgboost':
                return XGBClassifier(**params)
            elif model_name == 'knn':
                return KNeighborsClassifier(**params)
            elif model_name == 'svm':
                return SVC(**params)
        
        else:  # regression
            if model_name == 'linear_regression':
                return LinearRegression(**params)
            elif model_name == 'ridge':
                return Ridge(**params)
            elif model_name == 'random_forest':
                return RandomForestRegressor(**params)
            elif model_name == 'xgboost':
                return XGBRegressor(**params)
            elif model_name == 'svr':
                return SVR(**params)
        
        raise ModelingError(f"Unknown model: {model_name}")
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        task_type: str,
        model: Any,
        X_test: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            task_type: Task type
            model: Trained model
            X_test: Test features
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        try:
            if task_type == 'classification':
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                
                # For binary classification
                if len(np.unique(y_true)) == 2:
                    metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
                    metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
                    metrics['f1'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
                    
                    # ROC AUC if model supports predict_proba
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
                        except:
                            pass
                
                else:  # Multi-class
                    metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                    metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                    metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            
            else:  # regression
                metrics['r2'] = float(r2_score(y_true, y_pred))
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        
        except Exception as e:
            self.logger.warning(f"Failed to calculate some metrics: {e}")
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores or None
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return None
            
            if len(importances) == len(feature_names):
                return dict(zip(feature_names, importances.astype(float)))
        
        except Exception as e:
            self.logger.warning(f"Failed to extract feature importance: {e}")
        
        return None
    
    def _select_best_model(self, model_results: List[ModelResult], task_type: str) -> ModelResult:
        """Select the best performing model.
        
        Args:
            model_results: List of model results
            task_type: Task type
            
        Returns:
            Best model result
        """
        if task_type == 'classification':
            # Prefer ROC AUC, then F1, then accuracy
            for metric in ['roc_auc', 'f1', 'accuracy']:
                scored_models = [m for m in model_results if metric in m.metrics]
                if scored_models:
                    return max(scored_models, key=lambda x: x.metrics[metric])
        
        else:  # regression
            # Prefer R2, then negative MAE
            if any('r2' in m.metrics for m in model_results):
                return max(model_results, key=lambda x: x.metrics.get('r2', -np.inf))
            else:
                return min(model_results, key=lambda x: x.metrics.get('mae', np.inf))
        
        # Fallback to first model
        return model_results[0]
    
    def _get_primary_metric(self, model_result: ModelResult, task_type: str) -> float:
        """Get the primary metric value for a model.
        
        Args:
            model_result: Model result
            task_type: Task type
            
        Returns:
            Primary metric value
        """
        if task_type == 'classification':
            return model_result.metrics.get('roc_auc', 
                   model_result.metrics.get('f1',
                   model_result.metrics.get('accuracy', 0)))
        else:
            return model_result.metrics.get('r2', 0)
    
    def _save_best_model(self, best_model: ModelResult, context: AgentContext) -> str:
        """Save the best model to disk.
        
        Args:
            best_model: Best model result
            context: Pipeline context
            
        Returns:
            Path to saved model
        """
        try:
            model_filename = f"{best_model.model_name}_{int(time.time())}.pkl"
            model_path = settings.models_path / model_filename
            
            # Get the actual model instance
            model_instance = getattr(best_model, '_model_instance', None)
            if model_instance is None:
                raise ModelingError("Model instance not found for saving")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance, f)
            
            self.logger.info(f"Saved best model to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise ModelingError(f"Model saving failed: {e}")
    
    def _generate_modeling_insights(
        self,
        modeling_result: ModelingResult,
        context: AgentContext,
    ) -> str:
        """Generate LLM insights about modeling results.
        
        Args:
            modeling_result: Modeling results
            context: Pipeline context
            
        Returns:
            Generated insights text
        """
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "LLM service not available for modeling insights"
            
            model_info = {
                "best_model": modeling_result.best_model.model_name,
                "task_type": modeling_result.task_type,
                "models_trained": len(modeling_result.models_trained),
                "training_time": modeling_result.total_training_time,
            }
            
            metrics = modeling_result.best_model.metrics
            feature_importance = modeling_result.best_model.feature_importance
            
            insights = llm_service.interpret_model_results(
                model_info=model_info,
                metrics=metrics,
                feature_importance=feature_importance,
                max_tokens=800,
                temperature=0.7,
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate modeling insights: {e}")
            return f"Could not generate insights: {e}"
    
    def _generate_suggestions(self, modeling_result: ModelingResult) -> List[str]:
        """Generate suggestions based on modeling results.
        
        Args:
            modeling_result: Modeling results
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        best_score = self._get_primary_metric(modeling_result.best_model, modeling_result.task_type)
        
        if modeling_result.task_type == 'classification':
            if best_score < 0.7:
                suggestions.append(
                    f"Model performance is moderate (score: {best_score:.3f}). "
                    f"Consider more feature engineering or data collection."
                )
            elif best_score > 0.95:
                suggestions.append(
                    f"Very high performance (score: {best_score:.3f}). "
                    f"Check for data leakage or overfitting."
                )
        
        else:  # regression
            if best_score < 0.5:
                suggestions.append(
                    f"Low R² score ({best_score:.3f}). "
                    f"Consider different features or model approaches."
                )
        
        if modeling_result.total_training_time > 300:  # 5 minutes
            suggestions.append(
                f"Long training time ({modeling_result.total_training_time:.1f}s). "
                f"Consider feature selection or simpler models."
            )
        
        if len(modeling_result.models_trained) < 3:
            suggestions.append(
                "Few models were trained. Consider expanding the model selection."
            )
        
        return suggestions
    
    def _generate_model_recommendations(
        self,
        model_results: List[ModelResult],
        task_type: str,
    ) -> List[str]:
        """Generate recommendations based on model comparison.
        
        Args:
            model_results: List of model results
            task_type: Task type
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Sort by performance
        primary_metric = 'roc_auc' if task_type == 'classification' else 'r2'
        sorted_models = sorted(
            model_results,
            key=lambda x: x.metrics.get(primary_metric, 0),
            reverse=True
        )
        
        if len(sorted_models) >= 2:
            best = sorted_models[0]
            second_best = sorted_models[1]
            
            best_score = best.metrics.get(primary_metric, 0)
            second_score = second_best.metrics.get(primary_metric, 0)
            
            if abs(best_score - second_score) < 0.02:  # Very close performance
                recommendations.append(
                    f"Close performance between {best.model_name} and {second_best.model_name}. "
                    f"Consider ensemble methods."
                )
        
        return recommendations
    
    def _store_modeling_memory(
        self,
        modeling_result: ModelingResult,
        context: AgentContext,
    ) -> None:
        """Store modeling results in memory.
        
        Args:
            modeling_result: Modeling results to store
            context: Pipeline context
        """
        try:
            # Store best model performance
            dataset_profile = {
                "shape": context.data.shape,
                "task_type": modeling_result.task_type,
                "feature_count": context.data.shape[1] - 1,
            }
            
            memory_client.store_model_performance(
                model_name=modeling_result.best_model.model_name,
                hyperparameters=modeling_result.best_model.hyperparameters,
                metrics=modeling_result.best_model.metrics,
                dataset_profile=dataset_profile,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
            # Store comparison data
            modeling_data = {
                "task_type": modeling_result.task_type,
                "models_trained": len(modeling_result.models_trained),
                "best_model": modeling_result.best_model.model_name,
                "model_comparison": modeling_result.model_comparison,
                "total_training_time": modeling_result.total_training_time,
                "dataset_size": modeling_result.dataset_split["total_size"],
            }
            
            memory_client.store_symbolic(
                table_name="modeling_sessions",
                data=modeling_data,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store modeling memory: {e}")
    
    def _get_llm_prompt(self, context: AgentContext, feedback: Optional[UserFeedback] = None) -> str:
        """Generate LLM prompt for modeling insights.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            Formatted prompt string
        """
        data = context.data
        target = context.target
        
        # Determine task type
        task_type = 'classification' if data[target].nunique() < 20 else 'regression'
        
        prompt = f"""
        You are a machine learning expert selecting models for this dataset:

        Dataset Information:
        - Shape: {data.shape}
        - Target variable: {target}
        - Task type: {task_type}
        - Target distribution: {data[target].value_counts().head().to_dict()}

        Please recommend:
        1. Best model types for this data size and task
        2. Important hyperparameters to tune
        3. Potential challenges with this dataset
        4. Model interpretation considerations

        Focus on practical, actionable recommendations.
        """
        
        return prompt.strip()
    
    def _apply_overrides(
        self,
        context: AgentContext,
        overrides: Dict[str, Any],
    ) -> AgentContext:
        """Apply user overrides to modeling process.
        
        Args:
            context: Current context
            overrides: User overrides to apply
            
        Returns:
            Updated context
        """
        if "preferred_models" in overrides:
            preferred = overrides["preferred_models"]
            self.logger.info(f"User specified preferred models: {preferred}")
        
        if "max_training_time" in overrides:
            max_time = overrides["max_training_time"]
            self.max_training_time = max_time
            self.logger.info(f"Updated max training time to: {max_time}")
        
        if "primary_metric" in overrides:
            metric = overrides["primary_metric"]
            self.logger.info(f"User specified primary metric: {metric}")
        
        return context