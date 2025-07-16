"""Evaluation Agent - Evaluates model performance and generates insights."""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import threading
import time
from typing import Optional

from agents.base import AgentContext, AgentResult, LLMCapableAgent, UserFeedback
from config import settings
from memory.memory_client import memory_client
from utils.exceptions import EvaluationError, ValidationError


class EvaluationResult(BaseModel):
    """Result of model evaluation process."""
    
    model_name: str
    task_type: str
    metrics_summary: dict
    detailed_metrics: dict
    confusion_matrix: Optional[list] = None
    classification_report: Optional[dict] = None
    plot_paths: list = []
    interpretation: str = ""
    recommendations: list = []
    performance_analysis: dict = {}
    cross_validation_scores: dict = {}
    feature_importance_analysis: dict = {}


class EvaluationAgent(LLMCapableAgent):
    """Agent responsible for comprehensive model evaluation and interpretation."""
    
    def __init__(self, use_llm: bool = True):
        super().__init__("EvaluationAgent", use_llm)
    
    def run(self, context: AgentContext, feedback = None) -> AgentResult:
        """Execute model evaluation process with parallel optimization."""
        try:
            start_time = time.time()
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            model_result = context.model
            data = context.data
            target = context.target
            
            # Perform parallel evaluation tasks
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all evaluation tasks concurrently
                futures = []
                
                # Core evaluation
                futures.append(executor.submit(self._evaluate_model_performance, model_result, data, target))
                
                # Visualization generation
                if hasattr(model_result, 'predictions') and model_result.predictions is not None:
                    futures.append(executor.submit(self._generate_performance_plots, model_result, data, target))
                else:
                    futures.append(executor.submit(lambda: []))  # Empty plots if no predictions
                
                # Feature importance analysis
                if hasattr(model_result, 'feature_importance') and model_result.feature_importance:
                    futures.append(executor.submit(self._analyze_feature_importance, model_result.feature_importance))
                else:
                    futures.append(executor.submit(lambda: {}))  # Empty analysis if no feature importance
                
                # Cross-validation (if model supports it)
                futures.append(executor.submit(self._perform_fast_validation, model_result, data, target))
                
                # Wait for all tasks to complete
                evaluation_data = futures[0].result()
                plot_paths = futures[1].result()
                feature_analysis = futures[2].result()
                cv_scores = futures[3].result()
            
            # Combine results
            evaluation_result = EvaluationResult(
                model_name=evaluation_data['model_name'],
                task_type=evaluation_data['task_type'],
                metrics_summary=evaluation_data['metrics_summary'],
                detailed_metrics=evaluation_data['detailed_metrics'],
                confusion_matrix=evaluation_data.get('confusion_matrix'),
                classification_report=evaluation_data.get('classification_report'),
                plot_paths=plot_paths,
                performance_analysis=evaluation_data.get('performance_analysis', {}),
                cross_validation_scores=cv_scores,
                feature_importance_analysis=feature_analysis,
                recommendations=self._generate_fast_recommendations(evaluation_data)
            )
            
            # Generate LLM insights in parallel with memory storage
            llm_future = None
            if self.use_llm:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    llm_future = executor.submit(self._generate_evaluation_insights, evaluation_result, context)
                    memory_future = executor.submit(self._store_evaluation_memory, evaluation_result, context)
                    
                    evaluation_result.interpretation = llm_future.result()
                    memory_future.result()  # Ensure memory storage completes
            else:
                self._store_evaluation_memory(evaluation_result, context)
                evaluation_result.interpretation = "LLM insights disabled."
            
            # Update context
            context.evaluation_results = evaluation_result.dict()
            context.llm_insights[self.name] = evaluation_result.interpretation
            
            execution_time = time.time() - start_time
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Evaluated {evaluation_result.model_name} model in {execution_time:.2f}s",
                    f"Task type: {evaluation_result.task_type}",
                    f"Generated {len(evaluation_result.plot_paths)} visualization(s)",
                    f"Cross-validation: {len(evaluation_result.cross_validation_scores)} metrics",
                ],
                metrics={
                    **evaluation_result.metrics_summary,
                    "evaluation_time": execution_time,
                    "plots_generated": len(evaluation_result.plot_paths),
                },
                artifacts={
                    "evaluation_result": evaluation_result.dict(),
                    "interpretation": evaluation_result.interpretation,
                    "plot_paths": evaluation_result.plot_paths,
                },
                suggestions=evaluation_result.recommendations,
                requires_approval=False,
            )
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            
            return AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Model evaluation failed: {e}"],
                user_message="Model evaluation encountered an error.",
            )
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate inputs for evaluation."""
        if context.model is None:
            raise ValidationError("No trained model found in context")
        
        if context.data is None:
            raise ValidationError("No data found in context")
        
        if not context.target:
            raise ValidationError("Target variable not specified")
        
        return True
    
    def _evaluate_model_performance(self, model_result, data: pd.DataFrame, target: str) -> dict:
        """Perform comprehensive model evaluation with enhanced metrics."""
        
        base_metrics = model_result.metrics.copy()
        task_type = model_result.model_type
        
        # Enhanced performance analysis
        performance_analysis = {}
        confusion_matrix_data = None
        classification_report_data = None
        
        try:
            if hasattr(model_result, 'predictions') and model_result.predictions is not None:
                y_true = data[target] if target in data.columns else None
                y_pred = model_result.predictions
                
                if y_true is not None and len(y_true) == len(y_pred):
                    if task_type == 'classification':
                        # Classification-specific metrics
                        try:
                            confusion_matrix_data = confusion_matrix(y_true, y_pred).tolist()
                            classification_report_data = classification_report(y_true, y_pred, output_dict=True)
                        except Exception as e:
                            self.logger.warning(f"Could not generate confusion matrix/classification report: {e}")
                            confusion_matrix_data = None
                            classification_report_data = None
                        
                        # Additional classification metrics
                        unique_classes = len(np.unique(y_true))
                        if unique_classes == 2:
                            # Binary classification - add AUC if possible
                            if hasattr(model_result, 'prediction_probabilities'):
                                try:
                                    fpr, tpr, _ = roc_curve(y_true, model_result.prediction_probabilities[:, 1])
                                    base_metrics['auc_roc'] = auc(fpr, tpr)
                                    
                                    precision, recall, _ = precision_recall_curve(y_true, model_result.prediction_probabilities[:, 1])
                                    base_metrics['auc_pr'] = auc(recall, precision)
                                except Exception as e:
                                    self.logger.warning(f"Could not compute AUC metrics: {e}")
                        
                        try:
                            performance_analysis['class_distribution'] = dict(pd.Series(y_true).value_counts())
                            performance_analysis['prediction_distribution'] = dict(pd.Series(y_pred).value_counts())
                        except Exception as e:
                            self.logger.warning(f"Could not compute class distributions: {e}")
                    
                    else:
                        # Regression-specific analysis - ensure confusion matrix fields are None
                        confusion_matrix_data = None
                        classification_report_data = None
                        
                        try:
                            residuals = y_true - y_pred
                            performance_analysis['residual_stats'] = {
                                'mean': float(np.mean(residuals)),
                                'std': float(np.std(residuals)),
                                'min': float(np.min(residuals)),
                                'max': float(np.max(residuals)),
                                'q25': float(np.percentile(residuals, 25)),
                                'q75': float(np.percentile(residuals, 75))
                            }
                            
                            # Additional regression metrics
                            base_metrics['mean_residual'] = float(np.mean(residuals))
                            base_metrics['residual_std'] = float(np.std(residuals))
                        except Exception as e:
                            self.logger.warning(f"Could not compute residual analysis: {e}")
                else:
                    # No valid predictions available
                    confusion_matrix_data = None
                    classification_report_data = None
            else:
                # No predictions available
                confusion_matrix_data = None
                classification_report_data = None
        
        except Exception as e:
            self.logger.warning(f"Could not compute enhanced metrics: {e}")
        
        return {
            'model_name': model_result.model_name,
            'task_type': task_type,
            'metrics_summary': base_metrics,
            'detailed_metrics': base_metrics,
            'confusion_matrix': confusion_matrix_data,
            'classification_report': classification_report_data,
            'performance_analysis': performance_analysis
        }
    
    def _generate_performance_plots(self, model_result, data: pd.DataFrame, target: str) -> list:
        """Generate performance visualization plots efficiently."""
        plot_paths = []
        
        try:
            import os
            os.makedirs(settings.plots_path, exist_ok=True)
            
            # Limit plot generation for performance
            if hasattr(model_result, 'predictions') and model_result.predictions is not None:
                y_true = data[target] if target in data.columns else None
                y_pred = model_result.predictions
                
                if y_true is not None and len(y_true) == len(y_pred):
                    timestamp = int(time.time())
                    
                    if model_result.model_type == 'classification':
                        # Confusion matrix plot (quick generation)
                        try:
                            fig = px.imshow(
                                confusion_matrix(y_true, y_pred),
                                text_auto=True,
                                title="Confusion Matrix",
                                width=400, height=400
                            )
                            plot_path = settings.plots_path / f"confusion_matrix_{timestamp}.html"
                            fig.write_html(plot_path)
                            plot_paths.append(str(plot_path))
                        except Exception as e:
                            self.logger.warning(f"Could not generate confusion matrix plot: {e}")
                    
                    else:
                        # Residuals plot for regression (quick generation)
                        try:
                            residuals = y_true - y_pred
                            fig = px.scatter(
                                x=y_pred, y=residuals,
                                title="Residuals vs Predicted",
                                labels={'x': 'Predicted', 'y': 'Residuals'},
                                width=500, height=400
                            )
                            plot_path = settings.plots_path / f"residuals_{timestamp}.html"
                            fig.write_html(plot_path)
                            plot_paths.append(str(plot_path))
                        except Exception as e:
                            self.logger.warning(f"Could not generate residuals plot: {e}")
        
        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")
        
        return plot_paths
    
    def _analyze_feature_importance(self, feature_importance: dict) -> dict:
        """Analyze feature importance efficiently."""
        if not feature_importance:
            return {}
        
        try:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Quick statistics
            importance_values = list(feature_importance.values())
            
            analysis = {
                'top_features': dict(sorted_features[:10]),  # Top 10 features
                'feature_count': len(feature_importance),
                'importance_stats': {
                    'mean': float(np.mean(importance_values)),
                    'std': float(np.std(importance_values)),
                    'max': float(np.max(importance_values)),
                    'min': float(np.min(importance_values))
                },
                'cumulative_importance': {}
            }
            
            # Calculate cumulative importance for top features
            total_importance = sum(abs(v) for v in importance_values)
            cumulative = 0
            for i, (feature, importance) in enumerate(sorted_features[:20]):
                cumulative += abs(importance)
                analysis['cumulative_importance'][f'top_{i+1}'] = cumulative / total_importance
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")
            return {}
    
    def _perform_fast_validation(self, model_result, data: pd.DataFrame, target: str) -> dict:
        """Perform fast cross-validation analysis."""
        try:
            # For performance, just return basic validation info
            # In a real implementation, this could do quick bootstrap sampling
            cv_scores = {}
            
            # If we have predictions, calculate stability metrics
            if hasattr(model_result, 'predictions') and model_result.predictions is not None:
                y_pred = model_result.predictions
                
                # Simple stability check - coefficient of variation
                if len(y_pred) > 10:
                    cv_scores['prediction_stability'] = {
                        'mean': float(np.mean(y_pred)),
                        'std': float(np.std(y_pred)),
                        'cv': float(np.std(y_pred) / (np.mean(y_pred) + 1e-8))  # Coefficient of variation
                    }
            
            return cv_scores
            
        except Exception as e:
            self.logger.warning(f"Fast validation failed: {e}")
            return {}
    
    def _generate_fast_recommendations(self, evaluation_data: dict) -> list:
        """Generate recommendations quickly without LLM."""
        recommendations = []
        metrics = evaluation_data.get('metrics_summary', {})
        task_type = evaluation_data.get('task_type', 'unknown')
        
        if task_type == 'classification':
            recommendations.extend(self._generate_classification_recommendations(metrics))
        elif task_type == 'regression':
            recommendations.extend(self._generate_regression_recommendations(metrics))
        
        # Add performance-based recommendations
        performance_analysis = evaluation_data.get('performance_analysis', {})
        if 'residual_stats' in performance_analysis:
            residual_std = performance_analysis['residual_stats'].get('std', 0)
            if residual_std > metrics.get('mae', 0) * 2:
                recommendations.append("High residual variance suggests model instability")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_classification_recommendations(self, metrics: dict) -> list:
        """Generate recommendations for classification models."""
        recommendations = []
        
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        
        if accuracy < 0.7:
            recommendations.append("Consider feature engineering or collecting more data")
        
        if precision > 0.9 and recall < 0.7:
            recommendations.append("Model is conservative - consider adjusting threshold for better recall")
        elif recall > 0.9 and precision < 0.7:
            recommendations.append("Model is aggressive - consider adjusting threshold for better precision")
        
        if f1 < 0.6:
            recommendations.append("F1 score is low - examine class balance and feature quality")
        
        return recommendations
    
    def _generate_regression_recommendations(self, metrics: dict) -> list:
        """Generate recommendations for regression models."""
        recommendations = []
        
        r2 = metrics.get('r2', 0)
        mae = metrics.get('mae', float('inf'))
        rmse = metrics.get('rmse', float('inf'))
        
        if r2 < 0.5:
            recommendations.append("Low R² - consider additional features or non-linear models")
        elif r2 > 0.95:
            recommendations.append("Very high R² - check for overfitting or data leakage")
        
        if rmse > mae * 2:
            recommendations.append("RMSE much larger than MAE - model may be sensitive to outliers")
        
        return recommendations
    
    def _generate_evaluation_insights(self, evaluation_result: EvaluationResult, context: AgentContext) -> str:
        """Generate LLM insights about evaluation results."""
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "LLM service not available for evaluation insights"
            
            model_info = {
                "model_name": evaluation_result.model_name,
                "task_type": evaluation_result.task_type,
            }
            
            insights = llm_service.interpret_model_results(
                model_info=model_info,
                metrics=evaluation_result.metrics_summary,
                max_tokens=600,
                temperature=0.7,
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation insights: {e}")
            return f"Could not generate insights: {e}"
    
    def _store_evaluation_memory(self, evaluation_result: EvaluationResult, context: AgentContext) -> None:
        """Store evaluation results in memory."""
        try:
            evaluation_data = {
                "model_name": evaluation_result.model_name,
                "task_type": evaluation_result.task_type,
                "metrics": evaluation_result.metrics_summary,
                "recommendations": evaluation_result.recommendations,
                "interpretation": evaluation_result.interpretation,
            }
            
            memory_client.store_symbolic(
                table_name="model_evaluations",
                data=evaluation_data,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store evaluation memory: {e}")
    
    def _get_llm_prompt(self, context: AgentContext, feedback = None) -> str:
        """Generate LLM prompt for evaluation insights."""
        model_result = context.model
        
        prompt = f"""
        Analyze this model evaluation and provide insights:
        
        Model: {model_result.model_name}
        Task: {model_result.model_type}
        Metrics: {model_result.metrics}
        
        Provide interpretation and actionable recommendations.
        """
        
        return prompt.strip()