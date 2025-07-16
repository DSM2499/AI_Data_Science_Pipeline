"""Natural language feedback processing module."""

from typing import Any, Dict, List, Optional

from agents.base import AgentContext, AgentResult, UserFeedback
from utils.llm_service import llm_service
from utils.logging import get_agent_logger


class FeedbackProcessor:
    """Processes natural language feedback and converts it to structured actions."""
    
    def __init__(self):
        """Initialize the feedback processor."""
        self.logger = get_agent_logger("FeedbackProcessor")
    
    def process_natural_language_feedback(
        self,
        feedback_text: str,
        agent_result: AgentResult,
        context: AgentContext,
    ) -> UserFeedback:
        """Process natural language feedback into structured UserFeedback.
        
        Args:
            feedback_text: User's natural language feedback
            agent_result: Current agent result
            context: Current pipeline context
            
        Returns:
            UserFeedback object with processed actions and overrides
        """
        self.logger.info(f"Processing natural language feedback: {feedback_text[:100]}...")
        
        if not llm_service.is_available():
            self.logger.warning("LLM service not available for feedback processing")
            return UserFeedback(
                approved=False,
                feedback_text=feedback_text,
                overrides={},
                preferences={}
            )
        
        try:
            # Prepare context for LLM
            agent_context = self._prepare_agent_context(agent_result, context)
            current_result_summary = self._prepare_result_summary(agent_result)
            
            # Process with LLM
            processed_feedback = llm_service.process_natural_language_feedback(
                feedback_text=feedback_text,
                agent_context=agent_context,
                current_result=current_result_summary
            )
            
            # Convert to structured overrides
            overrides = self._convert_to_overrides(processed_feedback, agent_result, context)
            
            # Generate code if needed
            if processed_feedback.get('code'):
                overrides['custom_engineering_code'] = processed_feedback['code']
            elif processed_feedback.get('intent') in ['add', 'create', 'modify'] and 'code' not in processed_feedback:
                # Generate code from the feedback description
                generated_code = self._generate_code_from_feedback(
                    feedback_text, agent_result, context
                )
                if generated_code and generated_code.strip():
                    overrides['custom_engineering_code'] = generated_code
            
            # Log the processing result
            self.logger.info(
                f"Processed feedback intent: {processed_feedback.get('intent', 'unknown')}",
                actions=processed_feedback.get('actions', []),
                overrides_count=len(overrides)
            )
            
            return UserFeedback(
                approved=False,
                feedback_text=feedback_text,
                overrides=overrides,
                preferences=processed_feedback.get('parameters', {}),
                additional_context={
                    'processed_feedback': processed_feedback,
                    'llm_explanation': processed_feedback.get('explanation', ''),
                    'intent': processed_feedback.get('intent', 'unknown')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing natural language feedback: {e}")
            return UserFeedback(
                approved=False,
                feedback_text=feedback_text,
                overrides={},
                preferences={},
                additional_context={'error': str(e)}
            )
    
    def _prepare_agent_context(
        self,
        agent_result: AgentResult,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Prepare context information for LLM processing."""
        data_info = {}
        available_operations = []
        
        if context.data is not None:
            data_info = {
                'shape': context.data.shape,
                'columns': list(context.data.columns),
                'dtypes': context.data.dtypes.to_dict() if hasattr(context.data, 'dtypes') else {},
                'target': context.target
            }
        
        # Extract available operations based on agent type
        if agent_result.agent_name == "FeatureEngineeringAgent":
            available_operations = [
                "ratio features", "polynomial features", "interaction features",
                "binning", "log transform", "scaling", "aggregation"
            ]
        elif agent_result.agent_name == "DataCleaningAgent":
            available_operations = [
                "remove duplicates", "fill missing values", "remove outliers",
                "drop columns", "data type conversion"
            ]
        elif agent_result.agent_name == "FeatureSelectionAgent":
            available_operations = [
                "correlation filtering", "variance filtering", "univariate selection",
                "recursive feature elimination"
            ]
        
        return {
            'agent_name': agent_result.agent_name,
            'data_info': data_info,
            'available_operations': available_operations
        }
    
    def _prepare_result_summary(self, agent_result: AgentResult) -> Dict[str, Any]:
        """Prepare a summary of the current agent result."""
        summary = {
            'success': agent_result.success,
            'agent': agent_result.agent_name,
            'metrics': agent_result.metrics,
        }
        
        # Add agent-specific result information
        if agent_result.agent_name == "FeatureEngineeringAgent" and agent_result.artifacts:
            engineering_result = agent_result.artifacts.get('engineering_result', {})
            summary.update({
                'operations_applied': len(engineering_result.get('operations_applied', [])),
                'features_added': engineering_result.get('features_added', 0),
                'new_features': engineering_result.get('new_features', [])[:5]  # First 5
            })
        
        return summary
    
    def _convert_to_overrides(
        self,
        processed_feedback: Dict[str, Any],
        agent_result: AgentResult,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Convert processed feedback to agent-specific overrides."""
        overrides = {}
        
        # Handle skip operations
        skip_operations = processed_feedback.get('skip_operations', [])
        if skip_operations:
            if agent_result.agent_name == "FeatureEngineeringAgent":
                overrides['skip_feature_operations'] = skip_operations
            elif agent_result.agent_name == "DataCleaningAgent":
                overrides['skip_operations'] = skip_operations
        
        # Handle parameter adjustments
        parameters = processed_feedback.get('parameters', {})
        for param, value in parameters.items():
            if param in ['max_new_features_ratio', 'correlation_threshold', 'variance_threshold']:
                overrides[param] = value
        
        # Handle specific actions
        actions = processed_feedback.get('actions', [])
        for action in actions:
            if action == 'rollback':
                overrides['rollback'] = True
            elif action == 'skip_all_polynomial':
                if agent_result.agent_name == "FeatureEngineeringAgent":
                    overrides['skip_feature_operations'] = overrides.get('skip_feature_operations', [])
                    overrides['skip_feature_operations'].extend(['polynomial: poly_'])
        
        return overrides
    
    def _generate_code_from_feedback(
        self,
        feedback_text: str,
        agent_result: AgentResult,
        context: AgentContext
    ) -> str:
        """Generate Python code from natural language feedback."""
        if not llm_service.is_available():
            return ""
        
        try:
            # Prepare context for code generation
            code_context = {
                'data_info': {
                    'shape': context.data.shape if context.data is not None else (0, 0),
                    'columns': list(context.data.columns) if context.data is not None else []
                },
                'columns': list(context.data.columns) if context.data is not None else [],
                'target': context.target
            }
            
            # Determine code type based on agent
            code_type = "feature_engineering"
            if agent_result.agent_name == "DataCleaningAgent":
                code_type = "data_cleaning"
            elif agent_result.agent_name == "FeatureSelectionAgent":
                code_type = "feature_selection"
            
            # Generate code
            generated_code = llm_service.generate_code_from_description(
                description=feedback_text,
                context=code_context,
                code_type=code_type
            )
            
            self.logger.info(f"Generated {len(generated_code)} characters of code from feedback")
            return generated_code
            
        except Exception as e:
            self.logger.error(f"Error generating code from feedback: {e}")
            return f"# Error generating code: {e}\n# Original feedback: {feedback_text}"


# Global feedback processor instance
feedback_processor = FeedbackProcessor()