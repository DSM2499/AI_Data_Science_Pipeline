"""Base agent class and interfaces for Data Science Agent pipeline."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from utils.exceptions import AgentError
from utils.logging import get_agent_logger


class AgentContext(BaseModel):
    """Context object containing pipeline state and data."""
    
    # Core data
    data: Optional[Any] = None
    target: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing artifacts
    profile_summary: Optional[Dict[str, Any]] = None
    cleaned_data: Optional[Any] = None
    selected_features: Optional[List[str]] = None
    enriched_data: Optional[Any] = None
    model: Optional[Any] = None
    predictions: Optional[Any] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    
    # Pipeline state
    phase_snapshots: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # LLM insights
    llm_insights: Dict[str, str] = Field(default_factory=dict)
    
    # Feature engineering state
    pre_engineering_data: Optional[Any] = None
    user_overrides: Dict[str, Any] = Field(default_factory=dict)
    feedback_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class AgentResult(BaseModel):
    """Result object returned by agents."""
    
    success: bool
    agent_name: str
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Updated context
    context: AgentContext
    
    # Execution details
    logs: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    
    # User interaction
    user_message: Optional[str] = None
    requires_approval: bool = False
    suggestions: List[str] = Field(default_factory=list)
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class UserFeedback(BaseModel):
    """User feedback for agent execution."""
    
    approved: bool = True
    feedback_text: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    custom_code: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all pipeline agents."""
    
    def __init__(self, name: str):
        """Initialize the base agent.
        
        Args:
            name: Human-readable name for the agent
        """
        self.name = name
        self.logger = get_agent_logger(name)
        self._execution_history: List[AgentResult] = []
    
    @abstractmethod
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute the agent's core functionality.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback for refinement
            
        Returns:
            AgentResult with updated context and execution details
            
        Raises:
            AgentError: If execution fails
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate that the context contains required inputs.
        
        Args:
            context: Pipeline context to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            AgentError: If validation fails
        """
        pass
    
    def rollback(self, context: AgentContext) -> AgentContext:
        """Rollback the agent's changes to the context.
        
        Args:
            context: Current context to rollback
            
        Returns:
            Context restored to pre-execution state
        """
        snapshot_key = f"{self.name}_pre_execution"
        if snapshot_key in context.phase_snapshots:
            self.logger.info(f"Rolling back {self.name} changes")
            return AgentContext(**context.phase_snapshots[snapshot_key])
        
        self.logger.warning(f"No rollback snapshot found for {self.name}")
        return context
    
    def _create_snapshot(self, context: AgentContext) -> None:
        """Create a snapshot of the context before execution.
        
        Args:
            context: Context to snapshot
        """
        snapshot_key = f"{self.name}_pre_execution"
        context.phase_snapshots[snapshot_key] = context.dict()
        self.logger.debug(f"Created snapshot: {snapshot_key}")
    
    def _log_execution(self, result: AgentResult) -> None:
        """Log agent execution details.
        
        Args:
            result: Execution result to log
        """
        self._execution_history.append(result)
        
        if result.success:
            self.logger.info(
                f"Agent {self.name} completed successfully",
                execution_id=result.execution_id,
                metrics=result.metrics,
            )
        else:
            self.logger.error(
                f"Agent {self.name} failed: {result.error}",
                execution_id=result.execution_id,
                error=result.error,
            )
    
    def get_execution_history(self) -> List[AgentResult]:
        """Get the execution history for this agent.
        
        Returns:
            List of execution results
        """
        return self._execution_history.copy()
    
    def _handle_user_feedback(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback],
    ) -> AgentContext:
        """Process user feedback and apply overrides.
        
        Args:
            context: Current context
            feedback: User feedback to process
            
        Returns:
            Updated context with feedback applied
        """
        if not feedback:
            return context
        
        # Log feedback processing
        if feedback.feedback_text:
            self.logger.info(
                f"Processing user feedback: {feedback.feedback_text[:100]}...",
                agent=self.name,
                has_overrides=bool(feedback.overrides),
                has_preferences=bool(feedback.preferences)
            )
        
        # Check if this is natural language feedback that needs processing
        if (hasattr(feedback, 'additional_context') and 
            feedback.additional_context and 
            feedback.additional_context.get('processed_feedback')):
            
            processed_feedback = feedback.additional_context['processed_feedback']
            self.logger.info(
                f"Applying LLM-processed feedback with intent: {processed_feedback.get('intent', 'unknown')}",
                explanation=processed_feedback.get('explanation', '')[:100]
            )
        
        # Apply user preferences
        if feedback.preferences:
            context.user_preferences.update(feedback.preferences)
            self.logger.info(
                f"Applied user preferences: {feedback.preferences}",
                agent=self.name,
            )
        
        # Apply overrides (agent-specific implementation)
        if feedback.overrides:
            context = self._apply_overrides(context, feedback.overrides)
        
        # Store feedback context for reference
        context.feedback_history.append({
            'timestamp': self._get_timestamp(),
            'agent': self.name,
            'feedback_text': feedback.feedback_text,
            'overrides': feedback.overrides,
            'preferences': feedback.preferences,
            'additional_context': getattr(feedback, 'additional_context', None)
        })
        
        return context
    
    def _apply_overrides(
        self,
        context: AgentContext,
        overrides: Dict[str, Any],
    ) -> AgentContext:
        """Apply user overrides to the context.
        
        Default implementation - should be overridden by specific agents.
        
        Args:
            context: Current context
            overrides: User overrides to apply
            
        Returns:
            Updated context
        """
        self.logger.warning(
            f"No override handler implemented for {self.name}",
            overrides=overrides,
        )
        return context
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string.
        
        Returns:
            Current timestamp in ISO format
        """
        return datetime.utcnow().isoformat()


class LLMCapableAgent(BaseAgent):
    """Base class for agents that use LLM capabilities."""
    
    def __init__(self, name: str, use_llm: bool = True):
        """Initialize LLM-capable agent.
        
        Args:
            name: Human-readable name for the agent
            use_llm: Whether to enable LLM features
        """
        super().__init__(name)
        self.use_llm = use_llm
    
    @abstractmethod
    def _get_llm_prompt(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> str:
        """Generate LLM prompt for this agent's task.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def _call_llm(
        self,
        prompt: str,
        context: AgentContext,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Call LLM with the given prompt.
        
        Args:
            prompt: Formatted prompt
            context: Pipeline context for additional info
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LLM response text
            
        Raises:
            AgentError: If LLM call fails
        """
        if not self.use_llm:
            return "LLM disabled for this agent"
        
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                self.logger.warning("LLM service not available, returning placeholder")
                return f"LLM service not configured for {self.name}"
            
            response = llm_service.generate_simple_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            self.logger.info(
                f"LLM response generated for {self.name}",
                prompt_length=len(prompt),
                response_length=len(response),
            )
            
            return response
        
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            # Return graceful fallback instead of raising exception
            return f"LLM call failed for {self.name}: {str(e)}"
    
    def _generate_insight(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> str:
        """Generate LLM-based insight for the current context.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            Generated insight text
        """
        if not self.use_llm:
            return ""
        
        prompt = self._get_llm_prompt(context, feedback)
        insight = self._call_llm(prompt, context)
        
        # Store insight in context
        context.llm_insights[self.name] = insight
        
        return insight