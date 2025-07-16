"""Orchestration Agent - Central controller for the data science pipeline."""

from enum import Enum
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

from agents.base import AgentContext, AgentResult, BaseAgent, UserFeedback
from memory.memory_client import memory_client
from utils.exceptions import OrchestrationError
from utils.logging import get_agent_logger


class PipelinePhase(Enum):
    """Enumeration of pipeline phases."""
    
    DATA_INGESTION = "data_ingestion"
    DATA_PROFILING = "data_profiling"
    DATA_CLEANING = "data_cleaning"
    FEATURE_SELECTION = "feature_selection"
    FEATURE_ENGINEERING = "feature_engineering"
    MODELING = "modeling"
    EVALUATION = "evaluation"
    REPORT_GENERATION = "report_generation"


class PipelineState(Enum):
    """Enumeration of pipeline states."""
    
    INITIALIZED = "initialized"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class OrchestrationAgent(BaseAgent):
    """Central controller that manages pipeline state, routing logic, and agent execution."""
    
    def __init__(self):
        """Initialize the orchestration agent."""
        super().__init__("OrchestrationAgent")
        
        # Pipeline state
        self.current_phase = PipelinePhase.DATA_INGESTION
        self.pipeline_state = PipelineState.INITIALIZED
        self.session_id = str(uuid4())
        
        # Phase queue and execution order
        self.phase_queue = list(PipelinePhase)
        self.completed_phases: List[PipelinePhase] = []
        self.failed_phases: List[PipelinePhase] = []
        
        # Agent registry
        self.agents: Dict[PipelinePhase, BaseAgent] = {}
        
        # Execution history
        self.execution_log: List[Dict[str, Any]] = []
        
        self.logger.info(f"Orchestration agent initialized with session: {self.session_id}")
    
    def register_agent(self, phase: PipelinePhase, agent: BaseAgent) -> None:
        """Register an agent for a specific pipeline phase.
        
        Args:
            phase: Pipeline phase
            agent: Agent instance to register
        """
        self.agents[phase] = agent
        self.logger.info(f"Registered agent {agent.name} for phase {phase.value}")
    
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute the orchestration logic.
        
        This method manages the overall pipeline execution, including:
        - Phase sequencing
        - Agent invocation
        - Snapshot management
        - Error handling
        - User approval workflows
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            AgentResult with pipeline status and next steps
        """
        try:
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            # Handle feedback if provided
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            # Determine next action based on current state
            if self.pipeline_state == PipelineState.WAITING_FOR_APPROVAL:
                return self._handle_approval_state(context)
            
            elif self.pipeline_state in [PipelineState.INITIALIZED, PipelineState.RUNNING]:
                return self._execute_next_phase(context)
            
            elif self.pipeline_state == PipelineState.COMPLETED:
                return self._handle_completion(context)
            
            else:
                raise OrchestrationError(f"Invalid pipeline state: {self.pipeline_state}")
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            self.pipeline_state = PipelineState.FAILED
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Orchestration failed: {e}"],
            )
            
            self._log_execution(result)
            return result
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate that the context is ready for orchestration.
        
        Args:
            context: Pipeline context to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            OrchestrationError: If validation fails
        """
        if self.current_phase == PipelinePhase.DATA_INGESTION:
            # For initial phase, minimal validation
            return True
        
        # For subsequent phases, ensure previous phases completed
        required_phases = self.phase_queue[:self.phase_queue.index(self.current_phase)]
        missing_phases = [p for p in required_phases if p not in self.completed_phases]
        
        if missing_phases:
            raise OrchestrationError(
                f"Cannot execute {self.current_phase.value}. "
                f"Missing prerequisite phases: {[p.value for p in missing_phases]}"
            )
        
        return True
    
    def _execute_next_phase(self, context: AgentContext) -> AgentResult:
        """Execute the next phase in the pipeline.
        
        Args:
            context: Current pipeline context
            
        Returns:
            AgentResult from the executed phase
        """
        if self.current_phase not in self.agents:
            raise OrchestrationError(
                f"No agent registered for phase: {self.current_phase.value}"
            )
        
        self.pipeline_state = PipelineState.RUNNING
        agent = self.agents[self.current_phase]
        
        self.logger.info(
            f"Executing phase: {self.current_phase.value}",
            agent=agent.name,
            session_id=self.session_id,
        )
        
        try:
            # Execute the agent
            agent_result = agent.run(context)
            
            # Log execution
            execution_entry = {
                "phase": self.current_phase.value,
                "agent": agent.name,
                "execution_id": agent_result.execution_id,
                "success": agent_result.success,
                "timestamp": agent_result.timestamp,
                "session_id": self.session_id,
            }
            self.execution_log.append(execution_entry)
            
            if agent_result.success:
                # Store snapshot after successful execution
                snapshot_key = f"{self.current_phase.value}_post_execution"
                agent_result.context.phase_snapshots[snapshot_key] = agent_result.context.dict()
                
                # Mark phase as completed
                self.completed_phases.append(self.current_phase)
                
                # Update orchestration result
                if agent_result.requires_approval:
                    self.pipeline_state = PipelineState.WAITING_FOR_APPROVAL
                    agent_result.user_message = (
                        f"Phase '{self.current_phase.value}' completed successfully. "
                        f"Please review the results and approve to continue."
                    )
                else:
                    # Auto-advance to next phase
                    self._advance_to_next_phase()
                
                # Store execution in memory
                self._store_execution_memory(agent_result)
                
            else:
                # Handle agent failure
                self.failed_phases.append(self.current_phase)
                self.pipeline_state = PipelineState.FAILED
                
                agent_result.user_message = (
                    f"Phase '{self.current_phase.value}' failed: {agent_result.error}. "
                    f"Please review the error and provide feedback to retry."
                )
            
            # Update the result with orchestration metadata
            agent_result.metrics.update({
                "current_phase": self.current_phase.value,
                "pipeline_state": self.pipeline_state.value,
                "completed_phases": [p.value for p in self.completed_phases],
                "session_id": self.session_id,
            })
            
            self._log_execution(agent_result)
            return agent_result
            
        except Exception as e:
            self.logger.error(f"Phase execution failed: {e}")
            self.failed_phases.append(self.current_phase)
            self.pipeline_state = PipelineState.FAILED
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=f"Phase {self.current_phase.value} failed: {e}",
                logs=[f"Phase execution error: {e}"],
                user_message=f"Phase '{self.current_phase.value}' encountered an error. Please review and provide feedback.",
            )
            
            self._log_execution(result)
            return result
    
    def _handle_approval_state(self, context: AgentContext) -> AgentResult:
        """Handle the approval waiting state.
        
        Args:
            context: Current pipeline context
            
        Returns:
            AgentResult with approval status
        """
        return AgentResult(
            success=True,
            agent_name=self.name,
            context=context,
            logs=[f"Waiting for user approval for phase: {self.current_phase.value}"],
            user_message=f"Please review the results of '{self.current_phase.value}' and provide approval.",
            requires_approval=True,
            metrics={
                "current_phase": self.current_phase.value,
                "pipeline_state": self.pipeline_state.value,
                "awaiting_approval": True,
            },
        )
    
    def _handle_completion(self, context: AgentContext) -> AgentResult:
        """Handle pipeline completion.
        
        Args:
            context: Final pipeline context
            
        Returns:
            AgentResult with completion status
        """
        completion_summary = {
            "session_id": self.session_id,
            "completed_phases": [p.value for p in self.completed_phases],
            "failed_phases": [p.value for p in self.failed_phases],
            "total_execution_time": self._calculate_total_execution_time(),
        }
        
        # Store completion summary in memory
        memory_client.store_symbolic(
            table_name="pipeline_completions",
            data=completion_summary,
            source_agent=self.name,
            project_id=context.user_preferences.get("project_id", "default"),
        )
        
        return AgentResult(
            success=True,
            agent_name=self.name,
            context=context,
            logs=["Pipeline execution completed successfully"],
            user_message="Data science pipeline completed successfully! Review your results in the output directory.",
            metrics=completion_summary,
        )
    
    def _advance_to_next_phase(self) -> None:
        """Advance to the next phase in the pipeline."""
        current_index = self.phase_queue.index(self.current_phase)
        
        if current_index < len(self.phase_queue) - 1:
            self.current_phase = self.phase_queue[current_index + 1]
            self.logger.info(f"Advanced to phase: {self.current_phase.value}")
        else:
            self.pipeline_state = PipelineState.COMPLETED
            self.logger.info("Pipeline execution completed")
    
    def _handle_user_feedback(
        self,
        context: AgentContext,
        feedback: UserFeedback,
    ) -> AgentContext:
        """Process user feedback and determine next actions.
        
        Args:
            context: Current context
            feedback: User feedback to process
            
        Returns:
            Updated context
        """
        if feedback.approved:
            if self.pipeline_state == PipelineState.WAITING_FOR_APPROVAL:
                # User approved, advance to next phase
                self._advance_to_next_phase()
                self.pipeline_state = PipelineState.RUNNING
                
                self.logger.info(
                    f"User approved phase: {self.completed_phases[-1].value}",
                    next_phase=self.current_phase.value if self.pipeline_state == PipelineState.RUNNING else "completed",
                )
            
        else:
            # User wants to retry or modify
            if feedback.feedback_text:
                self.logger.info(f"User feedback: {feedback.feedback_text}")
            
            # Handle rollback if requested
            if "rollback" in feedback.overrides:
                rollback_phase = feedback.overrides.get("rollback_to_phase")
                if rollback_phase:
                    self._rollback_to_phase(rollback_phase, context)
        
        # Apply user preferences
        context = super()._handle_user_feedback(context, feedback)
        
        return context
    
    def _rollback_to_phase(
        self,
        target_phase: str,
        context: AgentContext,
    ) -> None:
        """Rollback the pipeline to a specific phase.
        
        Args:
            target_phase: Phase to rollback to
            context: Current context to rollback
        """
        try:
            target_enum = PipelinePhase(target_phase)
            target_index = self.phase_queue.index(target_enum)
            
            # Remove completed phases after the target
            self.completed_phases = [
                p for p in self.completed_phases 
                if self.phase_queue.index(p) <= target_index
            ]
            
            # Clear failed phases
            self.failed_phases = []
            
            # Reset current phase
            self.current_phase = target_enum
            self.pipeline_state = PipelineState.RUNNING
            
            # Clear downstream snapshots
            phases_to_clear = [
                p for p in self.phase_queue 
                if self.phase_queue.index(p) > target_index
            ]
            
            for phase in phases_to_clear:
                snapshot_key = f"{phase.value}_post_execution"
                context.phase_snapshots.pop(snapshot_key, None)
            
            self.logger.info(
                f"Rolled back to phase: {target_phase}",
                cleared_phases=[p.value for p in phases_to_clear],
            )
            
        except ValueError:
            raise OrchestrationError(f"Invalid rollback phase: {target_phase}")
    
    def _store_execution_memory(self, result: AgentResult) -> None:
        """Store execution information in memory for future reference.
        
        Args:
            result: Agent execution result to store
        """
        try:
            execution_data = {
                "phase": self.current_phase.value,
                "agent_name": result.agent_name,
                "execution_id": result.execution_id,
                "success": result.success,
                "metrics": result.metrics,
                "user_interaction": result.requires_approval,
                "session_id": self.session_id,
            }
            
            # Store in symbolic store
            memory_client.store_symbolic(
                table_name="pipeline_executions",
                data=execution_data,
                source_agent=self.name,
                project_id=result.context.user_preferences.get("project_id", "default"),
            )
            
            # Create semantic description for vector store
            description = f"""
            Pipeline execution for phase '{self.current_phase.value}':
            - Agent: {result.agent_name}
            - Success: {result.success}
            - Required approval: {result.requires_approval}
            - Key metrics: {', '.join(f'{k}={v}' for k, v in result.metrics.items() if isinstance(v, (int, float, str)))}
            """
            
            if result.error:
                description += f"\n- Error: {result.error}"
            
            tags = ["pipeline_execution", self.current_phase.value]
            if result.success:
                tags.append("successful_execution")
            else:
                tags.append("failed_execution")
            
            memory_client.store_vector(
                content=description.strip(),
                tags=tags,
                source_agent=self.name,
                project_id=result.context.user_preferences.get("project_id", "default"),
                metadata={"execution_id": result.execution_id},
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store execution memory: {e}")
    
    def _calculate_total_execution_time(self) -> float:
        """Calculate total pipeline execution time.
        
        Returns:
            Total execution time in seconds
        """
        if not self.execution_log:
            return 0.0
        
        start_time = self.execution_log[0]["timestamp"]
        end_time = self.execution_log[-1]["timestamp"]
        
        return (end_time - start_time).total_seconds()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and progress.
        
        Returns:
            Dictionary with pipeline status information
        """
        total_phases = len(self.phase_queue)
        completed_count = len(self.completed_phases)
        progress_percentage = (completed_count / total_phases) * 100
        
        return {
            "session_id": self.session_id,
            "current_phase": self.current_phase.value,
            "pipeline_state": self.pipeline_state.value,
            "progress_percentage": progress_percentage,
            "completed_phases": [p.value for p in self.completed_phases],
            "failed_phases": [p.value for p in self.failed_phases],
            "total_phases": total_phases,
            "execution_log_count": len(self.execution_log),
        }
    
    def _apply_overrides(
        self,
        context: AgentContext,
        overrides: Dict[str, Any],
    ) -> AgentContext:
        """Apply user overrides to orchestration agent.
        
        Args:
            context: Current pipeline context
            overrides: User overrides to apply
            
        Returns:
            Updated context
        """
        # Handle rollback requests
        if "rollback" in overrides:
            rollback_phase = overrides.get("rollback_to_phase")
            if rollback_phase:
                try:
                    self._rollback_to_phase(rollback_phase, context)
                    self.logger.info(f"Applied rollback to phase: {rollback_phase}")
                except Exception as e:
                    self.logger.error(f"Failed to apply rollback: {e}")
            else:
                # General rollback - go back one phase
                if self.completed_phases:
                    last_phase = self.completed_phases[-1]
                    self._rollback_to_phase(last_phase.value, context)
                    self.logger.info(f"Applied general rollback to: {last_phase.value}")
        
        # Handle pipeline state overrides
        if "force_phase" in overrides:
            force_phase = overrides["force_phase"]
            try:
                target_phase = PipelinePhase(force_phase)
                self.current_phase = target_phase
                self.logger.info(f"Forced pipeline to phase: {force_phase}")
            except ValueError:
                self.logger.warning(f"Invalid phase for force override: {force_phase}")
        
        # Handle session management
        if "reset_session" in overrides and overrides["reset_session"]:
            old_session = self.session_id
            self.session_id = str(uuid4())
            self.logger.info(f"Reset session from {old_session} to {self.session_id}")
        
        # Handle pipeline preferences
        if "skip_phases" in overrides:
            skip_phases = overrides["skip_phases"]
            if isinstance(skip_phases, list):
                self.logger.info(f"User requested to skip phases: {skip_phases}")
                # Store skip preferences in context for reference
                context.user_preferences["skip_phases"] = skip_phases
        
        self.logger.info(f"Applied orchestration overrides: {list(overrides.keys())}")
        return context
    
    def reset_pipeline(self) -> None:
        """Reset the pipeline to initial state."""
        self.current_phase = PipelinePhase.DATA_INGESTION
        self.pipeline_state = PipelineState.INITIALIZED
        self.session_id = str(uuid4())
        self.completed_phases = []
        self.failed_phases = []
        self.execution_log = []
        
        self.logger.info(f"Pipeline reset with new session: {self.session_id}")