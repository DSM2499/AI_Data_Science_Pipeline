"""Tests for OrchestrationAgent."""

import pytest

from agents.base import AgentContext
from agents.orchestration import OrchestrationAgent, PipelinePhase, PipelineState


class TestOrchestrationAgent:
    """Test cases for OrchestrationAgent."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = OrchestrationAgent()
        
        assert orchestrator.name == "OrchestrationAgent"
        assert orchestrator.current_phase == PipelinePhase.DATA_INGESTION
        assert orchestrator.pipeline_state == PipelineState.INITIALIZED
        assert len(orchestrator.completed_phases) == 0
        assert len(orchestrator.failed_phases) == 0
        assert len(orchestrator.agents) == 0
    
    def test_pipeline_status(self):
        """Test pipeline status reporting."""
        orchestrator = OrchestrationAgent()
        status = orchestrator.get_pipeline_status()
        
        assert isinstance(status, dict)
        assert "session_id" in status
        assert "current_phase" in status
        assert "pipeline_state" in status
        assert "progress_percentage" in status
        assert status["progress_percentage"] == 0.0
        assert status["total_phases"] == len(PipelinePhase)
    
    def test_reset_pipeline(self):
        """Test pipeline reset functionality."""
        orchestrator = OrchestrationAgent()
        original_session_id = orchestrator.session_id
        
        # Simulate some progress
        orchestrator.completed_phases = [PipelinePhase.DATA_INGESTION]
        orchestrator.current_phase = PipelinePhase.DATA_PROFILING
        
        # Reset
        orchestrator.reset_pipeline()
        
        assert orchestrator.session_id != original_session_id
        assert orchestrator.current_phase == PipelinePhase.DATA_INGESTION
        assert orchestrator.pipeline_state == PipelineState.INITIALIZED
        assert len(orchestrator.completed_phases) == 0
        assert len(orchestrator.execution_log) == 0
    
    def test_validation_no_agents(self):
        """Test validation fails when no agents are registered."""
        orchestrator = OrchestrationAgent()
        context = AgentContext()
        
        with pytest.raises(Exception):
            orchestrator.run(context)