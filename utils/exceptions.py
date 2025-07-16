"""Custom exceptions for Data Science Agent."""


class DataScienceAgentError(Exception):
    """Base exception for Data Science Agent."""
    pass


class AgentError(DataScienceAgentError):
    """Base exception for agent-related errors."""
    pass


class DataIngestionError(AgentError):
    """Exception raised during data ingestion."""
    pass


class DataProfilingError(AgentError):
    """Exception raised during data profiling."""
    pass


class DataCleaningError(AgentError):
    """Exception raised during data cleaning."""
    pass


class FeatureSelectionError(AgentError):
    """Exception raised during feature selection."""
    pass


class FeatureEngineeringError(AgentError):
    """Exception raised during feature engineering."""
    pass


class ModelingError(AgentError):
    """Exception raised during modeling."""
    pass


class EvaluationError(AgentError):
    """Exception raised during model evaluation."""
    pass


class ReportGenerationError(AgentError):
    """Exception raised during report generation."""
    pass


class MemoryError(DataScienceAgentError):
    """Exception raised during memory operations."""
    pass


class OrchestrationError(DataScienceAgentError):
    """Exception raised during pipeline orchestration."""
    pass


class ValidationError(DataScienceAgentError):
    """Exception raised during data validation."""
    pass


class ConfigurationError(DataScienceAgentError):
    """Exception raised for configuration issues."""
    pass


class APIError(DataScienceAgentError):
    """Exception raised for external API issues."""
    pass


class InsufficientDataError(DataScienceAgentError):
    """Exception raised when data is insufficient for processing."""
    pass