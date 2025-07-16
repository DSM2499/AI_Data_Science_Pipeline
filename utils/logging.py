"""Logging utilities for Data Science Agent."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: str = settings.log_level,
    enable_console: bool = True,
) -> None:
    """Set up structured logging with loguru.
    
    Args:
        log_file: Optional file path for log output
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to enable console logging
    """
    # Remove default handler
    logger.remove()
    
    # Console logging
    if enable_console:
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
        )
    
    # File logging
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )
    
    # Add structured logging for agents
    logger.add(
        settings.output_path / "logs" / "agent_activity.jsonl",
        level="INFO",
        format="{time} | {level} | {extra} | {message}",
        serialize=True,
        rotation="1 day",
        retention="30 days",
    )


def get_agent_logger(agent_name: str):
    """Get a logger instance for a specific agent.
    
    Args:
        agent_name: Name of the agent for structured logging
        
    Returns:
        Logger instance with agent context
    """
    return logger.bind(agent=agent_name)


# Initialize logging
setup_logging(
    log_file=settings.output_path / "logs" / "application.log",
    log_level=settings.log_level,
)