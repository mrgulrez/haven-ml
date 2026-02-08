"""Unified logging setup for the Empathy System."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

def setup_logger(
    log_level: str = "INFO",
    log_dir: Optional[str] = "./logs",
    enable_file_logging: bool = True
) -> None:
    """
    Configure loguru logger with console and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_file_logging: Whether to write logs to files
    """
    # Remove default logger
    logger.remove()
    
    # Console logger with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    if enable_file_logging and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # General log file (rotates daily)
        logger.add(
            log_path / "empathy_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="00:00",  # Rotate at midnight
            retention="7 days",
            compression="zip"
        )
        
        # Error-only log file
        logger.add(
            log_path / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
            level="ERROR",
            rotation="00:00",
            retention="30 days",
            compression="zip"
        )
    
    logger.info(f"Logging initialized at {log_level} level")


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)
