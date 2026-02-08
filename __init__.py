"""Empathy System - Multimodal Affective Computing Platform."""

__version__ = "0.1.0"
__author__ = "Empathy System Team"

from .config import config
from .utils.logger import setup_logger, get_logger

# Initialize logging
setup_logger(
    log_level=config.get('system.log_level', 'INFO'),
    log_dir=config.get('system.log_dir', './logs')
)

__all__ = ['config', 'setup_logger', 'get_logger']
