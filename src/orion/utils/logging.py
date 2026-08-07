"""
Structured Logging Utility

WHY it exists:
Standard `print()` statements get lost in large jobs and lack context (time, file, level).
We use `loguru` for structured, colorful, and file-based logging.
This ensures all outputs are recorded, making debugging easier on remote servers.
"""

import sys
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig

def setup_logger(config: DictConfig, log_dir: str | Path) -> None:
    """
    Configures the Loguru logger based on configuration.
    
    Args:
        config: System configuration containing logging settings.
        log_dir: Directory to save log files.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_level = config.get("logging", {}).get("level", "INFO")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler (stderr)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # Add file handler
    log_file = log_dir / "experiment.log"
    logger.add(
        log_file,
        level="DEBUG", # Always log DEBUG to file for tracing
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB", # Rotate log file if it gets too large
        retention="1 month",
    )
    
    logger.info(f"Logging initialized. Writing to {log_file}")
