import logging
import sys
from pathlib import Path
from typing import Optional
import logging.handlers
import time

class CustomLogger:
    """Custom logger configuration"""
    
    @staticmethod
    def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Setup logger with custom configuration
        
        Args:
            name: Logger name
            log_file: Optional log file path
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log_file is specified
        if log_file:
            # Create logs directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=7
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

default_logger = CustomLogger.setup_logger(
    'ORC_Project',
    log_file='logs/ORC_Project.log'
)