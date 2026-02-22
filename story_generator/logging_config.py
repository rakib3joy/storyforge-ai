"""
Professional Logging Configuration for AI Story Generator
"""

import logging
import sys
from datetime import datetime
import os

def setup_logging(level=logging.INFO, log_to_file=True):
    """
    Setup professional logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to save logs to file
    """
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s | %(name)s | %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (for development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    if log_to_file:
        # File handler (for production)
        timestamp = datetime.now().strftime('%Y%m%d')
        log_filename = f'logs/story_generator_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Save all details to file
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler (separate file for errors)
        error_filename = f'logs/errors_{timestamp}.log'
        error_handler = logging.FileHandler(error_filename, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    
    # Create logger for our application
    app_logger = logging.getLogger('story_generator')
    app_logger.info("Logging system initialized successfully")
    
    return app_logger

# Performance logging decorator
def log_execution_time(func):
    """Decorator to log function execution time"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f}s: {str(e)}")
            raise
    
    return wrapper

# Initialize logging when module is imported
setup_logging()