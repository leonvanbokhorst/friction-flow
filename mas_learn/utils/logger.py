from typing import Optional
import logging
import sys
from pathlib import Path
import json

class MASLogger:
    def __init__(self):
        self.console_logger = self._setup_console_logger()
        self.file_logger = self._setup_file_logger()
        self.agent_loggers = {}  # Store agent-specific loggers
        
    def _setup_console_logger(self) -> logging.Logger:
        console_logger = logging.getLogger('mas_console')
        console_logger.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Simple format for console - just the message
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        
        console_logger.addHandler(console_handler)
        return console_logger

    def _setup_file_logger(self) -> logging.Logger:
        file_logger = logging.getLogger('mas_debug')
        file_logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler('logs/mas_debug.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed format for file logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        file_logger.addHandler(file_handler)
        return file_logger

    def console(self, message: str):
        """Log message to console"""
        self.console_logger.info(message)

    def debug(self, message: str, component: Optional[str] = None):
        """Log debug message to file"""
        if component:
            message = f"[{component}] {message}"
        self.file_logger.debug(message)

    def error(self, message: str, component: Optional[str] = None):
        """Log error to both console and file"""
        if component:
            message = f"[{component}] {message}"
        self.console_logger.error(f"ERROR: {message}")
        self.file_logger.error(message)

    def setup_agent_logger(self, agent_name: str) -> logging.Logger:
        """Create a dedicated logger for an agent"""
        if agent_name not in self.agent_loggers:
            agent_logger = logging.getLogger(f'agent_{agent_name}')
            agent_logger.setLevel(logging.DEBUG)
            
            # Create agent-specific log file
            log_dir = Path("logs/agents")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(f'logs/agents/{agent_name}.log')
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            agent_logger.addHandler(file_handler)
            self.agent_loggers[agent_name] = agent_logger
            
        return self.agent_loggers[agent_name]
        
    def agent_activity(self, agent_name: str, action: str, details: dict = None):
        """Log agent activity with structured details"""
        if agent_name not in self.agent_loggers:
            self.setup_agent_logger(agent_name)
            
        message = f"[{agent_name}] {action}"
        if details:
            message += f": {json.dumps(details, indent=2)}"
            
        self.agent_loggers[agent_name].info(message)
        self.console_logger.info(f"🤖 {message}")

    def track_progress(self, component: str, phase: str, details: dict = None):
        """Track execution progress with detailed logging"""
        message = f"[{component}] {phase}"
        if details:
            message += f": {json.dumps(details, indent=2)}"
        
        # Log to both console and file
        self.console_logger.info(message)
        self.file_logger.info(message)
        
        # If component has a specific logger, log there too
        if component in self.agent_loggers:
            self.agent_loggers[component].info(message)

# Global logger instance
mas_logger = MASLogger() 