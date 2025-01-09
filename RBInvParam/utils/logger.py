import logging
import sys
from pathlib import Path
from typing import Union
from datetime import datetime


# ANSI escape codes for colors
COLORS = {
    'DEBUG': '\033[94m',    # Blue
    'INFO': '',             # Default color
    'WARNING': '\033[93m',  # Yellow
    'ERROR': '\033[91m',    # Red
    'CRITICAL': '\033[95m', # Magenta
    'RESET': '\033[0m'      # Reset
}


class ColoredFormatter(logging.Formatter):    
    def format(self, record):
        # Get the original message
        message = super().format(record)
        # Add color based on the log level
        color = COLORS.get(record.levelname, COLORS['RESET'])
        reset = COLORS['RESET']
        return f"{color}{message}{reset}"

class PlainFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record)


def get_default_logger(logger_name: str = None,
                       logfile_path: Union[str, Path] = None,
                       use_timestemp: bool = False) -> logging.Logger:

    # if not logfile_path:
    #     logfile_path = Path('./logs/log')

    if logfile_path:
        logfile_path = Path(logfile_path)
        assert logfile_path.parent.exists()

        if use_timestemp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_with_timestamp = f"{timestamp}_{logfile_path.stem}{logfile_path.suffix}"
            logfile_path = logfile_path.parent / filename_with_timestamp
        
        assert not logfile_path.exists()
    
    log_format = '[%(asctime)s][%(funcName)s] - %(message)s'

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    colored_formatter = ColoredFormatter(log_format)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

    if logfile_path:
        file_handler = logging.FileHandler(logfile_path)
        plain_formatter = PlainFormatter(log_format)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    return logger