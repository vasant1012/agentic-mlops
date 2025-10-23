import os
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler

now = datetime.now()
date = now.strftime("%Y-%m-%d")

# Ensure 'logs' folder exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file_path = f"{log_dir}/ml_log_analyzer_logs_{date}.log"

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Log format
formatter = logging.Formatter(
    fmt="%(asctime)s | %(filename)s | %(module)s | line:%(lineno)d | %(levelname)s | %(message)s",  # NOQA E501
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when='midnight',
    backupCount=0,  # Keep only the current day's log
    encoding="utf-8",
    utc=False
)
file_handler.setFormatter(formatter)

# Stream Handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add handlers to the logger, ensuring no duplicates
if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
