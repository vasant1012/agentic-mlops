import os
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler

now = datetime.now()
time_for_filename = now.strftime("%Y-%m-%d")

# Ensure 'logs' folder exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file_path = os.path.join(
    log_dir, f"feature_store_{time_for_filename}_IST.log")

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Log format
formatter = logging.Formatter(
    fmt="%(asctime)s | %(filename)s | %(module)s | line:%(lineno)d | %(levelname)s | %(message)s",  # NOQA E501
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Stream Handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Timed Rotating File Handler (new file every hour)
file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="H",               # Rotate every hour
    interval=1,
    backupCount=48,         # Keep last 48 hours of logs
    encoding="utf-8",
    utc=False               # Set to True if you want UTC time
)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers
if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
