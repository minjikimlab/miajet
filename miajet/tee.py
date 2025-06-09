"""
To redirect print stream output to both console and a log file
"""
import sys
import os
from datetime import datetime

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def set_logging_file(config):
    """
    Makes a unique log file for each run to redirect all output streams to
    """
    # Generate a timestamp string like "20250531_143205" (YYYYMMDD_HHMMSS)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") # this should be a unique timestamp
    log_file = os.path.join(config.save_dir, f"{config.root}_{ts}.log")
    sys.stdout = Tee(log_file)
