import logging
import os
from datetime import datetime

class SessionLogger:
    def __init__(self):
        self.log_dir = "logs/sessions"
        # Ensure logs/sessions/ directory exists in the project root
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%M_%H_%d_%m_%Y")
        self.log_path = os.path.join(self.log_dir, f"whisper_session_{timestamp}.log")
        self.logger = logging.getLogger(f"WhisperSessionLogger_{timestamp}")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.handlers = []
        self.logger.addHandler(file_handler)

    def log(self, message: str):
        self.logger.info(message)

    def log_start(self, config: dict):
        self.log("=== SESSION START ===")
        for k, v in config.items():
            self.log(f"CONFIG: {k} = {v}")

    def log_end(self):
        self.log("=== SESSION END ===")
