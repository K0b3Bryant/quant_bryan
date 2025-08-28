"""Utility functions for the project."""

import os
from pathlib import Path
import logging

def setup_logging():
    """Sets up basic logging."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(path: Path):
    """Ensures a directory exists, creating it if necessary."""
    if not path.exists():
        logging.info(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
