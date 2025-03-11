import os
import random
from time import time

# Directory with wildcard files
# Set WILDCARD_DIR relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of wildcard_cache.py
WILDCARD_DIR = os.path.join(SCRIPT_DIR, "wildcards")     # wildcards/ folder at same level

class WildcardCache:
    def __init__(self):
        self.file_list = None
        self.last_count = None
        self.last_check = 0
        self.refresh_interval = 5

    def get_file_list(self):
        current_time = time()
        if current_time - self.last_check < self.refresh_interval and self.file_list is not None:
            return self.file_list

        current_count = len([f for f in os.listdir(WILDCARD_DIR) if f.endswith(".txt")])
        if self.file_list is None or current_count != self.last_count:
            self.file_list = [f for f in os.listdir(WILDCARD_DIR) if f.endswith(".txt")]
            self.last_count = current_count
            print(f"Refreshed cache: {len(self.file_list)} files")

        self.last_check = current_time
        return self.file_list

    def load_wildcards(self, filename):
        filepath = os.path.join(WILDCARD_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []


