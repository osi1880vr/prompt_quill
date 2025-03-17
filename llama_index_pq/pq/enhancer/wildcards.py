import os
import random
from time import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WILDCARD_DIR = os.path.join(SCRIPT_DIR, "wildcards")
AUTOWILDCARD_DIR = os.path.join(WILDCARD_DIR, "autowildcards")

class WildcardCache:
    def __init__(self):
        self.file_list = None
        self.last_count = None
        self.last_check = 0
        self.refresh_interval = 5
        self.auto_swap_data = {}
        self.auto_last_count = None  # Track autowildcards separately
        self.auto_last_check = 0

    def get_file_list(self):
        current_time = time()
        if current_time - self.last_check < self.refresh_interval and self.file_list is not None:
            return self.file_list

        current_count = len([f for f in os.listdir(WILDCARD_DIR) if f.endswith(".txt")])
        if self.file_list is None or current_count != self.last_count:
            self.file_list = [f for f in os.listdir(WILDCARD_DIR) if f.endswith(".txt")]
            self.last_count = current_count
            print(f"Refreshed wildcards cache: {len(self.file_list)} files")
            # Don’t auto-refresh auto_swap_data here; let enhance_prompt handle it

        self.last_check = current_time
        return self.file_list

    def load_wildcards(self, filename, directory=WILDCARD_DIR):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def load_auto_swap_categories(self):
        """Load all .txt files from autowildcards/ as swappable categories, checking for changes."""
        if not os.path.exists(AUTOWILDCARD_DIR):
            os.makedirs(AUTOWILDCARD_DIR)
            print(f"Created {AUTOWILDCARD_DIR}")
            self.auto_last_count = 0
            return

        current_time = time()
        auto_files = [f for f in os.listdir(AUTOWILDCARD_DIR) if f.endswith(".txt")]
        current_auto_count = len(auto_files)

        # Refresh if it’s been 5 seconds or the number of files changed
        if (current_time - self.auto_last_check >= self.refresh_interval or
                self.auto_last_count != current_auto_count or not self.auto_swap_data):
            self.auto_swap_data = {}
            for filename in auto_files:
                category = filename[:-4]  # e.g., "colors", "haircolors"
                self.auto_swap_data[category] = self.load_wildcards(filename, AUTOWILDCARD_DIR)
                if not self.auto_swap_data[category]:
                    print(f"Warning: No data loaded for '{category}' from {filename}")
            self.auto_last_count = current_auto_count
            self.auto_last_check = current_time


    def get_auto_swap_options(self, category):
        return self.auto_swap_data.get(category, [])