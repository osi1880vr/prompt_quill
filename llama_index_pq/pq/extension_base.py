# extension_base.py
class PromptQuillExtension:
    def __init__(self):
        self.name = "Unnamed Extension"
        self.description = "No description provided."
        self.tab_name = None  # If None, creates a new tab; otherwise, extends an existing tab

    def setup(self, tab, *args, **kwargs):
        """Set up the extension UI and logic within the specified tab."""
        raise NotImplementedError("Extension must implement setup()")

    def process(self, input_data, *args, **kwargs):
        """Process input data (optional, for extensions with processing logic)."""
        return input_data  # Default: pass-through