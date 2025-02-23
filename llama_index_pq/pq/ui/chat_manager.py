# chat_manager.py
import globals
from settings.io import settings_io
from llm_fw import llm_interface_qdrant

class ChatManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.settings_io = settings_io()
        self.interface = llm_interface_qdrant.get_interface()

    def run_llm_response(self, query, history):
        prompt = self.interface.run_llm_response(query, history)
        return prompt

    def set_prompt_input(self):
        return self.g.context_prompt

    def set_translate(self, translate):
        self.g.settings_data['translate'] = translate
        self.settings_io.write_settings(self.g.settings_data)

    def set_batch(self, batch):
        self.g.settings_data['batch'] = batch
        self.settings_io.write_settings(self.g.settings_data)

    def set_summary(self, summary):
        self.g.settings_data['summary'] = summary
        self.settings_io.write_settings(self.g.settings_data)