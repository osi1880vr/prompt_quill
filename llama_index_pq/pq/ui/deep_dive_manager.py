# deep_dive_manager.py
import globals
import gradio as gr
from llm_fw import llm_interface_qdrant

class DeepDiveManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.interface = llm_interface_qdrant.get_interface()
        self.max_top_k = 50  # Matches max_top_k from app.py

    def dive_into(self, text):
        self.g.context_prompt = text
        self.interface.retrieve_context(text)
        context = self.g.last_context_list
        if len(context) < self.max_top_k - 1:
            context.extend([''] * (self.max_top_k - 1 - len(context)))
        return context

    def get_context_details(self, *args):
        context_details = self.g.last_context_list
        textboxes = [gr.Textbox(f"{detail}") for detail in context_details]
        if len(textboxes) < len(args):
            textboxes.extend([''] * (len(args) - len(textboxes)))
        return textboxes

    def variable_outputs(self, k):
        self.g.settings_data['top_k'] = int(k)
        self.interface.set_top_k(self.g.settings_data['top_k'])
        k = int(k)
        return [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (self.max_top_k - k)