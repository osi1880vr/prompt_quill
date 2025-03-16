# extensions/fun_box/chat_inject.py
from extension_base import PromptQuillExtension
import gradio as gr
from .helpers import add_silliness

class ChatSillyExtension(PromptQuillExtension):
    def __init__(self):
        self.name = "Chat Silly Button"
        self.description = "Adds a silliness button to the Chat tab."
        self.tab_name = "Chat"

    def setup(self, tab, ui, chat_manager):
        with tab:
            silly_btn = gr.Button("Make it Silly!")
            silly_out = gr.Textbox(label="Silly Output")
            silly_btn.click(
                fn=self.process,
                inputs=ui.prompt_input,
                outputs=silly_out
            )

    def process(self, input_data, *args, **kwargs):
        return add_silliness(input_data)