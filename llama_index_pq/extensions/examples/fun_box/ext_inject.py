# extensions/fun_box/ext_inject.py
"""Example: Injecting into another extension (chat_enhancer)."""

from extension_base import PromptQuillExtension
import gradio as gr
from .helpers import add_silliness

class ChatEnhancerTwistExtension(PromptQuillExtension):
    def __init__(self):
        self.name = "Chat Enhancer Twist"
        self.description = "Tweaks the Chat Enhancer extension if it’s there."
        self.tab_name = "Chat"  # Same tab as chat_enhancer

    def setup(self, tab, ui, chat_manager):
        """Inject a twist into the Chat tab, alongside chat_enhancer."""
        with tab:
            twist_btn = gr.Button("Twist the Sentiment!")
            twist_out = gr.Textbox(label="Twisted Sentiment")
            twist_btn.click(
                fn=self.process,
                inputs=ui.prompt_input,
                outputs=twist_out
            )

    def process(self, input_data, *args, **kwargs):
        """Add a silly twist to the sentiment idea."""
        return f"Twisted: {add_silliness(input_data)} feels EXTRA happy!"