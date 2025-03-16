# extensions/chat_enhancer/__init__.py
from extension_base import PromptQuillExtension
import gradio as gr

class ChatEnhancerExtension(PromptQuillExtension):
    def __init__(self):
        self.name = "Chat Enhancer"
        self.description = "Adds a sentiment analysis button to the Chat tab."
        self.tab_name = "Chat"

    def setup(self, tab, ui, chat_manager):
        with tab:
            sentiment_btn = gr.Button("Analyze Sentiment")
            sentiment_output = gr.Textbox(label="Sentiment Result")
            sentiment_btn.click(
                fn=self.process,
                inputs=ui.prompt_input,
                outputs=sentiment_output
            )

    def process(self, input_data, *args, **kwargs):
        return f"Sentiment of '{input_data}': Positive"