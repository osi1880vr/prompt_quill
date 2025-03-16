# extensions/fun_box/new_tab.py
from extension_base import PromptQuillExtension
import gradio as gr

class FunBoxTabExtension(PromptQuillExtension):
    def __init__(self):
        self.name = "Fun Box"
        self.description = "A silly new tab with fun sub-tabs!"
        self.tab_name = None

    def setup(self, tab, *args, **kwargs):
        with tab:
            gr.Markdown("### Welcome to the Fun Box!")
            with gr.Tabs():
                with gr.Tab("Joke Time"):
                    joke_btn = gr.Button("Tell me a Joke!")
                    joke_out = gr.Textbox(label="Here’s your joke!")
                    joke_btn.click(
                        fn=lambda: "Why did the robot laugh? It had a byte of humor!",
                        inputs=None,
                        outputs=joke_out
                    )
                with gr.Tab("Silly Sounds"):
                    sound_btn = gr.Button("Make a Noise!")
                    sound_out = gr.Textbox(label="Listen up!")
                    sound_btn.click(
                        fn=lambda: "BOING! BEEP! WHEEE!",
                        inputs=None,
                        outputs=sound_out
                    )