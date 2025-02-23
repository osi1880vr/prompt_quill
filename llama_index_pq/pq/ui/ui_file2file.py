import gradio as gr
import globals
from .ui_helpers import create_textbox, create_checkbox, create_button

g = globals.get_globals()

def setup_file2file_tab(batch_run, ui_code):
    components = {}
    with gr.Row():
        with gr.Column(scale=3):
            components['f2f_input_file'] = gr.Files()
        with gr.Column(scale=1):
            components['f2f_status'] = create_textbox("Status", "", "Status")
            components['f2f_summary'] = create_checkbox("Summary", g.settings_data['summary'],
                                                        "Create a summary from the LLM prompt?")
            components['f2f_submit_button'] = create_button("Run Batch")

    components['f2f_submit_button'].click(fn=ui_code.run_batch,
                                          inputs=components['f2f_input_file'],
                                          outputs=components['f2f_status'])
    components['f2f_summary'].change(fn=ui_code.set_summary,
                                     inputs=components['f2f_summary'],
                                     outputs=None)

    return components