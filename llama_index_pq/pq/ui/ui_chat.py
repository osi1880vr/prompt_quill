import gradio as gr
import globals
from .ui_helpers import create_checkbox

g = globals.get_globals()

def setup_chat_tab(chat, ui, ui_code):
    with gr.Row():
        translate = create_checkbox("Translate", g.settings_data['translate'],
                                    "Translate your native language to english?")
        batch = create_checkbox("Batch", g.settings_data['batch'],
                                "Run every entry from the context as a input prompt?")
        summary = create_checkbox("Summary", g.settings_data['summary'],
                                  "Create a summary from the LLM prompt?")

    gr.ChatInterface(
        ui_code.run_llm_response,
        textbox=ui.prompt_input,
        chatbot=gr.Chatbot(height=500, render=False),
        theme="soft",
        retry_btn="🔄  Retry",
        undo_btn="↩️ Undo",
        clear_btn="Clear",
    )

    chat.select(ui_code.set_prompt_input, None, ui.prompt_input)
    translate.change(ui_code.set_translate, translate, None)
    batch.change(ui_code.set_batch, batch, None)
    summary.change(ui_code.set_summary, summary, None)