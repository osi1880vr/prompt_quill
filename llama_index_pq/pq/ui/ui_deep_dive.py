import gradio as gr
import globals
from .ui_helpers import create_textbox, create_slider

g = globals.get_globals()

def setup_deep_dive_tab(deep_dive, ui_code, max_top_k):
    components = {}

    components['top_k_slider'] = create_slider(
        "How many entries to retrieve:", g.settings_data['top_k'], min_val=1, max_val=max_top_k, step=1
    )
    components['search'] = create_textbox("Context search", "")

    # Dynamic textboxes
    components['textboxes'] = []
    visible = True
    for i in range(max_top_k - 1):
        if i + 1 > g.settings_data['top_k']:
            visible = False
        t = create_textbox(f"Context {i + 1}", f"Retrieved context {i}", visible=visible)
        components['textboxes'].append(t)

    # Event handlers
    for text in components['textboxes']:
        text.focus(ui_code.dive_into, text, components['textboxes'])

    deep_dive.select(ui_code.get_context_details, components['textboxes'], components['textboxes'])
    components['top_k_slider'].change(ui_code.variable_outputs, components['top_k_slider'], components['textboxes'])
    components['search'].change(ui_code.dive_into, components['search'], components['textboxes'])

    return components