import gradio as gr

def create_slider(label, value, min_val=1, max_val=10000, step=1, info=None, **kwargs):
    return gr.Slider(
        minimum=min_val,
        maximum=max_val,
        step=step,
        value=value,
        label=label,
        info=info or f"Choose between {min_val} and {max_val}",
        **kwargs
    )

def create_checkbox(label, value, info=None, **kwargs):
    return gr.Checkbox(label=label, value=value, info=info, **kwargs)

def create_textbox(label, value, placeholder=None, lines=1, interactive=True, **kwargs):
    return gr.Textbox(
        label=label,
        value=value,
        placeholder=placeholder,
        lines=lines,
        interactive=interactive,
        **kwargs
    )

def create_dropdown(label, choices, value, **kwargs):
    return gr.Dropdown(label=label, choices=choices, value=value, **kwargs)

def create_button(label, **kwargs):
    return gr.Button(label, **kwargs)

def create_gallery(label, height=300, rows=1, columns=6, format='png', interactive=True, **kwargs):
    return gr.Gallery(
        label=label,
        height=height,
        rows=rows,
        columns=columns,
        format=format,
        interactive=interactive,
        **kwargs
    )