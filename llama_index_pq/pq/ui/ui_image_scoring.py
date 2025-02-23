import gradio as gr
from .ui_helpers import create_textbox, create_slider, create_button, create_checkbox

def image_scoring_single(image_score):
    components = {}
    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column(scale=3):
                components['score_image'] = gr.Image(label="Image", type="pil")
            with gr.Column(scale=1):
                components['score_result'] = create_textbox("Image Score", "", "The Score of your Image", lines=1)
                components['score_button'] = create_button("Score Image")
        components['score_button'].click(image_score.get_single_aestetics_score,
                                         components['score_image'],
                                         components['score_result'])
    return components

def image_scoring_folder(image_score):
    components = {}
    with gr.Tab("Image Folder"):
        with gr.Row():
            with gr.Column(scale=3):
                components['score_min_aestetics_level'] = create_slider(
                    "Minimum Score", 7, min_val=0, max_val=10, step=0.1, info="Choose between 1 and 10"
                )
                components['score_keep_structure'] = create_checkbox("Create new Folder", False)
                components['score_output_folder'] = create_textbox("Where to store the scored images", "", lines=1)
                components['score_images'] = gr.File(file_count="directory")
            with gr.Column(scale=1):
                components['score_images_result'] = create_textbox("Status", "", "Status", lines=1)
                components['score_images_button'] = create_button("Score Image")
        components['score_images_button'].click(
            fn=image_score.run_aestetic_prediction,
            inputs=[components['score_images'], components['score_min_aestetics_level'],
                    components['score_keep_structure'], components['score_output_folder']],
            outputs=components['score_images_result']
        )
    return components

def setup_image_scoring_tab(image_score):
    components = {}
    components.update(image_scoring_single(image_score))
    components.update(image_scoring_folder(image_score))
    return components