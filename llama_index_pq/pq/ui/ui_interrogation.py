import gradio as gr
import globals
from .ui_helpers import create_textbox, create_slider, create_button, create_checkbox, create_gallery

g = globals.get_globals()

def interrogation_file_renamer():
    components = {}
    with gr.Tab("File rename"):
        with gr.Row():
            with gr.Column(scale=3):
                components['molmo_folder_name'] = create_textbox(
                    "The path and all its substructures will be processed?",
                    g.settings_data['interrogate']['molmo_folder_name'], "give a path..."
                )
                components['molmo_file_renamer_prompt'] = create_textbox(
                    "File renamer prompt",
                    g.settings_data['interrogate']['molmo_file_renamer_prompt'], "How to make the filename"
                )
            with gr.Column(scale=1):
                components['molmo_folder_status'] = create_textbox("Status", "", "status")
                components['molmo_folder_submit'] = create_button("Submit")
                components['molmo_folder_stop'] = create_button("Stop")
        with gr.Row():
            components['sail_max_gallery_size'] = create_slider(
                "Max Gallery size", g.settings_data['sailing']['sail_max_gallery_size'], max_val=500,
                info="Limit the number of images kept in the gallery choose between 1 and 500"
            )
    return components

def interrogation_story_teller():
    components = {}
    with gr.Tab("Story Teller"):
        components['molmo_story_teller_enabled'] = create_checkbox(
            "Enable", g.settings_data['interrogate']['molmo_story_teller_enabled'], "Enable Story telling?"
        )
        components['molmo_story_teller_prompt'] = create_textbox(
            "Story telling prompt", g.settings_data['interrogate']['molmo_story_teller_prompt'],
            "Make it tell a story", scale=3  # Note: scale=3 was in original, but not directly supported here; adjust via CSS if needed
        )
    return components

def interrogation_settings():
    components = {}
    with gr.Tab("Settings"):
        components['molmo_temperatur'] = create_slider(
            "Temperatur", g.settings_data['interrogate']['molmo_temperatur'], min_val=0.1, max_val=3, step=0.1
        )
        components['molmo_max_new_tokens'] = create_slider(
            "Max new Tokens", g.settings_data['interrogate']['molmo_max_new_tokens'], max_val=1000
        )
        components['molmo_top_k'] = create_slider(
            "Top K", g.settings_data['interrogate']['molmo_top_k'], max_val=200, step=0.1
        )
        components['molmo_top_p'] = create_slider(
            "Top P", g.settings_data['interrogate']['molmo_top_p'], min_val=0.1, max_val=1, step=0.1
        )
    return components

def interrogation_file_organizer():
    components = {}
    with gr.Tab("File Organizer"):
        with gr.Row():
            with gr.Column(scale=3):
                components['molmo_source_folder_name'] = create_textbox(
                    "The path and all its substructures will be processed?",
                    g.settings_data['interrogate']['molmo_source_folder_name'], "give a path..."
                )
                components['molmo_destination_folder_name'] = create_textbox(
                    "Destination categories folder",
                    g.settings_data['interrogate']['molmo_destination_folder_name'], "How to make the filename"
                )
                components['molmo_organize_prompt'] = create_textbox(
                    "Prompt for the task",
                    g.settings_data['interrogate']['molmo_organize_prompt'], "How to make the filename"
                )
            with gr.Column(scale=1):
                components['molmo_organize_status'] = create_textbox("Status", "", "status")
                components['molmo_organize_submit'] = create_button("Submit")
                components['molmo_organize_stop'] = create_button("Stop")
    return components

def interrogation_img2txt2img():
    components = {}
    with gr.Tab("Img2Txt2Img"):
        with gr.Row():
            with gr.Column(scale=3):
                components['iti_folder_name'] = create_textbox(
                    "The path and all its substructures will be processed?",
                    g.settings_data['interrogate']['iti_folder_name'], "give a path..."
                )
                components['iti_file_renamer_prompt'] = create_textbox(
                    "File renamer prompt",
                    g.settings_data['interrogate']['iti_file_renamer_prompt'], "How to make the filename"
                )
            with gr.Column(scale=1):
                components['iti_folder_status'] = create_textbox("Status", "", "status")
                components['iti_folder_submit'] = create_button("Submit")
                components['iti_folder_stop'] = create_button("Stop")
        with gr.Row():
            components['iti_result_images'] = create_gallery("output images")
        with gr.Row():
            components['iti_result'] = create_textbox("Your journey journal", "", "Your journey logs", autoscroll=True)
    return components

def setup_interrogation_tab(interrogation, ui_code):
    components = {}  # To store components needed for event handlers

    with gr.Tab("Molmo"):
        components.update(interrogation_file_renamer())
        components.update(interrogation_story_teller())
        components.update(interrogation_settings())
        components.update(interrogation_file_organizer())
        components.update(interrogation_img2txt2img())

    with gr.Tab("PNG Info"):
        with gr.Row():
            with gr.Column(scale=3):
                components['png_info_img'] = gr.Image(type="pil", label="Upload an Image", height=300)
            with gr.Column(scale=1):
                components['png_info_output'] = gr.Markdown(label="Response")
        components['png_info_img'].change(fn=ui_code.png_info_get, inputs=components['png_info_img'],
                                          outputs=components['png_info_output'])

    # Event handlers for Molmo
    molmo_inputs = [
        components[key] for key in [
            'molmo_folder_name', 'molmo_file_renamer_prompt', 'molmo_story_teller_enabled',
            'molmo_story_teller_prompt', 'molmo_temperatur', 'molmo_max_new_tokens',
            'molmo_top_k', 'molmo_top_p', 'molmo_source_folder_name',
            'molmo_destination_folder_name', 'molmo_organize_prompt'
        ]
    ]
    gr.on(triggers=[comp.change for comp in molmo_inputs], fn=ui_code.set_molmo, inputs=molmo_inputs, outputs=None)

    iti_inputs = [components['iti_folder_name'], components['iti_file_renamer_prompt']]
    gr.on(triggers=[comp.change for comp in iti_inputs], fn=ui_code.set_iti, inputs=iti_inputs, outputs=None)

    components['molmo_folder_submit'].click(fn=ui_code.molmo_file_rename, inputs=components['molmo_folder_name'],
                                            outputs=components['molmo_folder_status'])
    components['molmo_folder_stop'].click(fn=ui_code.molmo_file_rename_stop, inputs=None,
                                          outputs=components['molmo_folder_status'])
    components['molmo_organize_submit'].click(fn=ui_code.molmo_organize,
                                              inputs=[components['molmo_source_folder_name'], components['molmo_destination_folder_name']],
                                              outputs=components['molmo_folder_status'])
    components['molmo_organize_stop'].click(fn=ui_code.molmo_file_rename_stop, inputs=None,
                                            outputs=components['molmo_organize_status'])
    components['iti_folder_submit'].click(fn=ui_code.run_iti, inputs=components['iti_folder_name'],
                                          outputs=[components['iti_result'], components['iti_result_images'], components['iti_folder_status']])
    components['iti_folder_stop'].click(fn=ui_code.molmo_file_rename_stop, inputs=None,
                                        outputs=components['iti_folder_status'])

    return components  # Return components in case they’re needed elsewhere