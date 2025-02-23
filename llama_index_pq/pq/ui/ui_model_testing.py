import gradio as gr
import globals
from .ui_helpers import create_textbox, create_slider, create_button, create_checkbox, create_gallery, create_dropdown

g = globals.get_globals()

def model_testing_setup_test(ui_code):
    components = {}
    with gr.Tab("Setup test"):
        with gr.Row():
            with gr.Column(scale=3):
                components['model_test_list'] = gr.Dropdown(
                    label="Select the List(s) you want to use",
                    choices=ui_code.prompt_iterator.selectors,
                    multiselect=True,
                    value=g.settings_data['model_test']['model_test_list'],
                    interactive=True
                )
                with gr.Row():
                    components['model_test_type'] = gr.Radio(
                        ['Largest List', 'Full Run'],
                        label="Select type of test, Full run may take very long time",
                        value=g.settings_data['model_test']['model_test_type'],
                        interactive=True
                    )
                    components['model_test_gen_type'] = gr.Radio(
                        ['Largest List', 'Full Run'],
                        label="Select type of test generation params, Full run may take very long time",
                        value=g.settings_data['model_test']['model_test_gen_type'],
                        interactive=True
                    )
                components['model_test_result_images'] = create_gallery("output images")
                components['model_test_sample'] = create_textbox("A sample of your selection", "", "Sample", lines=5)
            with gr.Column(scale=1):
                components['model_test_status'] = create_textbox("Status", "", "Status", lines=1)
                components['model_test_sample_button'] = create_button("Get a sample")
                components['model_test_all_sample_button'] = create_button("Get all samples")
                components['model_test_run_button'] = create_button("Run test")
                components['model_test_stop_button'] = create_button("Stop test")
    return components

def model_testing_generation_settings(ui_code):
    components = {}
    with gr.Tab("Generation Settings"):
        components['model_test_steps'] = ui_code.prompt_iterator.setting_dropdown(
            g.settings_data['model_test']['model_test_steps_list'], "Steps",
            g.settings_data['model_test']['model_test_steps']
        )
        components['model_test_cfg'] = ui_code.prompt_iterator.setting_dropdown(
            g.settings_data['model_test']['model_test_cfg_list'], "CFG",
            g.settings_data['model_test']['model_test_cfg']
        )
        components['model_test_dimensions'] = ui_code.prompt_iterator.setting_dropdown(
            g.settings_data['model_test']['model_test_dimensions_list'], "Image Dimension",
            g.settings_data['model_test']['model_test_dimensions']
        )
    return components

def model_testing_characters(ui_code):
    components = {}
    with gr.Tab("Characters"):
        components['model_test_character'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.character, "Character",
            g.settings_data['model_test']['model_test_setup']['Character']
        )
        components['model_test_celebrities'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.celebrities, "Celebrities",
            g.settings_data['model_test']['model_test_setup']['Celebrities']
        )
        components['model_test_creature_air'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.creature_air, "Air Creatures",
            g.settings_data['model_test']['model_test_setup']['Air Creatures']
        )
        components['model_test_creature_land'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.creature_land, "Land Creatures",
            g.settings_data['model_test']['model_test_setup']['Land Creatures']
        )
        components['model_test_creature_sea'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.creature_sea, "Sea Creatures",
            g.settings_data['model_test']['model_test_setup']['Sea Creatures']
        )
        components['model_test_character_objects'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.character_objects, "Character Objects",
            g.settings_data['model_test']['model_test_setup']['Character Objects']
        )
        components['model_test_character_adj'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.character_adj, "Character Adjectives",
            g.settings_data['model_test']['model_test_setup']['Character Adjectives']
        )
    return components

def model_testing_vehicles(ui_code):
    components = {}
    with gr.Tab("Vehicles"):
        components['model_test_vehicles_air'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.vehicles_air, "Air Vehicle",
            g.settings_data['model_test']['model_test_setup']['Air Vehicle']
        )
        components['model_test_vehicles_land'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.vehicles_land, "Land Vehicle",
            g.settings_data['model_test']['model_test_setup']['Land Vehicle']
        )
        components['model_test_vehicles_sea'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.vehicles_sea, "Sea Vehicle",
            g.settings_data['model_test']['model_test_setup']['Sea Vehicle']
        )
        components['model_test_vehicles_space'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.vehicles_space, "Space Vehicle",
            g.settings_data['model_test']['model_test_setup']['Space Vehicle']
        )
    return components

def model_testing_relations(ui_code):
    components = {}
    with gr.Tab("Relations"):
        components['model_test_moving_relation'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.moving_relation, "Moving relation",
            g.settings_data['model_test']['model_test_setup']['Moving relation']
        )
        components['model_test_still_relation'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.still_relation, "Still relation",
            g.settings_data['model_test']['model_test_setup']['Still relation']
        )
    return components

def model_testing_adjectives(ui_code):
    components = {}
    with gr.Tab("Adjectives"):
        components['model_test_object_adj'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.object_adj, "Object Adjectives",
            g.settings_data['model_test']['model_test_setup']['Object Adjectives']
        )
        components['model_test_visual_adj'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.visual_adj, "Visual Adjectives",
            g.settings_data['model_test']['model_test_setup']['Visual Adjectives']
        )
        components['model_test_visual_qualities'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.visual_qualities, "Visual Qualities",
            g.settings_data['model_test']['model_test_setup']['Visual Qualities']
        )
    return components

def model_testing_settings(ui_code):
    components = {}
    with gr.Tab("Settings"):
        components['model_test_settings'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.settings, "Setup",
            g.settings_data['model_test']['model_test_setup']['Setup']
        )
        components['model_test_things'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.things, "Things",
            g.settings_data['model_test']['model_test_setup']['Things']
        )
    return components

def model_testing_style(ui_code):
    components = {}
    with gr.Tab("Style"):
        components['model_test_colors'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.colors, "Colors",
            g.settings_data['model_test']['model_test_setup']['Colors']
        )
        components['model_test_styles'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.styles, "Styles",
            g.settings_data['model_test']['model_test_setup']['Styles']
        )
        components['model_test_artists'] = ui_code.prompt_iterator.data_dropdown(
            ui_code.prompt_iterator.artists, "Artists",
            g.settings_data['model_test']['model_test_setup']['Artists']
        )
    return components

def model_testing_instruction_prompt():
    components = {}
    with gr.Tab("Instruction Prompt"):
        components['model_test_inst_prompt'] = create_textbox(
            "Prompt instruction for Model test",
            g.settings_data['prompt_templates']['model_test_instruction'], lines=20
        )
    return components

def setup_model_testing_tab(model_test, ui_code):
    components = {}

    components.update(model_testing_setup_test(ui_code))
    components.update(model_testing_generation_settings(ui_code))
    components.update(model_testing_characters(ui_code))
    components.update(model_testing_vehicles(ui_code))
    components.update(model_testing_relations(ui_code))
    components.update(model_testing_adjectives(ui_code))
    components.update(model_testing_settings(ui_code))
    components.update(model_testing_style(ui_code))
    components.update(model_testing_instruction_prompt())

    # Event handlers
    save_inputs = [
        components['model_test_list'], components['model_test_inst_prompt'],
        components['model_test_type'], components['model_test_steps'],
        components['model_test_dimensions'], components['model_test_gen_type'],
        components['model_test_cfg']
    ]
    gr.on(
        triggers=[comp.change for comp in save_inputs],
        fn=ui_code.prompt_iterator.save_test_data,
        inputs=save_inputs,
        outputs=None
    )

    components['model_test_sample_button'].click(
        fn=ui_code.prompt_iterator.get_sample,
        inputs=None,
        outputs=[components['model_test_sample'], components['model_test_status']]
    )
    components['model_test_all_sample_button'].click(
        fn=ui_code.prompt_iterator.get_all_samples,
        inputs=None,
        outputs=[components['model_test_sample'], components['model_test_status']]
    )
    model_test_run = components['model_test_run_button'].click(
        fn=ui_code.run_test,
        inputs=None,
        outputs=[components['model_test_result_images'], components['model_test_status']]
    )
    components['model_test_stop_button'].click(
        fn=ui_code.stop_job,
        inputs=None,
        outputs=None,
        cancels=[model_test_run]
    )

    return components