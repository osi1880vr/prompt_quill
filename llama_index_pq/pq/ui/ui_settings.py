import gradio as gr
import globals
from .ui_helpers import create_textbox, create_slider, create_button, create_dropdown, create_checkbox

g = globals.get_globals()




def settings_presets(ui_code):
    components = {}
    with gr.Tab("Presets") as presets:
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=3):
                        components['preset_select'] = create_dropdown(
                            "Preset", g.settings_data['preset_list'], g.settings_data['selected_preset']
                        )
                    with gr.Column(scale=1):
                        components['preset_reload_button'] = create_button("Reload presets")
                        components['preset_load_button'] = create_button("Load preset")
                        components['preset_save_button'] = create_button("Save preset")
                with gr.Row():
                    with gr.Column(scale=3):
                        components['preset_name'] = create_textbox("Filename", "", "Enter preset name", lines=1)
                    with gr.Column(scale=1):
                        components['preset_create_button'] = create_button("Create new preset")
            with gr.Column(scale=1):
                components['preset_status'] = create_textbox("Status", "", "Status", lines=1)

        gr.on(triggers=[presets.select], fn=ui_code.load_preset_list, inputs=None, outputs=[components['preset_select']])
        components['preset_load_button'].click(fn=ui_code.load_preset, inputs=components['preset_select'],
                                               outputs=components['preset_status'])
        components['preset_save_button'].click(fn=ui_code.save_preset, inputs=components['preset_select'],
                                               outputs=components['preset_status'])
        components['preset_create_button'].click(fn=ui_code.save_preset, inputs=components['preset_name'],
                                                 outputs=components['preset_status'])
        components['preset_reload_button'].click(fn=ui_code.load_preset_list, inputs=None,
                                                 outputs=[components['preset_select'], components['preset_status']])
    return components

def settings_model_settings(ui_code):
    components = {}
    with gr.Tab("Model Settings") as llm_settings:
        with gr.Row():
            with gr.Column(scale=3):
                components['llm'] = create_dropdown(
                    "LLM Model", g.settings_data['model_list'].keys(), g.settings_data['LLM Model'],
                    info="Will add more LLMs later!"
                )
                components['collection'] = create_dropdown(
                    "Collection", g.settings_data['collections_list'], g.settings_data['collection'],
                    info="If you got more than one collection!"
                )
                components['embedding_model'] = create_dropdown(
                    "Embedding Model", g.settings_data['embedding_model_list'], g.settings_data['embedding_model'],
                    info="If you dont know better, use all-MiniLM-L12-v2!"
                )
                components['temperature'] = create_slider(
                    "Temperature", g.settings_data['Temperature'], min_val=0, max_val=2, step=0.1,
                    info="Choose between 0 and 1"
                )
                components['context'] = create_slider(
                    "Context Length", g.settings_data['Context Length'], min_val=0, max_val=8192, step=1,
                    info="Choose between 1 and 8192"
                )
                components['gpu_layers'] = create_slider(
                    "GPU Layers", g.settings_data['GPU Layers'], min_val=0, max_val=1024, step=1,
                    info="Choose between 1 and 1024"
                )
                components['max_out_token'] = create_slider(
                    "max output Tokens", g.settings_data['max output Tokens'], min_val=0, max_val=10240, step=1,
                    info="Choose between 1 and 10240"
                )
                components['top_k'] = create_slider(
                    "how many entrys to be fetched from the vector store", g.settings_data['top_k'],
                    min_val=0, max_val=g.settings_data['max_top_k'], step=1,
                    info="Choose between 1 and 50 be careful not to overload the context window of the LLM"
                )
            with gr.Column(scale=1):
                components['model_status'] = create_textbox("Status", "", "Status")
                components['model_save_button'] = create_button("Save model settings")

        gr.on(triggers=[llm_settings.select], fn=ui_code.get_llm_settings, inputs=None,
              outputs=[components['collection'], components['llm'], components['embedding_model'],
                       components['temperature'], components['context'], components['gpu_layers'],
                       components['max_out_token'], components['top_k']])
        components['model_save_button'].click(
            fn=ui_code.set_model,
            inputs=[components['collection'], components['llm'], components['embedding_model'],
                    components['temperature'], components['context'], components['gpu_layers'],
                    components['max_out_token'], components['top_k']],
            outputs=components['model_status']
        )
    return components


def settings_advanced_model_settings(ui_code):
    components = {}
    with gr.Tab("Advanced Model Settings") as adv_settings:
        with gr.Row():
            with gr.Column(scale=3):
                components['reset_model'] = create_checkbox(
                    "do we reset the model before each request",
                    g.settings_data['sailing']['reset_model'],
                    "if set to true there no internal buildup of a history, this can be good or bad for the results. try it to see whats best."
                )
                components['top_p'] = create_slider(
                    "Top P", g.settings_data["llm_settings"].get("top_p", 1.0), min_val=0, max_val=1, step=0.05,
                    info="Controls diversity via nucleus sampling (0-1)"
                )
                components['min_p'] = create_slider(
                    "Min P", g.settings_data["llm_settings"].get("min_p", 0.05), min_val=0, max_val=1, step=0.01,
                    info="Minimum probability threshold (0-1)"
                )
                components['typical_p'] = create_slider(
                    "Typical P", g.settings_data["llm_settings"].get("typical_p", 1.0), min_val=0, max_val=1, step=0.05,
                    info="Typical probability for sampling (0-1)"
                )
                components['frequency_penalty'] = create_slider(
                    "Frequency Penalty", g.settings_data["llm_settings"].get("frequency_penalty", 0.0), min_val=-2, max_val=2, step=0.1,
                    info="Penalizes repeated tokens (-2 to 2)"
                )
                components['presence_penalty'] = create_slider(
                    "Presence Penalty", g.settings_data["llm_settings"].get("presence_penalty", 0.0), min_val=-2, max_val=2, step=0.1,
                    info="Penalizes already-present tokens (-2 to 2)"
                )
                components['repeat_penalty'] = create_slider(
                    "Repeat Penalty", g.settings_data["llm_settings"].get("repeat_penalty", 1.0), min_val=0, max_val=2, step=0.05,
                    info="Discourages repetition (0-2)"
                )
                components['top_k'] = create_slider(
                    "Top K", g.settings_data["llm_settings"].get("top_k", 0), min_val=0, max_val=100, step=1,
                    info="Limits sampling to top K tokens (0 disables)"
                )
            with gr.Column(scale=1):
                components['adv_status'] = create_textbox("Status", "", "Status")
                components['adv_save_button'] = create_button("Save advanced settings")

        gr.on(triggers=[adv_settings.select], fn=ui_code.get_advanced_model_settings, inputs=None,
              outputs=[components['top_p'], components['min_p'], components['typical_p'],
                       components['frequency_penalty'], components['presence_penalty'],
                       components['repeat_penalty'], components['top_k'], components['reset_model']])
        components['adv_save_button'].click(
            fn=ui_code.set_advanced_model_settings,
            inputs=[components['top_p'], components['min_p'], components['typical_p'],
                    components['frequency_penalty'], components['presence_penalty'],
                    components['repeat_penalty'], components['top_k'], components['reset_model']],
            outputs=components['adv_status']
        )
    return components


def settings_prompting_settings(ui_code):
    components = {}
    with gr.Tab("Prompting Settings") as defaults:
        with gr.Tab("Negative Prompt"):
            with gr.Row():
                with gr.Column(scale=3):
                    components['neg_prompt_text'] = create_textbox(
                        "Default Negative Prompt", g.settings_data['negative_prompt'], lines=10
                    )
                with gr.Column(scale=1):
                    components['neg_prompt_status'] = create_textbox("Status", "", "Status")
                    components['neg_prompt_submit_button'] = create_button("Save Negative Prompt")
            components['neg_prompt_submit_button'].click(fn=ui_code.set_neg_prompt,
                                                         inputs=components['neg_prompt_text'],
                                                         outputs=components['neg_prompt_status'])

        with gr.Tab("Rephrase instruction"):
            with gr.Row():
                with gr.Column(scale=3):
                    components['rephrase_instruction_text'] = create_textbox(
                        "Rephrase instruction", g.settings_data['rephrase_instruction'], lines=5
                    )
                with gr.Column(scale=1):
                    components['rephrase_instruction_status'] = create_textbox("Status", "", "Status", lines=1)
                    components['rephrase_instruction_submit_button'] = create_button("Save Rephrase instruction")
            components['rephrase_instruction_submit_button'].click(
                fn=ui_code.set_rephrase_instruction,
                inputs=components['rephrase_instruction_text'],
                outputs=components['rephrase_instruction_status']
            )

        with gr.Tab("Character"):
            with gr.Row():
                with gr.Column(scale=3):
                    components['prompt_template'] = gr.TextArea(
                        g.settings_data["prompt_templates"][g.settings_data["selected_template"]], lines=20
                    )
                    components['prompt_template_select'] = create_dropdown(
                        "Template", g.settings_data["prompt_templates"].keys(),
                        g.settings_data["selected_template"], interactive=True
                    )
                with gr.Column(scale=1):
                    components['prompt_template_status'] = create_textbox("Status", "", "Status", lines=1)
                    components['prompt_template_submit_button'] = create_button("Save Character")

            gr.on(triggers=[defaults.select], fn=ui_code.get_prompt_template, inputs=None,
                  outputs=[components['prompt_template']])
            gr.on(triggers=[components['prompt_template_select'].select], fn=ui_code.set_prompt_template_select,
                  inputs=[components['prompt_template_select']], outputs=[components['prompt_template']])
            components['prompt_template_submit_button'].click(
                fn=ui_code.set_prompt_template,
                inputs=[components['prompt_template_select'], components['prompt_template']],
                outputs=[components['prompt_template_status']]
            )
    return components

def setup_settings_tab(ui_code):
    components = {}
    components.update(settings_presets(ui_code))
    components.update(settings_model_settings(ui_code))
    components.update(settings_prompting_settings(ui_code))
    components.update(settings_advanced_model_settings(ui_code))
    return components
