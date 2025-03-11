# ui_generator.py (refactored full version)
import gradio as gr
import globals
from .ui_helpers import create_textbox, create_slider, create_dropdown, create_button, create_gallery, create_checkbox

g = globals.get_globals()

def setup_generator_tab(generator, ui, generator_manager, ui_share):
    components = {}

    with gr.Tab("Automatic 1111 / Forge") as automatic1111:
        with gr.Tab('Generate') as generate:
            with gr.Row():
                with gr.Column(scale=3):
                    components['automa_prompt_input'] = gr.TextArea(g.last_prompt, lines=5, label="Prompt")
                    components['automa_negative_prompt_input'] = gr.TextArea(g.last_negative_prompt, lines=3, label="Negative Prompt")
                    with gr.Row():
                        with gr.Column(scale=2):
                            components['automa_sampler'] = gr.Dropdown(
                                choices=g.settings_data['automa']['automa_samplers'],
                                value=g.settings_data['automa']['automa_sampler'],
                                label='Sampler')
                        with gr.Column(scale=2):
                            components['automa_scheduler'] = gr.Dropdown(
                                choices=g.settings_data['automa']['automa_schedulers'],
                                value=g.settings_data['automa']['automa_scheduler'],
                                label='Scheduler')
                    with gr.Row():
                        with gr.Column(scale=2):
                            components['automa_checkpoint'] = gr.Dropdown(
                                choices=g.settings_data['automa']['automa_checkpoints'],
                                value=g.settings_data['automa']['automa_checkpoint'],
                                label='Checkpoint')
                        with gr.Column(scale=2):
                            components['automa_vae'] = gr.Dropdown(
                                choices=g.settings_data['automa']['automa_vaes'],
                                value=g.settings_data['automa']['automa_vae'],
                                label='VAE')
                    with gr.Row():
                        components['automa_steps'] = gr.Slider(1, 100, step=1, value=g.settings_data['automa']['automa_steps'],
                                                               label="Steps", info="Choose between 1 and 100")
                        components['automa_CFG'] = gr.Slider(0, 20, step=0.1, value=g.settings_data['automa']['automa_cfg_scale'],
                                                             label="CFG Scale", info="Choose between 1 and 20")
                        components['automa_clip_skip'] = gr.Slider(0, 12, step=1, value=g.settings_data['automa']['automa_clip_skip'],
                                                                   label="Clip Skip", info="Choose between 1 and 12")
                    with gr.Row():
                        with gr.Column(scale=3):
                            components['automa_width'] = gr.Slider(1, 2048, step=1, value=g.settings_data['automa']['automa_width'],
                                                                   label="Width", info="Choose between 1 and 2048")
                            components['automa_height'] = gr.Slider(1, 2048, step=1, value=g.settings_data['automa']['automa_height'],
                                                                    label="Height", info="Choose between 1 and 2048")
                        with gr.Column(scale=0):
                            automa_size_button = gr.Button('Switch size', elem_classes="gr-small-button")
                        with gr.Column(scale=1):
                            components['automa_Batch'] = gr.Slider(1, 250, step=1, value=g.settings_data['automa']['automa_batch'],
                                                                   label="Batch", info="The number of simultaneous images in each batch, range from 1-250.")
                            components['automa_n_iter'] = gr.Slider(1, 500, step=1, value=g.settings_data['automa']['automa_n_iter'],
                                                                    label="Iterations", info="The number of sequential batches to be run, range from 1-500.")
                    with gr.Row():
                        components['automa_url'] = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa']['automa_url'])
                        components['automa_save'] = gr.Checkbox(label="Save", info="Save the image?",
                                                                value=g.settings_data['automa']['automa_save'])
                        components['automa_save_on_api_host'] = gr.Checkbox(label="Save on API host",
                                                                            info="Save the image on API host?",
                                                                            value=g.settings_data['automa']['automa_save_on_api_host'])
                with gr.Column(scale=1):
                    components['automa_new_forge'] = gr.Checkbox(label="enable new Forge API", value=g.settings_data['automa']['automa_new_forge'])
                    automa_refresh_button = gr.Button('Refresh')
                    automa_start_button = gr.Button('Generate')
                    automa_result_images = gr.Gallery(label='output images')

            automa_size_button.click(fn=generator_manager.automa_switch_size,
                                     inputs=[components['automa_width'], components['automa_height']],
                                     outputs=[components['automa_width'], components['automa_height']])
            automa_refresh_button.click(fn=generator_manager.automa_refresh,
                                        inputs=None,
                                        outputs=[components['automa_sampler'], components['automa_checkpoint'],
                                                 components['automa_vae'], components['automa_scheduler']])
            automa_start_button.click(fn=generator_manager.run_automatics_generation,
                                      inputs=[components['automa_prompt_input'], components['automa_negative_prompt_input'],
                                              components['automa_sampler'], components['automa_checkpoint'],
                                              components['automa_steps'], components['automa_CFG'],
                                              components['automa_width'], components['automa_height'],
                                              components['automa_Batch'], components['automa_n_iter'],
                                              components['automa_url'], components['automa_save'],
                                              components['automa_save_on_api_host'], components['automa_vae'],
                                              components['automa_clip_skip'], components['automa_scheduler']],
                                      outputs=automa_result_images)
            gr.on(triggers=[comp.change for comp in [components['automa_prompt_input'], components['automa_negative_prompt_input'],
                                                     components['automa_sampler'], components['automa_steps'],
                                                     components['automa_CFG'], components['automa_width'],
                                                     components['automa_height'], components['automa_Batch'],
                                                     components['automa_n_iter'], components['automa_url'],
                                                     components['automa_save'], components['automa_save_on_api_host'],
                                                     components['automa_checkpoint'], components['automa_vae'],
                                                     components['automa_clip_skip'], components['automa_new_forge'],
                                                     components['automa_scheduler']]],
                  fn=generator_manager.set_automa_settings,
                  inputs=[components['automa_prompt_input'], components['automa_negative_prompt_input'],
                          components['automa_sampler'], components['automa_checkpoint'],
                          components['automa_steps'], components['automa_CFG'],
                          components['automa_width'], components['automa_height'],
                          components['automa_Batch'], components['automa_n_iter'],
                          components['automa_url'], components['automa_save'],
                          components['automa_save_on_api_host'], components['automa_vae'],
                          components['automa_clip_skip'], components['automa_new_forge'],
                          components['automa_scheduler']],
                  outputs=None)

        with gr.Tab('Interrogate') as interrogate:
            input_image = gr.Image(type='filepath', height=300)
            output_interrogation = gr.Textbox()
            interrogate_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa']['automa_url'])
            button_interrogate = gr.Button('Interrogate')
            button_interrogate.click(generator_manager.run_automa_interrogation, [input_image, interrogate_url],
                                     output_interrogation)
            input_image.change(generator_manager.run_automa_interrogation, [input_image, interrogate_url],
                               output_interrogation)

        with gr.Tab('Interrogate Batch') as interrogate_batch:
            input_image_gallery = gr.Gallery()
            output_interrogation = gr.Textbox()
            with gr.Row():
                save = gr.Checkbox(label="Save to File", info="Save the whole to a file?", value=True)
                interrogate_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa']['automa_url'])
            button_interrogate = gr.Button('Interrogate')
            button_interrogate.click(generator_manager.run_automa_interrogation_batch,
                                     [input_image_gallery, interrogate_url, save], output_interrogation)

        with gr.Tab('Extensions') as extensions:
            with gr.Tab('Adetailer') as adetailer:
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Tab('Adetailer Stage 1') as adetailer_1:
                            ui_share.generate_ad_block(1)
                        with gr.Tab('Adetailer Stage 2') as adetailer_2:
                            ui_share.generate_ad_block(2)
                        with gr.Tab('Adetailer Stage 3') as adetailer_3:
                            ui_share.generate_ad_block(3)
                        with gr.Tab('Adetailer Stage 4') as adetailer_4:
                            ui_share.generate_ad_block(4)
            with gr.Tab("ADetailer Help"):
                gr.Markdown("""### ADetailer Prompt Tips
- **Basic Prompt**: One prompt applies to all detected faces (e.g., "realistic face").
- **Per-Face Prompts**: Use commas to set unique prompts for each face in detection order (e.g., "smiling face, serious face, angry face").
- **Negative Prompts**: Same deal—global (e.g., "blurry") or per-face (e.g., "blurry, dark circles, distorted").
- **Wildcards**: Add dynamic variety with `*` or custom lists (e.g., "face with * hairstyle") if supported by extensions.
- **Style Control**: Match your base model (e.g., "anime face" for Pony) for seamless blending.
- **Multiple Blocks**: Each block (1-4) can target different areas (e.g., Block 1: faces, Block 2: hands).
- **[SEP]**: You can separate the prompts with [SEP] to apply them in order.
- **[SKIP]**: If the prompt for any part is [SKIP], that part is skipped without inpainting.
- **[PROMPT]**: The [PROMPT] token is replaced with the original prompt (or negative prompt).
- **unknown faces**: positive prompt: generic diverse face, unique features, no known identity negative prompt: famous person, celebrity, cartoon character, specific likeness
""")


            with gr.Tab('Layer Diffusion') as layerdiffuse:
                components['automa_layerdiffuse_enable'] = gr.Checkbox(label="Enable Layer Diffusion",
                                                                       value=g.settings_data['automa']['automa_layerdiffuse_enable'])
                gr.on(triggers=[components['automa_layerdiffuse_enable'].change],
                      fn=generator_manager.set_automa_layerdiffuse,
                      inputs=[components['automa_layerdiffuse_enable']],
                      outputs=None)

    with gr.Tab("HordeAI") as hordeai:
        gr.on(triggers=[hordeai.select], fn=generator_manager.hordeai_get_last_prompt, inputs=None,
              outputs=[ui.hordeai_prompt_input, ui.hordeai_negative_prompt_input, ui.horde_api_key, ui.horde_model,
                       ui.horde_sampler, ui.horde_steps, ui.horde_CFG, ui.horde_width, ui.horde_height, ui.horde_clipskip])
        gr.Interface(
            generator_manager.run_hordeai_generation,
            [ui.hordeai_prompt_input, ui.hordeai_negative_prompt_input, ui.horde_api_key, ui.horde_model,
             ui.horde_sampler, ui.horde_steps, ui.horde_CFG, ui.horde_width, ui.horde_height, ui.horde_clipskip],
            outputs=gr.Image(label="Generated Image"),
            allow_flagging='never',
            flagging_options=None,
        )

    gr.on(triggers=[generator.select], fn=generator_manager.all_get_last_prompt, inputs=None,
          outputs=[ui.hordeai_prompt_input, ui.hordeai_negative_prompt_input, ui.horde_api_key, ui.horde_model,
                   ui.horde_sampler, ui.horde_steps, ui.horde_CFG, ui.horde_width, ui.horde_height, ui.horde_clipskip,
                   components['automa_prompt_input'], components['automa_negative_prompt_input'], components['automa_sampler'],
                   components['automa_steps'], components['automa_CFG'], components['automa_width'], components['automa_height'],
                   components['automa_Batch'], components['automa_n_iter'], components['automa_url'], components['automa_save'],
                   components['automa_save_on_api_host'], components['automa_checkpoint'], components['automa_vae'],
                   components['automa_clip_skip'], components['automa_scheduler']])
    gr.on(triggers=[automatic1111.select, generate.select], fn=generator_manager.automa_get_last_prompt, inputs=None,
          outputs=[components['automa_prompt_input'], components['automa_negative_prompt_input'], components['automa_sampler'],
                   components['automa_steps'], components['automa_CFG'], components['automa_width'], components['automa_height'],
                   components['automa_Batch'], components['automa_n_iter'], components['automa_url'], components['automa_save'],
                   components['automa_save_on_api_host'], components['automa_checkpoint']])

    return components