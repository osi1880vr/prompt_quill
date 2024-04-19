# Copyright 2023 osiworx

# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

import os
import globals
from settings.io import settings_io
g = globals.get_globals()
g.settings_data = settings_io().load_settings()


import gradio as gr

from llama_cpp_hijack import llama_cpp_hijack

hijack = llama_cpp_hijack()



from ui import ui_actions,ui_staff
from style import style
css = style

ui = ui_staff()
ui_code = ui_actions()

max_top_k = 50
textboxes = []



with gr.Blocks(css=css, title='Prompt Quill') as pq_ui:
    with gr.Row():
        # Image element (adjust width as needed)
        gr.Image(os.path.join(os.getcwd(), "logo/pq_v_small.jpg"), width="20vw", show_label=False,
                 show_download_button=False, container=False, elem_classes="gr-image", )

        # Title element (adjust font size and styling with CSS if needed)
        gr.Markdown("**Prompt Quill**", elem_classes="app-title")  # Add unique ID for potential CSS styling

    with gr.Tab("Chat") as chat:
        with gr.Row():
            translate = gr.Checkbox(label="Translate", info="Translate your native language to english?",
                                    value=g.settings_data['translate'])
            batch = gr.Checkbox(label="Batch", info="Run every entry from the context as a input prompt?",
                                value=g.settings_data['batch'])
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

    with gr.Tab('Deep Dive') as deep_dive:
        top_k_slider = gr.Slider(1, max_top_k, value=g.settings_data['top_k'], step=1,
                                 label="How many entries to retrieve:")
        search = gr.Textbox(f"", label=f'Context search')
        textboxes = []
        visible = True
        for i in range(max_top_k - 1):
            if i + 1 > g.settings_data['top_k']:
                visible = False
            t = gr.Textbox(f"Retrieved context {i}", label=f'Context {i + 1}', visible=visible)
            textboxes.append(t)

        output = textboxes
        # output.append(prompt_input)

        for text in textboxes:
            text.focus(ui_code.dive_into, text, output)

        deep_dive.select(ui_code.get_context_details, textboxes, textboxes)
        top_k_slider.change(ui_code.variable_outputs, top_k_slider, textboxes)
        search.change(ui_code.dive_into, search, textboxes)

    with gr.Tab("File2File") as batch_run:
        input_file = gr.Files()
        finish = gr.Textbox(f"", label=f'Wait for its done',placeholder='Here you upload a file with your input prompts, it will then generate a new prompt based on each line of your file and write that to output file')

        file_submit_button = gr.Button('Run Batch')

        file_submit_button.click(ui_code.run_batch,input_file,finish)

    with gr.Tab("Generator") as generator:
        gr.on(
            triggers=[generator.select],
            fn=ui_code.all_get_last_prompt,
            inputs=None,
            outputs=[ui.civitai_prompt_input,
                     ui.civitai_negative_prompt_input,
                     ui.civitai_Air,
                     ui.civitai_Steps,
                     ui.civitai_CFG,
                     ui.civitai_Width,
                     ui.civitai_Height,
                     ui.civitai_Clipskip,
                     ui.hordeai_prompt_input,
                     ui.hordeai_negative_prompt_input,
                     ui.horde_api_key,
                     ui.horde_Model,
                     ui.horde_Sampler,
                     ui.horde_Steps,
                     ui.horde_CFG,
                     ui.horde_Width,
                     ui.horde_Height,
                     ui.horde_Clipskip,
                     ui.automa_prompt_input,
                     ui.automa_negative_prompt_input,
                     ui.automa_Sampler,
                     ui.automa_Steps,
                     ui.automa_CFG,
                     ui.automa_Width,
                     ui.automa_Height,
                     ui.automa_url
                     ]
        )
        with gr.Tab("Automatic 1111 / Forge") as automatic1111:
            with gr.Tab('Generate') as generate:
                gr.on(
                    triggers=[ui.automa_Sampler.change,
                              ui.automa_Steps.change,
                              ui.automa_CFG.change,
                              ui.automa_Width.change,
                              ui.automa_Height.change,
                              ui.automa_Batch.change,
                              ui.automa_n_iter.change,
                              ui.automa_url.change,
                              ui.automa_save.change,
                              ui.automa_save_on_api_host.change],
                    fn=ui_code.set_automa_settings,
                    inputs=[ui.automa_Sampler,
                            ui.automa_Steps,
                            ui.automa_CFG,
                            ui.automa_Width,
                            ui.automa_Height,
                            ui.automa_Batch,
                            ui.automa_n_iter,
                            ui.automa_url,
                            ui.automa_save,
                            ui.automa_save_on_api_host],
                    outputs=None)
                gr.on(
                    triggers=[automatic1111.select,generate.select],
                    fn=ui_code.automa_get_last_prompt,
                    inputs=None,
                    outputs=[ui.automa_prompt_input,
                             ui.automa_negative_prompt_input,
                             ui.automa_Sampler,
                             ui.automa_Steps,
                             ui.automa_CFG,
                             ui.automa_Width,
                             ui.automa_Height,
                             ui.automa_Batch,
                             ui.automa_n_iter,
                             ui.automa_url,
                             ui.automa_save,
                             ui.automa_save_on_api_host]
                )
                gr.Interface(
                    ui_code.run_automatics_generation,
                    [ui.automa_prompt_input,
                     ui.automa_negative_prompt_input,
                     ui.automa_Sampler,
                     ui.automa_Steps,
                     ui.automa_CFG,
                     ui.automa_Width,
                     ui.automa_Height,
                     ui.automa_Batch,
                     ui.automa_n_iter,
                     ui.automa_url,
                     ui.automa_save,
                     ui.automa_save_on_api_host]
                    , outputs=gr.Gallery(label="Generated Image"),  # "text",
                    allow_flagging='never',
                    flagging_options=None
                    # live=True
                )

            with gr.Tab('Interrogate') as interrogate:
                input_image = gr.Image(type='filepath')
                output_interrogation = gr.Textbox()
                interrogate_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa_url'])
                button_interrogate = gr.Button('Interrogate')
                button_interrogate.click(ui_code.run_automa_interrogation,[input_image,interrogate_url],output_interrogation)
            with gr.Tab('Interrogate Batch') as interrogate_batch:
                input_image_gallery = gr.Gallery()
                output_interrogation = gr.Textbox()
                with gr.Row():
                    save = gr.Checkbox(label="Save to File", info="Save the whole to a file?",value=True)
                    interrogate_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa_url'])
                button_interrogate = gr.Button('Interrogate')
                button_interrogate.click(ui_code.run_automa_interrogation_batch,[input_image_gallery,interrogate_url,save],output_interrogation)
        with gr.Tab("Civitai") as civitai:
            gr.Interface(
                ui_code.run_civitai_generation,
                [
                    ui.civitai_Air,
                    ui.civitai_prompt_input,
                    ui.civitai_negative_prompt_input,
                    ui.civitai_Steps,
                    ui.civitai_CFG,
                    ui.civitai_Width,
                    ui.civitai_Height,
                    ui.civitai_Clipskip
                ]
                , outputs=gr.Image(label="Generated Image"),  # "text",
                allow_flagging='never',
                flagging_options=None,
                # live=True
            )
        with gr.Tab("HordeAI") as hordeai:
            gr.on(
                triggers=[hordeai.select],
                fn=ui_code.hordeai_get_last_prompt,
                inputs=None,
                outputs=[ui.hordeai_prompt_input,
                         ui.hordeai_negative_prompt_input,
                         ui.horde_api_key,
                         ui.horde_Model,
                         ui.horde_Sampler,
                         ui.horde_Steps,
                         ui.horde_CFG,
                         ui.horde_Width,
                         ui.horde_Height,
                         ui.horde_Clipskip]
            )
            gr.Interface(
                ui_code.run_hordeai_generation,
                [
                    ui.hordeai_prompt_input,
                    ui.hordeai_negative_prompt_input,
                    ui.horde_api_key,
                    ui.horde_Model,
                    ui.horde_Sampler,
                    ui.horde_Steps,
                    ui.horde_CFG,
                    ui.horde_Width,
                    ui.horde_Height,
                    ui.horde_Clipskip,
                ]
                , outputs=gr.Image(label="Generated Image"),  # "text",
                allow_flagging='never',
                flagging_options=None,
                # live=True
            )



    with gr.Tab("Sail the data ocean") as sailor:


        with gr.Tab('Sailing') as sailing:

            with gr.Row():
                with gr.Column(scale=3):
                    sail_text = gr.Textbox("", label=f'Start your journey with',placeholder="Where do we set our sails",elem_id='sail-input-text')
                with gr.Column(scale=1):
                    with gr.Row():
                        sail_submit_button = gr.Button('Start your journey')
                        sail_stop_button = gr.Button('Interrupt your journey')
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        sail_width = gr.Slider(1, 2048, step=1, value=10, label="Sail steps",info="Choose between 1 and 2048")
                        sail_depth = gr.Slider(1, 2048, step=1, value=10, label="Sail depth",info="Choose between 1 and 2048")
                    with gr.Row():
                        sail_generate = gr.Checkbox(label="Generate with A1111", info="Do you want to directly generate the images?", value=False)
                        sail_check_connect_button = gr.Button('Check API Available')
                        sail_api_avail_ok = gr.Textbox("", label=f'API OK', placeholder="not checked yet")
                        sail_target = gr.Checkbox(label="Follow high distance", info="Which context to follow, the most near or the most distance?", value=True)
                    with gr.Row():
                        sail_sinus = gr.Checkbox(label="Add a sinus to the distance", info="This will create a sinus wave based movement along the distance", value=False)
                        sail_sinus_freq = gr.Slider(0.1, 10, step=0.1, value=0.1, label="Sinus Frequency",info="Choose between 0.1 and 10")
                        sail_sinus_range = gr.Slider(1, 500, step=1, value=10, label="Sinus Multiplier",info="Choose between 1 and 500")
                    with gr.Row():
                        sail_add_style = gr.Checkbox(label="Hard style specification", info="Add a text to each prompt", value=False)
                        sail_style = gr.Textbox("", label=f'Style Spec', placeholder="Enter your hardcoded style")
                    with gr.Row():
                        sail_add_search = gr.Checkbox(label="Hard search specification", info="Add a text to each search", value=False)
                        sail_search = gr.Textbox("", label=f'Search Spec', placeholder="Enter your hardcoded search")
                with gr.Column(scale=1):
                    sail_max_gallery_size = gr.Slider(1, 500, step=1, value=10, label="Max Gallery size",info="Limit the number of images keept in the gallery choose between 1 and 500")
                    sail_result_images = gr.Gallery(label='output images')
            with gr.Row():
                sail_result = gr.Textbox("", label=f'Your journey journal', placeholder="Your journey logs")


            gr.on(
                triggers=[sail_text.change,
                          sail_width.change,
                          sail_depth.change,
                          sail_generate.change,
                          sail_target.change,
                          sail_sinus.change,
                          sail_sinus_freq.change,
                          sail_sinus_range.change,
                          sail_add_style.change,
                          sail_style.change,
                          sail_add_search.change,
                          sail_search.change],
                fn=ui_code.set_sailing_settings,
                inputs=[sail_text,
                        sail_width,
                        sail_depth,
                        sail_generate,
                        sail_target,
                        sail_sinus,
                        sail_sinus_freq,
                        sail_sinus_range,
                        sail_add_style,
                        sail_style,
                        sail_add_search,
                        sail_search],
                outputs = None)
            start_sail = sail_submit_button.click(ui_code.run_t2t_sail,[sail_text,sail_width,sail_depth,sail_target,
                                                                        sail_generate,sail_sinus,sail_sinus_range,sail_sinus_freq,
                                                                        sail_add_style,sail_style,sail_add_search,sail_search,sail_max_gallery_size],[sail_result,sail_result_images])
            sail_stop_button.click(fn=ui_code.stop_t2t_sail, inputs=None, outputs=None, cancels=[start_sail])
            sail_check_connect_button.click(ui_code.check_api_avail,None,sail_api_avail_ok)
        with gr.Tab('Show'):
            with gr.Row():
                sail_show_submit_button = gr.Button('Start your journey')
            with gr.Row():
                sail_show_image = gr.Image(height=1024,width=1024)
            with gr.Row():
                sail_show_result = gr.Textbox("", label=f'Your journey journal', placeholder="Your journey logs",lines=4)

            start_sail_show = sail_show_submit_button.click(ui_code.run_t2t_show_sail,None,
                    [sail_show_result,sail_show_image])



    with gr.Tab('Settings'):
        with gr.Tab("Character") as Character:
            gr.on(
                triggers=[Character.select],
                fn=ui_code.get_prompt_template,
                inputs=None,
                outputs=[ui.prompt_template]
            )
            gr.on(
                triggers=[ui.prompt_template_select.select],
                fn=ui_code.set_prompt_template_select,
                inputs=ui.prompt_template_select,
                outputs=[ui.prompt_template]
            )
            gr.Interface(
                ui_code.set_prompt_template,
                [ui.prompt_template_select, ui.prompt_template, ]
                , outputs=None,
                allow_flagging='never',
                flagging_options=None

            )


        with gr.Tab("Model Settings") as llm_settings:
            gr.on(
                triggers=[llm_settings.select],
                fn=ui_code.llm_get_settings,
                inputs=None,
                outputs=[ui.LLM,
                         ui.Temperature,
                         ui.Context,
                         ui.GPU,
                         ui.max,
                         ui.top_k,
                         ui.Instruct
                         ]
            )

            gr.Interface(
                ui_code.set_model,
                [
                    ui.LLM,
                    ui.Temperature,
                    ui.Context,
                    ui.GPU,
                    ui.max,
                    ui.top_k,
                    ui.Instruct
                ]
                , outputs="text",
                allow_flagging='never',
                flagging_options=None

            )


        with gr.Tab("Default") as defaults:
            with gr.Tab('Negative Prompt') as negative_prompt:
                neg_prompt_text = gr.Textbox(g.settings_data['negative_prompt'], label=f'Default Negative Prompt')
                np_submit_button = gr.Button('Save Negative Prompt')

                np_submit_button.click(ui_code.set_neg_prompt,neg_prompt_text,None)

    #pq_ui.queue(max_size=20)


if __name__ == "__main__":
    pq_ui.launch(favicon_path='logo/favicon32x32.ico', inbrowser=True)  # share=True
