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


from llama_cpp_hijack import llama_cpp_hijack

hijack = llama_cpp_hijack()

import gradio as gr
import llm_interface_qdrant
from generators.civitai.client import civitai_client
from generators.hordeai.client import hordeai_client
from generators.automatics.client import automa_client
from generators.hordeai.client import hordeai_models
from settings import io

hordeai_model_list = hordeai_models().read_model_list()

import os

interface = llm_interface_qdrant.LLM_INTERFACE()

settings_io = io.settings_io()

settings_data = settings_io.load_settings()

max_top_k = 50


class local_mem:
    def __init__(self):
        self.context_prompt = ''


def set_llm_settings(model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
    settings_data['LLM Model'] = model
    settings_data['Temperature'] = temperature
    settings_data['Context Length'] = n_ctx
    settings_data['GPU Layers'] = n_gpu_layers
    settings_data['max output Tokens'] = max_tokens
    settings_data['top_k'] = top_k
    settings_data['Instruct Model'] = instruct
    settings_io.write_settings(settings_data)


def set_civitai_settings(air, steps, cfg, width, heigth, clipskip):
    settings_data['civitai_Air'] = air
    settings_data['civitai_Steps'] = steps
    settings_data['civitai_CFG Scale'] = cfg
    settings_data['civitai_Width'] = width
    settings_data['civitai_Height'] = heigth
    settings_data['civitai_Clipskip'] = clipskip
    settings_io.write_settings(settings_data)


def set_hordeai_settings(api_key, model, sampler, steps, cfg, width, heigth, clipskip):
    settings_data['horde_api_key'] = api_key
    settings_data['horde_Model'] = model
    settings_data['horde_Sampler'] = sampler
    settings_data['horde_Steps'] = steps
    settings_data['horde_CFG Scale'] = cfg
    settings_data['horde_Width'] = width
    settings_data['horde_Height'] = heigth
    settings_data['horde_Clipskip'] = clipskip
    settings_io.write_settings(settings_data)


def set_automa_settings(sampler, steps, cfg, width, heigth, url, save):
    settings_data['automa_Sampler'] = sampler
    settings_data['automa_Steps'] = steps
    settings_data['automa_CFG Scale'] = cfg
    settings_data['automa_Width'] = width
    settings_data['automa_Height'] = heigth
    settings_data['automa_url'] = url
    settings_data['automa_save'] = save
    settings_io.write_settings(settings_data)


def set_model(model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
    set_llm_settings(model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct)
    model = settings_data['model_list'][model]
    return interface.change_model(model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct)


def all_get_last_prompt():
    return interface.last_prompt, interface.last_negative_prompt, settings_data['civitai_Air'], settings_data[
        'civitai_Steps'], settings_data['civitai_CFG Scale'], settings_data['civitai_Width'], settings_data[
        'civitai_Height'], settings_data['civitai_Clipskip'], interface.last_prompt, interface.last_negative_prompt, \
    settings_data['horde_api_key'], settings_data['horde_Model'], settings_data['horde_Sampler'], settings_data[
        'horde_Steps'], settings_data['horde_CFG Scale'], settings_data['horde_Width'], settings_data['horde_Height'], \
    settings_data['horde_Clipskip'], interface.last_prompt, interface.last_negative_prompt, settings_data[
        'automa_Sampler'], settings_data['automa_Steps'], settings_data['automa_CFG Scale'], settings_data[
        'automa_Width'], settings_data['automa_Height'], settings_data['automa_url'], settings_data['automa_save']


def civitai_get_last_prompt():
    return interface.last_prompt, interface.last_negative_prompt, settings_data['civitai_Air'], settings_data[
        'civitai_Steps'], settings_data['civitai_CFG Scale'], settings_data['civitai_Width'], settings_data[
        'civitai_Height'], settings_data['civitai_Clipskip']


def hordeai_get_last_prompt():
    return interface.last_prompt, interface.last_negative_prompt, settings_data['horde_api_key'], settings_data[
        'horde_Model'], settings_data['horde_Sampler'], settings_data['horde_Steps'], settings_data['horde_CFG Scale'], \
    settings_data['horde_Width'], settings_data['horde_Height'], settings_data['horde_Clipskip']


def automa_get_last_prompt():
    return interface.last_prompt, interface.last_negative_prompt, settings_data['automa_Sampler'], settings_data[
        'automa_Steps'], settings_data['automa_CFG Scale'], settings_data['automa_Width'], settings_data[
        'automa_Height'], settings_data['automa_url'], settings_data['automa_save']


def llm_get_settings():
    return settings_data["LLM Model"], settings_data['Temperature'], settings_data['Context Length'], settings_data[
        'GPU Layers'], settings_data['max output Tokens'], settings_data['top_k'], settings_data['Instruct Model']


def get_prompt_template():
    interface.prompt_template = settings_data["prompt_templates"][settings_data["selected_template"]]
    return settings_data["prompt_templates"][settings_data["selected_template"]]


def set_prompt_template_select(value):
    settings_data['selected_template'] = value
    settings_io.write_settings(settings_data)
    return settings_data["prompt_templates"][value]


def set_prompt_template(selection, prompt_text):
    return_data = interface.set_prompt(prompt_text)
    settings_data["prompt_templates"][selection] = prompt_text
    settings_io.write_settings(settings_data)
    return return_data


def run_civitai_generation(air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip):
    set_civitai_settings(air, steps, cfg, width, heigth, clipskip)
    client = civitai_client()
    return client.request_generation(air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip)


def run_hordeai_generation(prompt, negative_prompt, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
    set_hordeai_settings(api_key, model, sampler, steps, cfg, width, heigth, clipskip)
    client = hordeai_client()
    return client.request_generation(api_key=api_key, prompt=prompt, negative_prompt=negative_prompt,
                                     sampler=sampler, model=model, steps=steps, cfg=cfg, width=width, heigth=heigth,
                                     clipskip=clipskip)


def run_automatics_generation(prompt, negative_prompt, sampler, steps, cfg, width, heigth, url, save):
    set_automa_settings(sampler, steps, cfg, width, heigth, url, save)
    client = automa_client()
    return client.request_generation(prompt=prompt, negative_prompt=negative_prompt,
                                     sampler=sampler, steps=steps, cfg=cfg, width=width, heigth=heigth, url=url,
                                     save=save)


def run_llm_response(query, history):
    return_data = interface.run_llm_response(query, history)
    return return_data


css = """
.gr-image {
  min-width: 60px !important;
  max-width: 60px !important;
  min-heigth: 65px !important;
  max-heigth: 65px !important;  
  
}
.app-title {
  font-size: 50px;
}
"""

civitai_prompt_input = gr.TextArea(interface.last_prompt, lines=10, label="Prompt")
civitai_negative_prompt_input = gr.TextArea(interface.last_negative_prompt, lines=5, label="Negative Prompt")
hordeai_prompt_input = gr.TextArea(interface.last_prompt, lines=10, label="Prompt")
hordeai_negative_prompt_input = gr.TextArea(interface.last_negative_prompt, lines=5, label="Negative Prompt")
automa_prompt_input = gr.TextArea(interface.last_prompt, lines=10, label="Prompt")
automa_negative_prompt_input = gr.TextArea(interface.last_negative_prompt, lines=5, label="Negative Prompt")

LLM = gr.Dropdown(
    settings_data['model_list'].keys(), value=settings_data['LLM Model'], label="LLM Model",
    info="Will add more LLMs later!"
)
Temperature = gr.Slider(0, 1, step=0.1, value=settings_data['Temperature'], label="Temperature",
                        info="Choose between 0 and 1")
Context = gr.Slider(0, 8192, step=1, value=settings_data['Context Length'], label="Context Length",
                    info="Choose between 1 and 8192")
GPU = gr.Slider(0, 1024, step=1, value=settings_data['GPU Layers'], label="GPU Layers",
                info="Choose between 1 and 1024")
max = gr.Slider(0, 1024, step=1, value=settings_data['max output Tokens'], label="max output Tokens",
                info="Choose between 1 and 1024")
top_k = gr.Slider(0, max_top_k, step=1, value=settings_data['top_k'],
                  label="how many entrys to be fetched from the vector store",
                  info="Choose between 1 and 50 be careful not to overload the context window of the LLM")
Instruct = gr.Checkbox(label='Instruct Model', value=settings_data['Instruct Model'])

civitai_Air = gr.TextArea(settings_data['civitai_Air'], lines=1, label="Air")
civitai_Steps = gr.Slider(0, 100, step=1, value=settings_data['civitai_Steps'], label="Steps",
                          info="Choose between 1 and 100")
civitai_CFG = gr.Slider(0, 20, step=0.1, value=settings_data['civitai_CFG Scale'], label="CFG Scale",
                        info="Choose between 1 and 20")
civitai_Width = gr.Slider(0, 2048, step=1, value=settings_data['civitai_Width'], label="Width",
                          info="Choose between 1 and 2048")
civitai_Height = gr.Slider(0, 2048, step=1, value=settings_data['civitai_Height'], label="Height",
                           info="Choose between 1 and 2048")
civitai_Clipskip = gr.Slider(0, 10, step=1, value=settings_data['civitai_Clipskip'], label="Clipskip",
                             info="Choose between 1 and 10")

horde_api_key = gr.TextArea(lines=1, label="API Key", value=settings_data['horde_api_key'], type='password')
horde_Model = gr.Dropdown(choices=hordeai_model_list.keys(), value=settings_data['horde_Model'], label='Model')
horde_Sampler = gr.Dropdown(choices=["k_dpmpp_2s_a", "k_lms", "k_heun", "k_heun", "k_euler", "k_euler_a",
                                     "k_dpm_2", "k_dpm_2_a", "k_dpm_fast", "k_dpm_adaptive", "k_dpmpp_2s_a",
                                     "k_dpmpp_2m", "dpmsolver", "k_dpmpp_sde", "lcm", "DDIM"
                                     ], value=settings_data['horde_Sampler'], label='Sampler')
horde_Steps = gr.Slider(0, 100, step=1, value=settings_data['horde_Steps'], label="Steps",
                        info="Choose between 1 and 100")
horde_CFG = gr.Slider(0, 20, step=0.1, value=settings_data['horde_CFG Scale'], label="CFG Scale",
                      info="Choose between 1 and 20")
horde_Width = gr.Slider(0, 2048, step=1, value=settings_data['horde_Width'], label="Width",
                        info="Choose between 1 and 2048")
horde_Height = gr.Slider(0, 2048, step=1, value=settings_data['horde_Height'], label="Height",
                         info="Choose between 1 and 2048")
horde_Clipskip = gr.Slider(0, 10, step=1, value=settings_data['horde_Clipskip'], label="Clipskip",
                           info="Choose between 1 and 10")

automa_url = gr.TextArea(lines=1, label="API URL", value=settings_data['automa_url'])
automa_Sampler = gr.Dropdown(
    choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a',
             'Euler',
             'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a',
             'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras',
             'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential',
             'DPM fast',
             'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras'
             ], value=settings_data['automa_Sampler'], label='Sampler')
automa_Steps = gr.Slider(0, 100, step=1, value=settings_data['automa_Steps'], label="Steps",
                         info="Choose between 1 and 100")
automa_CFG = gr.Slider(0, 20, step=0.1, value=settings_data['automa_CFG Scale'], label="CFG Scale",
                       info="Choose between 1 and 20")
automa_Width = gr.Slider(0, 2048, step=1, value=settings_data['automa_Width'], label="Width",
                         info="Choose between 1 and 2048")
automa_Height = gr.Slider(0, 2048, step=1, value=settings_data['automa_Height'], label="Height",
                          info="Choose between 1 and 2048")
automa_save = gr.Checkbox(label="Save", info="Save the image?", value=settings_data['automa_save'])

prompt_template = gr.TextArea(settings_data["prompt_templates"][settings_data["selected_template"]], lines=20)
prompt_template_select = gr.Dropdown(choices=settings_data["prompt_templates"].keys(),
                                     value=settings_data["selected_template"], label='Template', interactive=True)


def variable_outputs(k):
    settings_data['top_k'] = int(k)
    interface.set_top_k(settings_data['top_k'])
    k = int(k)
    out = [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (max_top_k - k)
    return out


textboxes = []
local_globals = local_mem()

prompt_input = gr.Textbox(placeholder="Make your prompts more creative", container=False, scale=7, render=False)


def get_context_details(*args):
    context_details = interface.get_context_details()
    textboxes = []
    for detail in context_details:
        t = gr.Textbox(f"{detail}")
        textboxes.append(t)
    if len(textboxes) < len(args):
        x = range(len(textboxes), len(args))
        for n in x:
            textboxes.append('')
    return textboxes


def dive_into(text):
    local_globals.context_prompt = text
    context = interface.retrieve_context(text)

    if len(context) < max_top_k - 1:
        x = range(len(context), max_top_k - 1)
        for n in x:
            context.append('')

    return context  # .append(text)


def set_prompt_input():
    return local_globals.context_prompt


def set_translate(translate):
    settings_data['translate'] = translate
    settings_io.write_settings(settings_data)
    interface.reload_settings()


def set_batch(batch):
    settings_data['batch'] = batch
    settings_io.write_settings(settings_data)
    interface.reload_settings()


with gr.Blocks(css=css) as pq_ui:
    with gr.Row():
        # Image element (adjust width as needed)
        gr.Image(os.path.join(os.getcwd(), "logo/pq_v_small.jpg"), width="20vw", show_label=False,
                 show_download_button=False, container=False, elem_classes="gr-image", )

        # Title element (adjust font size and styling with CSS if needed)
        gr.Markdown("**Prompt Quill**", elem_classes="app-title")  # Add unique ID for potential CSS styling

    with gr.Tab("Chat") as chat:
        with gr.Row():
            translate = gr.Checkbox(label="Translate", info="Translate your native language to english?",
                                    value=settings_data['translate'])
            batch = gr.Checkbox(label="Batch", info="Run every entry from the context as a input prompt?",
                                value=settings_data['batch'])
        gr.ChatInterface(
            interface.run_llm_response,
            chatbot=gr.Chatbot(height=500, render=False),
            textbox=prompt_input,
            theme="soft",
            retry_btn="🔄  Retry",
            undo_btn="↩️ Undo",
            clear_btn="Clear",
        )
        chat.select(set_prompt_input, None, prompt_input)
        translate.change(set_translate, translate, None)
        batch.change(set_batch, batch, None)

    with gr.Tab('Deep Dive') as deep_dive:
        top_k_slider = gr.Slider(1, max_top_k, value=settings_data['top_k'], step=1,
                                 label="How many entries to retrieve:")
        search = gr.Textbox(f"", label=f'Context search')
        textboxes = []
        visible = True
        for i in range(max_top_k - 1):
            if i + 1 > settings_data['top_k']:
                visible = False
            t = gr.Textbox(f"Retrieved context {i}", label=f'Context {i + 1}', visible=visible)
            textboxes.append(t)

        output = textboxes
        # output.append(prompt_input)

        for text in textboxes:
            text.focus(dive_into, text, output)

        deep_dive.select(get_context_details, textboxes, textboxes)
        top_k_slider.change(variable_outputs, top_k_slider, textboxes)
        search.change(dive_into, search, textboxes)

    with gr.Tab("Character") as Character:
        gr.on(
            triggers=[Character.select],
            fn=get_prompt_template,
            inputs=None,
            outputs=[prompt_template]
        )
        gr.on(
            triggers=[prompt_template_select.select],
            fn=set_prompt_template_select,
            inputs=prompt_template_select,
            outputs=[prompt_template]
        )
        gr.Interface(
            set_prompt_template,
            [prompt_template_select, prompt_template, ]
            , outputs=None,
            allow_flagging='never',
            flagging_options=None

        )

    with gr.Tab("Model Settings") as llm_settings:
        gr.on(
            triggers=[llm_settings.select],
            fn=llm_get_settings,
            inputs=None,
            outputs=[LLM,
                     Temperature,
                     Context,
                     GPU,
                     max,
                     top_k,
                     Instruct
                     ]
        )

        gr.Interface(
            set_model,
            [
                LLM,
                Temperature,
                Context,
                GPU,
                max,
                top_k,
                Instruct
            ]
            , outputs="text",
            allow_flagging='never',
            flagging_options=None

        )

    with gr.Tab("Generator") as generator:
        gr.on(
            triggers=[generator.select],
            fn=all_get_last_prompt,
            inputs=None,
            outputs=[civitai_prompt_input,
                     civitai_negative_prompt_input,
                     civitai_Air,
                     civitai_Steps,
                     civitai_CFG,
                     civitai_Width,
                     civitai_Height,
                     civitai_Clipskip,
                     hordeai_prompt_input,
                     hordeai_negative_prompt_input,
                     horde_api_key,
                     horde_Model,
                     horde_Sampler,
                     horde_Steps,
                     horde_CFG,
                     horde_Width,
                     horde_Height,
                     horde_Clipskip,
                     automa_prompt_input,
                     automa_negative_prompt_input,
                     automa_Sampler,
                     automa_Steps,
                     automa_CFG,
                     automa_Width,
                     automa_Height,
                     automa_url
                     ]
        )
        with gr.Tab("Civitai") as civitai:
            gr.Interface(
                run_civitai_generation,
                [
                    civitai_Air,
                    civitai_prompt_input,
                    civitai_negative_prompt_input,
                    civitai_Steps,
                    civitai_CFG,
                    civitai_Width,
                    civitai_Height,
                    civitai_Clipskip
                ]
                , outputs=gr.Image(label="Generated Image"),  # "text",
                allow_flagging='never',
                flagging_options=None,
                # live=True
            )
        with gr.Tab("HordeAI") as hordeai:
            gr.on(
                triggers=[hordeai.select],
                fn=hordeai_get_last_prompt,
                inputs=None,
                outputs=[hordeai_prompt_input,
                         hordeai_negative_prompt_input,
                         horde_api_key,
                         horde_Model,
                         horde_Sampler,
                         horde_Steps,
                         horde_CFG,
                         horde_Width,
                         horde_Height,
                         horde_Clipskip]
            )
            gr.Interface(
                run_hordeai_generation,
                [
                    hordeai_prompt_input,
                    hordeai_negative_prompt_input,
                    horde_api_key,
                    horde_Model,
                    horde_Sampler,
                    horde_Steps,
                    horde_CFG,
                    horde_Width,
                    horde_Height,
                    horde_Clipskip,
                ]
                , outputs=gr.Image(label="Generated Image"),  # "text",
                allow_flagging='never',
                flagging_options=None,
                # live=True
            )
        with gr.Tab("Automatic 1111 / Forge") as automatic1111:
            gr.on(
                triggers=[automatic1111.select],
                fn=automa_get_last_prompt,
                inputs=None,
                outputs=[automa_prompt_input,
                         automa_negative_prompt_input,
                         automa_Sampler,
                         automa_Steps,
                         automa_CFG,
                         automa_Width,
                         automa_Height,
                         automa_url,
                         automa_save]
            )
            gr.Interface(
                run_automatics_generation,
                [automa_prompt_input,
                 automa_negative_prompt_input,
                 automa_Sampler,
                 automa_Steps,
                 automa_CFG,
                 automa_Width,
                 automa_Height,
                 automa_url,
                 automa_save]
                , outputs=gr.Image(label="Generated Image"),  # "text",
                allow_flagging='never',
                flagging_options=None,
                # live=True
            )
if __name__ == "__main__":
    pq_ui.launch(inbrowser=True)  # share=True
