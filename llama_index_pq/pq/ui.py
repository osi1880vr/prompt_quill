import globals
import os
import gradio as gr
import base64

from generators.civitai.client import civitai_client
from generators.hordeai.client import hordeai_client
from generators.automatics.client import automa_client
from generators.hordeai.client import hordeai_models
from settings.io import settings_io

import llm_interface_qdrant

out_dir = 'api_out'
out_dir_t2t = os.path.join(out_dir, 'txt2txt')
os.makedirs(out_dir_t2t, exist_ok=True)
out_dir_i2t = os.path.join(out_dir, 'img2txt')
os.makedirs(out_dir_i2t, exist_ok=True)

max_top_k = 50


class ui_actions(object):
    def __init__(self):

        self.g = globals.get_globals()
        self.interface = llm_interface_qdrant.LLM_INTERFACE()
        self.settings_io = settings_io()


    async def run_llm_response(self,query, history):
        return await self.interface.run_llm_response(query, history)

    async def set_llm_settings(self, model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
        self.g.settings_data['LLM Model'] = model
        self.g.settings_data['Temperature'] = temperature
        self.g.settings_data['Context Length'] = n_ctx
        self.g.settings_data['GPU Layers'] = n_gpu_layers
        self.g.settings_data['max output Tokens'] = max_tokens
        self.g.settings_data['top_k'] = top_k
        self.g.settings_data['Instruct Model'] = instruct
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()


    async def set_civitai_settings(self, air, steps, cfg, width, heigth, clipskip):
        self.g.settings_data['civitai_Air'] = air
        self.g.settings_data['civitai_Steps'] = steps
        self.g.settings_data['civitai_CFG Scale'] = cfg
        self.g.settings_data['civitai_Width'] = width
        self.g.settings_data['civitai_Height'] = heigth
        self.g.settings_data['civitai_Clipskip'] = clipskip
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()
    
    
    async def set_hordeai_settings(self, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
        self.g.settings_data['horde_api_key'] = api_key
        self.g.settings_data['horde_Model'] = model
        self.g.settings_data['horde_Sampler'] = sampler
        self.g.settings_data['horde_Steps'] = steps
        self.g.settings_data['horde_CFG Scale'] = cfg
        self.g.settings_data['horde_Width'] = width
        self.g.settings_data['horde_Height'] = heigth
        self.g.settings_data['horde_Clipskip'] = clipskip
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()
    
    
    async def set_automa_settings(self, sampler, steps, cfg, width, heigth, url, save):
        self.g.settings_data['automa_Sampler'] = sampler
        self.g.settings_data['automa_Steps'] = steps
        self.g.settings_data['automa_CFG Scale'] = cfg
        self.g.settings_data['automa_Width'] = width
        self.g.settings_data['automa_Height'] = heigth
        self.g.settings_data['automa_url'] = url
        self.g.settings_data['automa_save'] = save
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()
    
    
    async def set_model(self, model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
        await self.set_llm_settings(model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct)
        model = self.g.settings_data['model_list'][model]
        return await self.interface.change_model(model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct)
    
    
    def all_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['civitai_Air'], self.g.settings_data[
            'civitai_Steps'], self.g.settings_data['civitai_CFG Scale'], self.g.settings_data['civitai_Width'], self.g.settings_data[
            'civitai_Height'], self.g.settings_data['civitai_Clipskip'], self.g.last_prompt, self.g.last_negative_prompt, \
            self.g.settings_data['horde_api_key'], self.g.settings_data['horde_Model'], self.g.settings_data['horde_Sampler'], self.g.settings_data[
            'horde_Steps'], self.g.settings_data['horde_CFG Scale'], self.g.settings_data['horde_Width'], self.g.settings_data['horde_Height'], \
            self.g.settings_data['horde_Clipskip'], self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data[
            'automa_Sampler'], self.g.settings_data['automa_Steps'], self.g.settings_data['automa_CFG Scale'], self.g.settings_data[
            'automa_Width'], self.g.settings_data['automa_Height'], self.g.settings_data['automa_url'], self.g.settings_data['automa_save']
    
    
    def civitai_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['civitai_Air'], self.g.settings_data[
            'civitai_Steps'], self.g.settings_data['civitai_CFG Scale'], self.g.settings_data['civitai_Width'], self.g.settings_data[
            'civitai_Height'], self.g.settings_data['civitai_Clipskip']
    
    
    def hordeai_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['horde_api_key'], self.g.settings_data[
            'horde_Model'], self.g.settings_data['horde_Sampler'], self.g.settings_data['horde_Steps'], self.g.settings_data['horde_CFG Scale'], \
            self.g.settings_data['horde_Width'], self.g.settings_data['horde_Height'], self.g.settings_data['horde_Clipskip']


    async def automa_get_last_prompt(self):
        client = automa_client()
        samplers = await client.get_samplers(self.g.settings_data['automa_url'])
        if samplers != -1:
            self.g.settings_data['automa_Sampler'] = samplers

        checkpoints = await client.get_checkpoints(self.g.settings_data['automa_url'])
        if checkpoints != -1:
            self.g.settings_data['automa_checkpoints'] = checkpoints


        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['automa_Sampler'], self.g.settings_data['automa_Steps'], self.g.settings_data['automa_CFG Scale'], self.g.settings_data['automa_Width'], self.g.settings_data['automa_Height'], self.g.settings_data['automa_url'], self.g.settings_data['automa_save']
    
    
    def llm_get_settings(self):
        return self.g.settings_data["LLM Model"], self.g.settings_data['Temperature'], self.g.settings_data['Context Length'], self.g.settings_data['GPU Layers'], self.g.settings_data['max output Tokens'], self.g.settings_data['top_k'], self.g.settings_data['Instruct Model']
    
    
    def get_prompt_template(self):
        self.interface.prompt_template = self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]
        return self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]
    
    
    async def set_prompt_template_select(self, value):
        self.g.settings_data['selected_template'] = value
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()
        return self.g.settings_data["prompt_templates"][value]
    
    async def set_neg_prompt(self, value):
        self.g.settings_data['negative_prompt'] = value
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()
    
    async def set_prompt_template(self, selection, prompt_text):
        return_data = await self.interface.set_prompt(prompt_text)
        self.g.settings_data["prompt_templates"][selection] = prompt_text
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()
        return return_data
    
    
    async def run_civitai_generation(self, air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip):
        await self.set_civitai_settings(air, steps, cfg, width, heigth, clipskip)
        client = civitai_client()
        return await client.request_generation(air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip)
    
    
    async def run_hordeai_generation(self, prompt, negative_prompt, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
        await self.set_hordeai_settings(api_key, model, sampler, steps, cfg, width, heigth, clipskip)
        client = hordeai_client()
        return await client.request_generation(api_key=api_key, prompt=prompt, negative_prompt=negative_prompt,
                                         sampler=sampler, model=model, steps=steps, cfg=cfg, width=width, heigth=heigth,
                                         clipskip=clipskip)
    
    
    async def run_automatics_generation(self, prompt, negative_prompt, sampler, steps, cfg, width, heigth, url, save):
        await self.set_automa_settings(sampler, steps, cfg, width, heigth, url, save)
        client = automa_client()
        return await client.request_generation(prompt=prompt, negative_prompt=negative_prompt,
                                         sampler=sampler, steps=steps, cfg=cfg, width=width, heigth=heigth, url=url,
                                         save=save)
    
    async def run_automa_interrogation(self, image_filename,url):
        with open(image_filename, mode='rb') as fp:
            base64_image = base64.b64encode(fp.read()).decode('utf-8')
        client = automa_client()
        response = await client.request_interrogation(base64_image,url)
        self.g.context_prompt = response
        return response
    
    async def run_automa_interrogation_batch(self, image_filenames,url, save):
    
        all_response = ''
    
        for file in image_filenames:
            response = await self.run_automa_interrogation(file[0],url)
            if all_response == '':
                all_response = response
            else:
                all_response = f'{all_response}\n{response}'
    
        if save:
            import time
            filename = f'{time.strftime("%Y%m%d-%H%M%S")}.txt'
            outfile = os.path.join(out_dir_i2t,filename)
            f = open(outfile,'a',encoding='utf8',errors='ignore')
            f.write(f'{all_response}\n')
            f.close()
    
        return all_response

    async def run_t2t_sail(self, sail_text,sail_width,sail_depth,sail_target,sail_generate,sail_sinus,sail_sinus_range,sail_sinus_freq,sail_add_style,sail_style,sail_add_search,sail_search):
        self.g.sailing_run = True
        return await self.interface.run_t2t_sail(sail_text,sail_width,sail_depth,sail_target, sail_generate,sail_sinus,sail_sinus_range,sail_sinus_freq,sail_add_style,sail_style,sail_add_search,sail_search)

    async def stop_sailing(self):
        self.g.sailing_run = False

    async def variable_outputs(self, k):
        self.g.settings_data['top_k'] = int(k)
        self.interface.set_top_k(self.g.settings_data['top_k'])
        k = int(k)
        out = [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (max_top_k - k)
        return out
    

    async def get_context_details(self, *args):
        context_details = await self.interface.get_context_details()
        textboxes = []
        for detail in context_details:
            t = gr.Textbox(f"{detail}")
            textboxes.append(t)
        if len(textboxes) < len(args):
            x = range(len(textboxes), len(args))
            for n in x:
                textboxes.append('')
        return textboxes
    
    
    async def dive_into(self, text):
        self.g.context_prompt = text
        context = await self.interface.retrieve_context(text)
    
        if len(context) < max_top_k - 1:
            x = range(len(context), max_top_k - 1)
            for n in x:
                context.append('')
    
        return context  # .append(text)

    async def set_prompt_input(self):
        return self.g.context_prompt


    async def set_translate(self, translate):
        self.g.settings_data['translate'] = translate
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()
    
    
    async def set_batch(self, batch):
        self.g.settings_data['batch'] = batch
        await self.settings_io.write_settings(self.g.settings_data)
        await self.interface.reload_settings()


    async def run_batch(self, files):
        for file in files:
            filename = os.path.basename(file)
            file_content = []
            f = open(file,'r',encoding='utf8',errors='ignore')
            file_content = f.readlines()
            f.close()
    
            outfile = os.path.join(out_dir_t2t,filename)
            f = open(outfile,'a',encoding='utf8',errors='ignore')
            for query in file_content:
                response= await self.interface.run_llm_response_batch(query)
                f.write(f'{response}\n')
            f.close()
        return 'done'


class ui_staff:

    def __init__(self):

        self.g = globals.get_globals()
        self.hordeai_model_list = hordeai_models().read_model_list()
        self.g.last_prompt = ''
        self.g.last_negative_prompt = ''
        self.g.last_context = []

        self.civitai_prompt_input = gr.TextArea(self.g.last_prompt, lines=10, label="Prompt")
        self.civitai_negative_prompt_input = gr.TextArea(self.g.last_negative_prompt, lines=5, label="Negative Prompt")
        self.hordeai_prompt_input = gr.TextArea(self.g.last_prompt, lines=10, label="Prompt")
        self.hordeai_negative_prompt_input = gr.TextArea(self.g.last_negative_prompt, lines=5, label="Negative Prompt")
        self.automa_prompt_input = gr.TextArea(self.g.last_prompt, lines=10, label="Prompt")
        self.automa_negative_prompt_input = gr.TextArea(self.g.last_negative_prompt, lines=5, label="Negative Prompt")

        self.LLM = gr.Dropdown(
            self.g.settings_data['model_list'].keys(), value=self.g.settings_data['LLM Model'], label="LLM Model",
            info="Will add more LLMs later!"
        )
        self.Temperature = gr.Slider(0, 1, step=0.1, value=self.g.settings_data['Temperature'], label="Temperature",
                                info="Choose between 0 and 1")
        self.Context = gr.Slider(0, 8192, step=1, value=self.g.settings_data['Context Length'], label="Context Length",
                            info="Choose between 1 and 8192")
        self.GPU = gr.Slider(0, 1024, step=1, value=self.g.settings_data['GPU Layers'], label="GPU Layers",
                        info="Choose between 1 and 1024")
        self.max = gr.Slider(0, 1024, step=1, value=self.g.settings_data['max output Tokens'], label="max output Tokens",
                        info="Choose between 1 and 1024")
        self.top_k = gr.Slider(0, max_top_k, step=1, value=self.g.settings_data['top_k'],
                          label="how many entrys to be fetched from the vector store",
                          info="Choose between 1 and 50 be careful not to overload the context window of the LLM")
        self.Instruct = gr.Checkbox(label='Instruct Model', value=self.g.settings_data['Instruct Model'])

        self.civitai_Air = gr.TextArea(self.g.settings_data['civitai_Air'], lines=1, label="Air")
        self.civitai_Steps = gr.Slider(0, 100, step=1, value=self.g.settings_data['civitai_Steps'], label="Steps",
                                  info="Choose between 1 and 100")
        self.civitai_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['civitai_CFG Scale'], label="CFG Scale",
                                info="Choose between 1 and 20")
        self.civitai_Width = gr.Slider(0, 2048, step=1, value=self.g.settings_data['civitai_Width'], label="Width",
                                  info="Choose between 1 and 2048")
        self.civitai_Height = gr.Slider(0, 2048, step=1, value=self.g.settings_data['civitai_Height'], label="Height",
                                   info="Choose between 1 and 2048")
        self.civitai_Clipskip = gr.Slider(0, 10, step=1, value=self.g.settings_data['civitai_Clipskip'], label="Clipskip",
                                     info="Choose between 1 and 10")

        self.horde_api_key = gr.TextArea(lines=1, label="API Key", value=self.g.settings_data['horde_api_key'], type='password')
        self.horde_Model = gr.Dropdown(choices=self.hordeai_model_list.keys(), value=self.g.settings_data['horde_Model'], label='Model')
        self.horde_Sampler = gr.Dropdown(choices=["k_dpmpp_2s_a", "k_lms", "k_heun", "k_heun", "k_euler", "k_euler_a",
                                             "k_dpm_2", "k_dpm_2_a", "k_dpm_fast", "k_dpm_adaptive", "k_dpmpp_2s_a",
                                             "k_dpmpp_2m", "dpmsolver", "k_dpmpp_sde", "lcm", "DDIM"
                                             ], value=self.g.settings_data['horde_Sampler'], label='Sampler')
        self.horde_Steps = gr.Slider(0, 100, step=1, value=self.g.settings_data['horde_Steps'], label="Steps",
                                info="Choose between 1 and 100")
        self.horde_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['horde_CFG Scale'], label="CFG Scale",
                              info="Choose between 1 and 20")
        self.horde_Width = gr.Slider(0, 2048, step=1, value=self.g.settings_data['horde_Width'], label="Width",
                                info="Choose between 1 and 2048")
        self.horde_Height = gr.Slider(0, 2048, step=1, value=self.g.settings_data['horde_Height'], label="Height",
                                 info="Choose between 1 and 2048")
        self.horde_Clipskip = gr.Slider(0, 10, step=1, value=self.g.settings_data['horde_Clipskip'], label="Clipskip",
                                   info="Choose between 1 and 10")

        self.automa_url = gr.TextArea(lines=1, label="API URL", value=self.g.settings_data['automa_url'])
        self.automa_Sampler = gr.Dropdown(
            choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a',
                     'Euler',
                     'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a',
                     'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras',
                     'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential',
                     'DPM fast',
                     'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras'
                     ], value=self.g.settings_data['automa_Sampler'], label='Sampler')
        self.automa_Steps = gr.Slider(0, 100, step=1, value=self.g.settings_data['automa_Steps'], label="Steps",
                                 info="Choose between 1 and 100")
        self.automa_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['automa_CFG Scale'], label="CFG Scale",
                               info="Choose between 1 and 20")
        self.automa_Width = gr.Slider(0, 2048, step=1, value=self.g.settings_data['automa_Width'], label="Width",
                                 info="Choose between 1 and 2048")
        self.automa_Height = gr.Slider(0, 2048, step=1, value=self.g.settings_data['automa_Height'], label="Height",
                                  info="Choose between 1 and 2048")
        self.automa_save = gr.Checkbox(label="Save", info="Save the image?", value=self.g.settings_data['automa_save'])

        self.prompt_template = gr.TextArea(self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]], lines=20)
        self.prompt_template_select = gr.Dropdown(choices=self.g.settings_data["prompt_templates"].keys(),
                                             value=self.g.settings_data["selected_template"], label='Template', interactive=True)


        self.prompt_input = gr.Textbox(placeholder="Make your prompts more creative", container=False, scale=7, render=False)




