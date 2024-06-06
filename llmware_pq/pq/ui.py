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

import globals
import os
import gradio as gr
import base64
from datetime import datetime
import math
from PIL import Image
from io import BytesIO
import time
from collections import deque

from settings.io import settings_io

from llm_fw.llm_interface_qdrant import LLM_INTERFACE

from generators.civitai.client import civitai_client
from generators.hordeai.client import hordeai_client
from generators.automatics.client import automa_client
from generators.hordeai.client import hordeai_models



out_dir = 'api_out'
out_dir_t2t = os.path.join(out_dir, 'txt2txt')
os.makedirs(out_dir_t2t, exist_ok=True)
out_dir_i2t = os.path.join(out_dir, 'img2txt')
os.makedirs(out_dir_i2t, exist_ok=True)
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
max_top_k = 50


class ui_actions:


    def __init__(self):
        self.g = globals.get_globals()
        self.interface = LLM_INTERFACE()
        self.settings_io = settings_io()
        self.max_top_k = 50
        self.automa_client = automa_client()
        self.set_sailing_settings('',1,1,False,'',False,0.1,1,False,'',False,'')


    def run_llm_response(self, query, history):
        return self.interface.run_llm_response( query, history)


    def set_llm_settings(self, model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
        self.g.settings_data['LLM Model'] = model
        self.g.settings_data['Temperature'] = temperature
        self.g.settings_data['Context Length'] = n_ctx
        self.g.settings_data['GPU Layers'] = n_gpu_layers
        self.g.settings_data['max output Tokens'] = max_tokens
        self.g.settings_data['top_k'] = top_k
        self.g.settings_data['Instruct Model'] = instruct
        self.settings_io.write_settings(self.g.settings_data)
        self.interface.reload_settings()


    def set_civitai_settings(self, air, steps, cfg, width, heigth, clipskip):
        self.g.settings_data['civitai_Air'] = air
        self.g.settings_data['civitai_Steps'] = steps
        self.g.settings_data['civitai_CFG Scale'] = cfg
        self.g.settings_data['civitai_Width'] = width
        self.g.settings_data['civitai_Height'] = heigth
        self.g.settings_data['civitai_Clipskip'] = clipskip
        self.settings_io.write_settings(self.g.settings_data)
        self.interface.reload_settings()


    def set_hordeai_settings(self, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
        self.g.settings_data['horde_api_key'] = api_key
        self.g.settings_data['horde_model'] = model
        self.g.settings_data['horde_sampler'] = sampler
        self.g.settings_data['horde_steps'] = steps
        self.g.settings_data['horde_cfg_scale'] = cfg
        self.g.settings_data['horde_width'] = width
        self.g.settings_data['horde_height'] = heigth
        self.g.settings_data['horde_clipskip'] = clipskip
        self.settings_io.write_settings(self.g.settings_data)
        self.interface.reload_settings()

    def set_automa_settings(self, sampler, steps, cfg, width, heigth, batch,n_iter, url, save, save_api):
        self.g.settings_data['automa_sampler'] = sampler
        self.g.settings_data['automa_steps'] = steps
        self.g.settings_data['automa_cfg_scale'] = cfg
        self.g.settings_data['automa_width'] = width
        self.g.settings_data['automa_height'] = heigth
        self.g.settings_data['automa_batch'] = batch
        self.g.settings_data['automa_n_iter'] = n_iter
        self.g.settings_data['automa_url'] = url
        self.g.settings_data['automa_save'] = save
        self.g.settings_data['automa_save_on_api_host'] = save_api
        self.settings_io.write_settings(self.g.settings_data)
        self.interface.reload_settings()

    def set_sailing_settings(self,sail_text, sail_width, sail_depth, sail_generate, sail_target, sail_sinus,
                             sail_sinus_freq, sail_sinus_range, sail_add_style, sail_style, sail_add_search,
                             sail_search):
        self.g.settings_data['sail_text'] = sail_text
        self.g.settings_data['sail_width'] = sail_width
        self.g.settings_data['sail_depth'] = sail_depth
        self.g.settings_data['sail_generate'] = sail_generate
        self.g.settings_data['sail_target'] = sail_target
        self.g.settings_data['sail_sinus'] = sail_sinus
        self.g.settings_data['sail_sinus_freq'] = sail_sinus_freq
        self.g.settings_data['sail_sinus_range'] = sail_sinus_range
        self.g.settings_data['sail_add_style'] = sail_add_style
        self.g.settings_data['sail_style'] = sail_style
        self.g.settings_data['sail_add_search'] = sail_add_search
        self.g.settings_data['sail_search'] = sail_search


    def set_neg_prompt(self, value):
        self.g.settings_data['negative_prompt'] = value
        self.settings_io.write_settings(self.g.settings_data)
        self.interface.reload_settings()


    def set_model(self, model, temperature, n_ctx, max_tokens, gpu_layers, top_k, instruct):
        self.set_llm_settings(model, temperature, n_ctx, max_tokens, gpu_layers, top_k, instruct)
        return self.interface.change_model(model, temperature, n_ctx, max_tokens, gpu_layers, top_k, instruct)

    def all_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['civitai_Air'], self.g.settings_data[
            'civitai_Steps'], self.g.settings_data['civitai_CFG Scale'], self.g.settings_data['civitai_Width'], self.g.settings_data[
            'civitai_Height'], self.g.settings_data['civitai_Clipskip'], self.g.last_prompt, self.g.last_negative_prompt, \
            self.g.settings_data['horde_api_key'], self.g.settings_data['horde_model'], self.g.settings_data['horde_sampler'], self.g.settings_data[
            'horde_steps'], self.g.settings_data['horde_cfg_scale'], self.g.settings_data['horde_width'], self.g.settings_data['horde_height'], \
            self.g.settings_data['horde_clipskip'], self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data[
            'automa_sampler'], self.g.settings_data['automa_steps'], self.g.settings_data['automa_cfg_scale'], self.g.settings_data[
            'automa_width'], self.g.settings_data['automa_height'], self.g.settings_data['automa_batch'],self.g.settings_data['automa_n_iter'], self.g.settings_data['automa_url'], self.g.settings_data['automa_save'], self.g.settings_data['automa_save_on_api_host']


    def civitai_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['civitai_Air'], self.g.settings_data[
            'civitai_Steps'], self.g.settings_data['civitai_CFG Scale'], self.g.settings_data['civitai_Width'], self.g.settings_data[
            'civitai_Height'], self.g.settings_data['civitai_Clipskip']


    def hordeai_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['horde_api_key'], self.g.settings_data[
            'horde_model'], self.g.settings_data['horde_sampler'], self.g.settings_data['horde_steps'], self.g.settings_data['horde_cfg_scale'], \
            self.g.settings_data['horde_width'], self.g.settings_data['horde_height'], self.g.settings_data['horde_clipskip']


    def automa_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['automa_sampler'], self.g.settings_data['automa_steps'], self.g.settings_data['automa_cfg_scale'], self.g.settings_data['automa_width'], self.g.settings_data['automa_height'], self.g.settings_data['automa_batch'],self.g.settings_data['automa_n_iter'], self.g.settings_data['automa_url'], self.g.settings_data['automa_save'], self.g.settings_data['automa_save_on_api_host']


    def llm_get_settings(self):
        return self.g.settings_data["LLM Model"], self.g.settings_data['Temperature'], self.g.settings_data['Context Length'], self.g.settings_data['GPU Layers'], self.g.settings_data['max output Tokens'], self.g.settings_data['top_k'], self.g.settings_data['Instruct Model']


    def run_civitai_generation(self, air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip):
        self.set_civitai_settings(air, steps, cfg, width, heigth, clipskip)
        client = civitai_client()
        return client.request_generation(air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip)


    def run_hordeai_generation(self, api_key, prompt, negative_prompt, model, sampler, steps, cfg, width, heigth, clipskip):
        self.set_hordeai_settings(api_key, model, sampler, steps, cfg, width, heigth, clipskip)
        client = hordeai_client()
        return client.request_generation(api_key=api_key, prompt=prompt, negative_prompt=negative_prompt,
                                         sampler=sampler, model=model, steps=steps, cfg=cfg, width=width, heigth=heigth,
                                         clipskip=clipskip)

    def timestamp(self):
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    def run_automatics_generation(self, prompt, negative_prompt, sampler, steps, cfg, width, heigth, batch,n_iter, url, save,save_api):
        self.g.running = True
        self.set_automa_settings(sampler, steps, cfg, width, heigth, batch,n_iter, url, save, save_api)
        response = self.automa_client.request_generation(prompt=prompt, negative_prompt=negative_prompt,
                                                         sampler=sampler, steps=steps, cfg=cfg, width=width, heigth=heigth, url=url,
                                                         save=save, batch=batch,n_iter=n_iter, save_api=save_api)
        images = []
        for index, image in enumerate(response.get('images')):
            img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
            if save:
                save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                self.automa_client.decode_and_save_base64(image, save_path)
            images.append(img)
        yield images

    def run_automa_interrogation(self, image_filename,url):
        with open(image_filename, mode='rb') as fp:
            base64_image = base64.b64encode(fp.read()).decode('utf-8')
        client = automa_client()
        response = client.request_interrogation(base64_image,url)
        self.g.context_prompt = response
        return response


    def run_automa_interrogation_batch(self, image_filenames,url, save):

        all_response = ''

        for file in image_filenames:
            response = self.run_automa_interrogation(file[0],url)
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



    def get_next_target(self, nodes, sail_target,sail_sinus,sail_sinus_range,sail_sinus_freq):

        target_dict = self.interface.get_query_texts(nodes)

        if len(target_dict.keys()) < self.sail_depth:
            self.sail_depth = self.sail_depth_start + len(self.g.sail_history)

        if sail_sinus:
            sinus = int(math.sin(self.sail_sinus_count/10.0)*sail_sinus_range)
            self.sail_sinus_count += sail_sinus_freq
            self.sail_depth += sinus
            if self.sail_depth < 0:
                self.sail_depth = 1

        if len(target_dict.keys()) > 0:

            if sail_target:
                out =  target_dict[min(target_dict.keys())]
                self.g.sail_history.append(out)
                return out
            else:
                out =  target_dict[max(target_dict.keys())]
                self.g.sail_history.append(out)
                return out
        else:
            return -1

    def check_api_avail(self):
        return self.automa_client.check_avail(self.g.settings_data['automa_url'])

    def sail_automa_gen(self, query):
        return self.automa_client.request_generation(query,
                                                     self.g.settings_data['negative_prompt'],
                                                     self.g.settings_data['automa_sampler'],
                                                     self.g.settings_data['automa_steps'],
                                                     self.g.settings_data['automa_cfg_scale'],
                                                     self.g.settings_data['automa_width'],
                                                     self.g.settings_data['automa_height'],
                                                     self.g.settings_data['automa_url'],
                                                     self.g.settings_data['automa_save'],
                                                     self.g.settings_data['automa_batch'],
                                                     self.g.settings_data['automa_n_iter'],
                                                     self.g.settings_data['automa_save_on_api_host'])

    def run_t2t_sail(self, query,sail_width,sail_depth,sail_target,sail_generate,sail_sinus,sail_sinus_range,sail_sinus_freq,sail_add_style,sail_style,sail_add_search,sail_search,sail_max_gallery_size):
        self.g.job_running = True


        self.g.sail_history = []
        self.sail_depth = sail_depth
        self.sail_depth_start = sail_depth
        self.sail_sinus_count = 1.0
        filename = os.path.join(out_dir_t2t, f'Journey_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')
        sail_log = ''

        if self.g.settings_data['translate']:
            query = self.interface.translate(query)


        images = deque(maxlen=int(sail_max_gallery_size))

        for n in range(sail_width):

            if sail_add_search:
                query = f'{sail_search}, {query}'
            prompt = self.interface.retrieve_query(query)

            if sail_add_style:
                prompt = f'{sail_style}, {prompt}'

            self.interface.log_raw(filename,f'{prompt}')
            self.interface.log_raw(filename,f'{n} ----------')
            sail_log = sail_log + f'{prompt}\n'
            sail_log = sail_log + f'{n} ----------\n'
            nodes = self.interface.retrieve_top_k_query(query, self.sail_depth)
            if sail_generate:
                print('start generate')

                response = self.sail_automa_gen(prompt)

                for index, image in enumerate(response.get('images')):
                    img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                    save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                    self.automa_client.decode_and_save_base64(image, save_path)
                    images.append(img)
                print(f'generated {len(list(images))} images')
                yield prompt,list(images)
            else:
                yield prompt,[]
            query = self.get_next_target(nodes,sail_target,sail_sinus,sail_sinus_range,sail_sinus_freq)
            if query == -1:
                self.interface.log_raw(filename,f'{n} sail is finished early due to rotating context')
                break
            if self.g.job_running is False:
                break

    def run_t2t_show_sail(self):
        self.g.job_running = True
        self.g.settings_data['automa_batch'] = 1
        self.g.settings_data['automa_n_iter'] = 1

        self.g.sail_history = []
        self.sail_depth = self.g.settings_data['sail_depth']
        self.sail_depth_start = self.g.settings_data['sail_depth']
        self.sail_sinus_count = 1.0
        filename = os.path.join(out_dir_t2t, f'Journey_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')
        sail_log = ''
        query = self.g.settings_data['sail_text']
        if self.g.settings_data['translate']:
            query = self.interface.translate(query)

        for n in range(self.g.settings_data['sail_width']):

            if self.g.settings_data['sail_add_search']:
                query = f"{self.g.settings_data['sail_search']}, {query}"
            prompt = self.interface.retrieve_query(query)
            if self.g.settings_data['sail_add_style']:
                prompt = f'{self.g.settings_data["sail_style"]}, {prompt}'

            self.interface.log_raw(filename,f'{prompt}')
            self.interface.log_raw(filename,f'{n} ----------')
            sail_log = sail_log + f'{prompt}\n'
            sail_log = sail_log + f'{n} ----------\n'
            nodes = self.interface.retrieve_top_k_query(query, self.sail_depth)
            if self.g.settings_data['sail_generate']:
                response = self.sail_automa_gen(prompt)

                for index, image in enumerate(response.get('images')):
                    img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                    save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                    self.automa_client.decode_and_save_base64(image, save_path)
                    yield prompt,img
            else:
                yield prompt,None
            query = self.get_next_target(nodes,self.g.settings_data['sail_target'],self.g.settings_data['sail_sinus'],self.g.settings_data['sail_sinus_range'],self.g.settings_data['sail_sinus_freq'])
            if query == -1:
                self.interface.log_raw(filename,f'{n} sail is finished early due to rotating context')
                break
            if self.g.job_running is False:
                break


    def stop_t2t_sail(self):
        self.g.job_running = False

    def get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt


    def llm_get_settings(self):
        return self.g.settings_data["LLM Model"], \
            self.g.settings_data['Temperature'], \
            self.g.settings_data['Context Length'], \
            self.g.settings_data['GPU Layers'], \
            self.g.settings_data['max output Tokens'], \
            self.g.settings_data['top_k'], \
            self.g.settings_data['Instruct Model']


    def get_prompt_template(self):
        self.interface.prompt_template = self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]
        return self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]["blurb1"]


    def set_prompt_template_select(self, value):
        self.g.settings_data['selected_template'] = value
        self.settings_io.write_settings(self.g.settings_data)
        return self.g.settings_data["prompt_templates"][value]


    def set_prompt_template(self, selection, prompt_text):
        return_data = self.interface.set_prompt(prompt_text)
        self.g.settings_data["prompt_templates"][selection]["blurb1"] = prompt_text
        self.settings_io.write_settings(self.g.settings_data)
        return return_data



    def variable_outputs(self, k):
        self.g.settings_data['top_k'] = int(k)
        self.interface.set_top_k(self.g.settings_data['top_k'])
        k = int(k)
        out = [gr.Textbox(visible=True)]*k + [gr.Textbox(visible=False)]*(self.max_top_k-k)
        return out


    def get_context_details(self, *args):
        context_details = self.interface.get_context_details()
        textboxes = []
        for detail in context_details:
            t = gr.Textbox(f"{detail}")
            textboxes.append(t)
        if len(textboxes) < len(args):
            x = range(len(textboxes), len(args))
            for n in x:
                textboxes.append('')
        return textboxes

    def dive_into(self, text):
        self.g.context_prompt = text
        context = self.interface.retrieve_context(text)

        if len(context) < self.max_top_k-1:
            x = range(len(context), self.max_top_k-1)
            for n in x:
                context.append('')

        return context  #.append(text)

    def set_prompt_input(self):
        return self.g.context_prompt

    def set_translate(self, translate):
        self.g.settings_data['translate'] = translate
        self.settings_io.write_settings(self.g.settings_data)
        self.interface.reload_settings()

    def set_batch(self, batch):
        self.g.settings_data['batch'] = batch
        self.settings_io.write_settings(self.g.settings_data)
        self.interface.reload_settings()

    def run_batch(self, files):
        output = ''
        for file in files:
            filename = os.path.basename(file)
            file_content = []
            f = open(file,'r',encoding='utf8',errors='ignore')
            file_content = f.readlines()
            f.close()

            outfile = os.path.join(out_dir_t2t,filename)
            f = open(outfile,'a',encoding='utf8',errors='ignore')
            n = 0
            for query in file_content:
                response= self.interface.run_llm_response_batch(query)
                f.write(f'{response}\n')
                output = f'{output}{response}\n{n} ---------\n'
                n += 1
                yield output
            f.close()


class ui_staff:


    def __init__(self):

        self.g = globals.get_globals()
        self.hordeai_model_list = hordeai_models().read_model_list()
        self.g.context_prompt = ''
        self.g.last_prompt = ''
        self.g.last_negative_prompt = ''
        self.g.last_context = []

        self.prompt_input = gr.Textbox(placeholder="Make your prompts more creative", container=False, scale=7, render=False)


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
        self.n_ctx = gr.Slider(0, 32768, step=1, value=self.g.settings_data['Context Length'], label="Context Length",
                          info="Choose between 0 and 1")
        self.max = gr.Slider(0, 1024, step=1, value=self.g.settings_data['max output Tokens'], label="max output Tokens",
                        info="Choose between 1 and 1024")
        self.GPU = gr.Slider(0, 1024, step=1, value=self.g.settings_data['GPU Layers'], label="GPU Layers",
                        info="Choose between 1 and 1024")
        self.top_k = gr.Slider(0, 50, step=1, value=self.g.settings_data['top_k'],
                          label="how many entrys to be fetched from the vector store",
                          info="Choose between 1 and 50 be careful not to overload the context window of the LLM")
        self.Instruct = gr.Checkbox(label='Instruct Model', value=self.g.settings_data['Instruct Model'])

        self.civitai_Air = gr.TextArea(lines=1, label="Air", value=self.g.settings_data['civitai_Air'])
        self.civitai_Steps = gr.Slider(0, 100, step=1, value=self.g.settings_data['civitai_Steps'], label="Steps",
                                  info="Choose between 1 and 100")
        self.civitai_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['civitai_CFG Scale'], label="CFG Scale",
                                info="Choose between 1 and 20")
        self.civitai_Width = gr.Slider(0, 1024, step=1, value=self.g.settings_data['civitai_Width'], label="Width",
                                  info="Choose between 1 and 1024")
        self.civitai_Height = gr.Slider(0, 1024, step=1, value=self.g.settings_data['civitai_Height'], label="Height",
                                   info="Choose between 1 and 1024")
        self.civitai_Clipskip = gr.Slider(0, 10, step=1, value=self.g.settings_data['civitai_Clipskip'], label="Clipskip",
                                     info="Choose between 1 and 10")

        self.horde_api_key = gr.TextArea(lines=1, label="API Key", value=self.g.settings_data['horde_api_key'], type='password')
        self.horde_model = gr.Dropdown(choices=self.hordeai_model_list.keys(), value='Deliberate 3.0', label='Model')
        self.horde_sampler = gr.Dropdown(choices=["k_dpmpp_2s_a", "k_lms", "k_heun", "k_heun", "k_euler", "k_euler_a",
                                             "k_dpm_2", "k_dpm_2_a", "k_dpm_fast", "k_dpm_adaptive", "k_dpmpp_2s_a",
                                             "k_dpmpp_2m", "dpmsolver", "k_dpmpp_sde", "lcm", "DDIM"
                                             ], value=self.g.settings_data['horde_sampler'], label='Sampler')

        self.horde_steps = gr.Slider(0, 100, step=1, value=self.g.settings_data['horde_steps'], label="Steps",
                                info="Choose between 1 and 100")
        self.horde_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['horde_cfg_scale'], label="CFG Scale",
                              info="Choose between 1 and 20")
        self.horde_width = gr.Slider(0, 1024, step=1, value=self.g.settings_data['horde_width'], label="Width",
                                info="Choose between 1 and 1024")
        self.horde_height = gr.Slider(0, 1024, step=1, value=self.g.settings_data['horde_height'], label="Height",
                                 info="Choose between 1 and 1024")
        self.horde_clipskip = gr.Slider(0, 10, step=1, value=self.g.settings_data['horde_clipskip'], label="Clipskip",
                                   info="Choose between 1 and 10")

        self.automa_url = gr.TextArea(lines=1, label="API URL", value=self.g.settings_data['automa_url'])
        self.automa_sampler = gr.Dropdown(
            choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a',
                     'Euler',
                     'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a',
                     'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras',
                     'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential',
                     'DPM fast',
                     'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras'
                     ], value=self.g.settings_data['automa_sampler'], label='Sampler')
        self.automa_steps = gr.Slider(0, 100, step=1, value=self.g.settings_data['automa_steps'], label="Steps",
                                 info="Choose between 1 and 100")
        self.automa_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['automa_cfg_scale'], label="CFG Scale",
                               info="Choose between 1 and 20")
        self.automa_width = gr.Slider(0, 2048, step=1, value=self.g.settings_data['automa_width'], label="Width",
                                 info="Choose between 1 and 2048")
        self.automa_height = gr.Slider(0, 2048, step=1, value=self.g.settings_data['automa_height'], label="Height",
                                  info="Choose between 1 and 2048")
        self.automa_Batch = gr.Slider(1, 250, step=1, value=self.g.settings_data['automa_batch'], label="Batch",
                                      info="The number of simultaneous images in each batch, range from 1-50.")
        self.automa_n_iter = gr.Slider(1, 500, step=1, value=self.g.settings_data['automa_n_iter'], label="Iterations",
                                       info="The number of sequential batches to be run, range from 1-500.")
        self.automa_save = gr.Checkbox(label="Save", info="Save the image?", value=self.g.settings_data['automa_save'])
        self.automa_save_on_api_host = gr.Checkbox(label="Save", info="Save the image on API host?", value=self.g.settings_data['automa_save_on_api_host'])

        self.automa_stop_button = gr.Button('Stop')


        self.prompt_template = gr.TextArea(self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]["blurb1"], lines=20)
        self.prompt_template_select = gr.Dropdown(choices=self.g.settings_data["prompt_templates"].keys(),
                                             value=self.g.settings_data["selected_template"], label='Template', interactive=True)

        self.sail_result_last_image = gr.Image(label='last Image')
