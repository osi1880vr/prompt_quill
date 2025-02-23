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
import json
import random
from collections import deque
from api import v1
import shared
from ui.ui_share import UiShare

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


from generators.automatics.client import automa_client
from generators.hordeai.client import hordeai_models
from settings.io import settings_io
from prompt_iteration import prompt_iterator
from llm_fw import llm_interface_qdrant


out_dir = '../api_out'
out_dir_t2t = os.path.join(out_dir, 'txt2txt')
os.makedirs(out_dir_t2t, exist_ok=True)
out_dir_i2t = os.path.join(out_dir, 'img2txt')
os.makedirs(out_dir_i2t, exist_ok=True)
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)



class ui_actions:
    def __init__(self):
        self.g = globals.get_globals()
        self.ui_share = UiShare()
        self.g.job_running = False
        self.interface = llm_interface_qdrant.get_interface()
        self.settings_io = settings_io()
        self.max_top_k = 50
        self.automa_client = automa_client()
        self.api = v1
        self.api.run_api()
        self.prompt_iterator = prompt_iterator()
        self.gen_step = 0
        self.gen_step_select = 0


        self.sail_log = ''
        self.sail_sinus_count = 1.0
        self.sinus = 0
        self.sail_depth_start = 0
        self.images_done = 1
        self.ui_share = UiShare()
        self.prompt_array_index = {}




    def timestamp(self):
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


    def sail_automa_gen(self, query):

        negative_prompt = self.g.settings_data['negative_prompt']

        if self.g.settings_data['sailing']['sail_dyn_neg']:
            if len(self.g.negative_prompt_list) > 0:
                negative_prompt = shared.get_negative_prompt()


        if self.g.settings_data['sailing']['sail_add_neg']:
            negative_prompt = f"{self.g.settings_data['sailing']['sail_neg_embed']},{self.g.settings_data['sailing']['sail_neg_prompt']}, {negative_prompt}"

        if len(negative_prompt) < 30:
            negative_prompt = self.g.settings_data['negative_prompt']

        self.g.act_neg_prompt = negative_prompt

        return self.automa_client.request_generation(query,
                                                     negative_prompt,
                                                     self.g.settings_data)

    def log_prompt(self, filename, prompt, orig_prompt, n, sail_log):

        if self.g.settings_data['sailing']['sail_generate']:

            if self.g.settings_data['sailing']['sail_sinus']:
                self.interface.log_raw(filename,f'{prompt} \nsinus {self.sinus} {n} ----------')
                if self.g.settings_data['sailing']['sail_rephrase']:
                    self.interface.log_raw(filename,f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------')
                    sail_log = sail_log + f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------\n'

                sail_log = sail_log + f'{prompt} \nsinus {self.sinus} {n} ----------\n'
            else:
                self.interface.log_raw(filename,f'{prompt}\n{n} ----------')
                if self.g.settings_data['sailing']['sail_rephrase']:
                    self.interface.log_raw(filename,f'original prompt: {orig_prompt} \n{n} ----------')
                    sail_log = sail_log + f'original prompt: {orig_prompt}\n{n} ----------\n'

                sail_log = sail_log + f'{prompt}\n{n} ----------\n'

        else:
            self.interface.log_raw(filename,prompt.replace('\n','').strip())

            if self.g.settings_data['sailing']['sail_rephrase']:
                sail_log = sail_log + f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------\n'
            sail_log = sail_log + f'{prompt} \nsinus {self.sinus} {n} ----------\n'

        return sail_log

    def run_sail_automa_gen(self, prompt, images,folder=None):

        if self.g.settings_data['sailing']['sail_gen_enabled']:
            if folder != None:
                folder = shared.sanitize_path_component(folder)
            if self.g.settings_data['sailing']['sail_unload_llm']:
                self.interface.del_llm_model()


            self.step_gen_data = []
            gen_array = [self.g.settings_data['sailing']['sail_dimensions'],
                         self.g.settings_data['sailing']['sail_checkpoint'],
                         self.g.settings_data['sailing']['sail_sampler'],
                         self.g.settings_data['sailing']['sail_vae'],
                         self.g.settings_data['sailing']['sail_scheduler'],
                         ]
            combinations = self.prompt_iterator.combine_all_arrays_to_arrays(gen_array)
            if self.g.settings_data['sailing']['sail_gen_type'] == 'Linear':
                if self.gen_step == self.g.settings_data['sailing']['sail_gen_steps']:
                    self.gen_step_select += 1
                    if self.gen_step_select > len(combinations)-1:
                        self.gen_step_select = 0
                    self.gen_step = 0
                step_gen_data = combinations[self.gen_step_select]
            else:
                step_gen_data = combinations[random.randint(0, len(combinations)-1)]
            self.gen_step += 1
            if len(step_gen_data) > 0:
                self.g.settings_data['automa']['automa_width'] = step_gen_data[0].split(',')[0]
                self.g.settings_data['automa']['automa_height'] = step_gen_data[0].split(',')[1]
                self.g.settings_data['automa']['automa_checkpoint'] = step_gen_data[1]
                self.g.settings_data['automa']['automa_sampler'] = step_gen_data[2]
                self.g.settings_data['automa']['automa_vae'] = step_gen_data[3]
                self.g.settings_data['automa']['automa_scheduler'] = step_gen_data[4]

        if folder == None and self.g.settings_data['sailing']['sail_store_folders']:
            folder = shared.sanitize_path_component(self.g.settings_data['automa']['automa_checkpoint'])

        response = self.sail_automa_gen(prompt)
        if self.g.settings_data['sailing']['sail_unload_llm']:
            self.automa_client.unload_checkpoint()

        if response != '':
            for index, image in enumerate(response.get('images')):
                img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                if folder == None:
                    save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                else:
                    save_path = os.path.join(out_dir_t2i,folder)
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f'txt2img-{self.timestamp()}-{index}.png')

                self.automa_client.decode_and_save_base64(image, save_path)
                images.append(img)


        return images


    def automa_gen(self, prompt, images,folder=None):

        response = self.sail_automa_gen(prompt)
        if response != '':
            for index, image in enumerate(response.get('images')):
                img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                if folder == None:
                    save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                else:
                    save_path = os.path.join(out_dir_t2i,folder)
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f'txt2img-{self.timestamp()}-{index}.png')

                self.automa_client.decode_and_save_base64(image, save_path)
                images.append(img)

        return images

    def shorten_string(self, text, max_bytes=1000):
        """Shortens a string to a maximum of 1000 bytes.

        Args:
          text: The string to shorten.
          max_bytes: The maximum number of bytes allowed (default: 1000).

        Returns:
          The shortened string, truncated at the last whole word before reaching
          max_bytes.
        """
        if len(text) <= max_bytes:
            return text

        # Encode the text as UTF-8 to get byte length
        encoded_text = text.encode('utf-8')

        # Truncate the string while staying under the byte limit
        while len(encoded_text) > max_bytes:
            # Split text by words on space
            words = text.rsplit()
            # Remove the last word and try again
            text = ' '.join(words[:-1])
            encoded_text = text.encode('utf-8')

        return text

    def stop_job(self):
        self.g.job_running = False

class ui_staff:

    def __init__(self):

        self.g = globals.get_globals()
        self.hordeai_model_list = hordeai_models().read_model_list()
        self.g.context_prompt = ''
        self.g.last_prompt = ''
        self.g.last_negative_prompt = ''
        self.g.last_context = []
        self.max_top_k = 50


        self.prompt_input = gr.Textbox(placeholder="Make your prompts more creative", container=False, scale=7)

        self.hordeai_prompt_input = gr.TextArea(self.g.last_prompt, lines=10, label="Prompt")
        self.hordeai_negative_prompt_input = gr.TextArea(self.g.last_negative_prompt, lines=5, label="Negative Prompt")
        self.automa_prompt_input = gr.TextArea(self.g.last_prompt, lines=10, label="Prompt")
        self.automa_negative_prompt_input = gr.TextArea(self.g.last_negative_prompt, lines=5, label="Negative Prompt")

        self.horde_api_key = gr.TextArea(lines=1, label="API Key", value=self.g.settings_data['horde']['horde_api_key'], type='password')
        self.horde_model = gr.Dropdown(choices=self.hordeai_model_list.keys(), value=self.g.settings_data['horde']['horde_model'], label='Model')
        self.horde_sampler = gr.Dropdown(choices=["k_dpmpp_2s_a", "k_lms", "k_heun", "k_heun", "k_euler", "k_euler_a",
                                             "k_dpm_2", "k_dpm_2_a", "k_dpm_fast", "k_dpm_adaptive", "k_dpmpp_2s_a",
                                             "k_dpmpp_2m", "dpmsolver", "k_dpmpp_sde", "lcm", "DDIM"
                                             ], value=self.g.settings_data['horde']['horde_sampler'], label='Sampler')
        self.horde_steps = gr.Slider(0, 100, step=1, value=self.g.settings_data['horde']['horde_steps'], label="Steps",
                                info="Choose between 1 and 100")
        self.horde_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['horde']['horde_cfg_scale'], label="CFG Scale",
                              info="Choose between 1 and 20")
        self.horde_width = gr.Slider(0, 2048, step=1, value=self.g.settings_data['horde']['horde_width'], label="Width",
                                info="Choose between 1 and 2048")
        self.horde_height = gr.Slider(0, 2048, step=1, value=self.g.settings_data['horde']['horde_height'], label="Height",
                                 info="Choose between 1 and 2048")
        self.horde_clipskip = gr.Slider(0, 10, step=1, value=self.g.settings_data['horde']['horde_clipskip'], label="Clipskip",
                                   info="Choose between 1 and 10")

        self.automa_url = gr.TextArea(lines=1, label="API URL", value=self.g.settings_data['automa']['automa_url'])
        self.automa_sampler = gr.Dropdown(
            choices=self.g.settings_data['automa']['automa_samplers'], value=self.g.settings_data['automa']['automa_sampler'], label='Sampler')
        self.automa_checkpoint = gr.Dropdown(
            choices=self.g.settings_data['automa']['automa_checkpoints'], value=self.g.settings_data['automa']['automa_checkpoint'], label='Checkpoint')
        self.automa_steps = gr.Slider(1, 100, step=1, value=self.g.settings_data['automa']['automa_steps'], label="Steps",
                                 info="Choose between 1 and 100")
        self.automa_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['automa']['automa_cfg_scale'], label="CFG Scale",
                               info="Choose between 1 and 20")
        self.automa_width = gr.Slider(1, 2048, step=1, value=self.g.settings_data['automa']['automa_width'], label="Width",
                                 info="Choose between 1 and 2048")
        self.automa_height = gr.Slider(1, 2048, step=1, value=self.g.settings_data['automa']['automa_height'], label="Height",
                                  info="Choose between 1 and 2048")
        self.automa_Batch = gr.Slider(1, 250, step=1, value=self.g.settings_data['automa']['automa_batch'], label="Batch",
                                       info="The number of simultaneous images in each batch, range from 1-50.")
        self.automa_n_iter = gr.Slider(1, 500, step=1, value=self.g.settings_data['automa']['automa_n_iter'], label="Iterations",
                                      info="The number of sequential batches to be run, range from 1-500.")
        self.automa_save = gr.Checkbox(label="Save", info="Save the image?", value=self.g.settings_data['automa']['automa_save'])
        self.automa_save_on_api_host = gr.Checkbox(label="Save", info="Save the image on API host?", value=self.g.settings_data['automa']['automa_save_on_api_host'])

        self.automa_stop_button = gr.Button('Stop')







