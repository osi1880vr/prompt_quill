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
import re
import json
from collections import deque
from api import v1
import shared

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from post_process.summary import extractive_summary


from generators.hordeai.client import hordeai_client
from generators.automatics.client import automa_client
from generators.hordeai.client import hordeai_models
from settings.io import settings_io

from llm_fw import llm_interface_qdrant

out_dir = 'api_out'
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
        self.interface = llm_interface_qdrant.get_interface()
        self.settings_io = settings_io()
        self.max_top_k = 50
        self.automa_client = automa_client()
        self.api = v1
        self.api.run_api()

    def run_llm_response(self,query, history):
        prompt = self.interface.run_llm_response(query, history)

        return prompt


    def set_llm_settings(self, collection, model, embedding_model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
        self.g.settings_data['collection'] = collection
        self.g.settings_data['LLM Model'] = model
        self.g.settings_data['embedding_model'] = embedding_model
        self.g.settings_data['Temperature'] = temperature
        self.g.settings_data['Context Length'] = n_ctx
        self.g.settings_data['GPU Layers'] = n_gpu_layers
        self.g.settings_data['max output Tokens'] = max_tokens
        self.g.settings_data['top_k'] = top_k
        self.g.settings_data['Instruct Model'] = instruct
        self.settings_io.write_settings(self.g.settings_data)

    
    def set_civitai_settings(self, air, steps, cfg, width, heigth, clipskip):
        self.g.settings_data['civitai_Air'] = air
        self.g.settings_data['civitai_Steps'] = steps
        self.g.settings_data['civitai_CFG Scale'] = cfg
        self.g.settings_data['civitai_Width'] = width
        self.g.settings_data['civitai_Height'] = heigth
        self.g.settings_data['civitai_Clipskip'] = clipskip
        self.settings_io.write_settings(self.g.settings_data)

    
    def set_hordeai_settings(self, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
        self.g.settings_data['horde_api_key'] = api_key
        self.g.settings_data['horde_Model'] = model
        self.g.settings_data['horde_Sampler'] = sampler
        self.g.settings_data['horde_Steps'] = steps
        self.g.settings_data['horde_CFG Scale'] = cfg
        self.g.settings_data['horde_Width'] = width
        self.g.settings_data['horde_Height'] = heigth
        self.g.settings_data['horde_Clipskip'] = clipskip
        self.settings_io.write_settings(self.g.settings_data)

    
    def set_automa_settings(self,prompt, negative_prompt, sampler, checkpoint, steps, cfg, width, heigth, batch,n_iter, url, save, save_api):
        self.g.last_prompt = prompt
        self.g.last_negative_prompt = negative_prompt
        self.g.settings_data['automa_Sampler'] = sampler
        self.g.settings_data['automa_Steps'] = steps
        self.g.settings_data['automa_CFG Scale'] = cfg
        self.g.settings_data['automa_Width'] = width
        self.g.settings_data['automa_Height'] = heigth
        self.g.settings_data['automa_batch'] = batch
        self.g.settings_data['automa_n_iter'] = n_iter
        self.g.settings_data['automa_url'] = url
        self.g.settings_data['automa_save'] = save
        self.g.settings_data['automa_save_on_api_host'] = save_api
        self.g.settings_data['automa_Checkpoint'] = checkpoint
        self.settings_io.write_settings(self.g.settings_data)

    def set_automa_adetailer(self, automa_adetailer_enable,
                             automa_ad_use_inpaint_width_height,
                             automa_ad_model,
                             automa_ad_denoising_strength,
                             automa_ad_clip_skip,
                             automa_ad_confidence):
        self.g.settings_data['automa_adetailer_enable'] = automa_adetailer_enable
        self.g.settings_data['automa_ad_use_inpaint_width_height'] = automa_ad_use_inpaint_width_height
        self.g.settings_data['automa_ad_model'] = automa_ad_model
        self.g.settings_data['automa_ad_denoising_strength'] = automa_ad_denoising_strength
        self.g.settings_data['automa_ad_clip_skip'] = automa_ad_clip_skip
        self.g.settings_data['automa_ad_confidence'] = automa_ad_confidence
        self.settings_io.write_settings(self.g.settings_data)


    def set_sailing_settings(self,sail_text, sail_width, sail_depth, sail_generate, sail_target,
                             sail_summary, sail_rephrase, sail_rephrase_prompt, sail_gen_rephrase, sail_sinus,
                             sail_sinus_freq, sail_sinus_range, sail_add_style, sail_style, sail_add_search,
                             sail_search, sail_max_gallery_size, sail_dyn_neg,
                             sail_add_neg, sail_neg_prompt,sail_filter_text,sail_filter_not_text,sail_filter_context,
                             sail_filter_prompt):
        if self.g.sail_running:
            self.sail_depth_start = sail_depth

        self.g.settings_data['sail_text'] = sail_text
        self.g.settings_data['sail_width'] = sail_width
        self.g.settings_data['sail_depth'] = sail_depth
        self.g.settings_data['sail_generate'] = sail_generate
        self.g.settings_data['sail_target'] = sail_target
        self.g.settings_data['sail_summary'] = sail_summary
        self.g.settings_data['sail_rephrase'] = sail_rephrase
        self.g.settings_data['sail_rephrase_prompt'] = sail_rephrase_prompt
        self.g.settings_data['sail_gen_rephrase'] = sail_gen_rephrase
        self.g.settings_data['sail_sinus'] = sail_sinus
        self.g.settings_data['sail_sinus_freq'] = sail_sinus_freq
        self.g.settings_data['sail_sinus_range'] = sail_sinus_range
        self.g.settings_data['sail_add_style'] = sail_add_style
        self.g.settings_data['sail_style'] = sail_style
        self.g.settings_data['sail_add_search'] = sail_add_search
        self.g.settings_data['sail_search'] = sail_search
        self.g.settings_data['sail_max_gallery_size'] = sail_max_gallery_size
        self.g.settings_data['sail_dyn_neg'] = sail_dyn_neg
        self.g.settings_data['sail_add_neg'] = sail_add_neg
        self.g.settings_data['sail_neg_prompt'] = sail_neg_prompt
        self.g.settings_data['sail_filter_text'] = sail_filter_text
        self.g.settings_data['sail_filter_not_text'] = sail_filter_not_text
        self.g.settings_data['sail_filter_context'] = sail_filter_context
        self.g.settings_data['sail_filter_prompt'] = sail_filter_prompt
        self.settings_io.write_settings(self.g.settings_data)


    def set_prompt_input(self):
        return self.g.context_prompt


    def set_translate(self, translate):
        self.g.settings_data['translate'] = translate
        self.settings_io.write_settings(self.g.settings_data)


    def set_batch(self, batch):
        self.g.settings_data['batch'] = batch
        self.settings_io.write_settings(self.g.settings_data)


    def set_summary(self, summary):
        self.g.settings_data['summary'] = summary
        self.settings_io.write_settings(self.g.settings_data)


    def set_model(self, collection, model,embedding_model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
        self.g.settings_data['collection'] = collection
        self.g.settings_data['embedding_model'] = embedding_model
        self.set_llm_settings(collection, model, embedding_model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct)
        return self.interface.change_model(self.g.settings_data['model_list'][model], temperature, n_ctx, max_tokens, n_gpu_layers, top_k, instruct)
    
    
    def all_get_last_prompt(self):
        if self.g.settings_data['automa_checkpoints'] == []:
            self.g.settings_data['automa_checkpoints'] = self.get_automa_checkpoints()
            self.g.settings_data['automa_samplers'] = self.get_automa_sampler()


        return self.g.last_prompt, self.g.last_negative_prompt, \
            self.g.settings_data['horde_api_key'], self.g.settings_data['horde_Model'], self.g.settings_data['horde_Sampler'], self.g.settings_data[
            'horde_Steps'], self.g.settings_data['horde_CFG Scale'], self.g.settings_data['horde_Width'], self.g.settings_data['horde_Height'], \
            self.g.settings_data['horde_Clipskip'], self.g.last_prompt, self.g.last_negative_prompt, gr.update(choices=self.g.settings_data['automa_samplers'], value=self.g.settings_data['automa_Sampler']
            ), self.g.settings_data['automa_Steps'], self.g.settings_data['automa_CFG Scale'], self.g.settings_data[
            'automa_Width'], self.g.settings_data['automa_Height'], self.g.settings_data['automa_batch'],self.g.settings_data[
            'automa_n_iter'], self.g.settings_data['automa_url'], self.g.settings_data['automa_save'], self.g.settings_data[
            'automa_save_on_api_host'] , gr.update(choices=self.g.settings_data['automa_checkpoints'], value=self.g.settings_data['automa_Checkpoint'])
    
    

    
    def hordeai_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, self.g.settings_data['horde_api_key'], self.g.settings_data[
            'horde_Model'], self.g.settings_data['horde_Sampler'], self.g.settings_data['horde_Steps'], self.g.settings_data['horde_CFG Scale'], \
            self.g.settings_data['horde_Width'], self.g.settings_data['horde_Height'], self.g.settings_data['horde_Clipskip']
    
    
    def automa_get_last_prompt(self):
        return self.g.last_prompt, self.g.last_negative_prompt, gr.update(choices=self.g.settings_data['automa_samplers'], value=self.g.settings_data['automa_Sampler']
                                                                          ), self.g.settings_data['automa_Steps'], self.g.settings_data['automa_CFG Scale'], self.g.settings_data[
            'automa_Width'], self.g.settings_data['automa_Height'], self.g.settings_data['automa_batch'],self.g.settings_data['automa_n_iter'], self.g.settings_data[
            'automa_url'], self.g.settings_data['automa_save'], self.g.settings_data['automa_save_on_api_host'], gr.update(choices=self.g.settings_data['automa_checkpoints'], value=self.g.settings_data['automa_Checkpoint'])
    
    
    def get_llm_settings(self):
        return gr.update(choices=self.g.settings_data['collections_list'], value=self.g.settings_data['collection']),self.g.settings_data["LLM Model"], self.g.settings_data["embedding_model"],self.g.settings_data['Temperature'], self.g.settings_data['Context Length'], self.g.settings_data['GPU Layers'], self.g.settings_data['max output Tokens'], self.g.settings_data['top_k'], self.g.settings_data['Instruct Model']
    
    def get_sailing_settings(self):
        if self.g.settings_data['automa_checkpoints'] == []:
            self.g.settings_data['automa_checkpoints'] = self.get_automa_checkpoints()
            self.g.settings_data['automa_samplers'] = self.get_automa_sampler()
        if self.g.settings_data['automa_Sampler'] == '':
            self.g.settings_data['automa_Sampler'] = self.g.settings_data['automa_samplers'][0]
        if self.g.settings_data['automa_Checkpoint'] == '':
            self.g.settings_data['automa_Checkpoint'] = self.g.settings_data['automa_checkpoints'][0]

        return self.g.settings_data["sail_text"], self.g.settings_data['sail_width'], self.g.settings_data['sail_depth'
        ],self.g.settings_data["sail_generate"],self.g.settings_data["sail_target"],self.g.settings_data["sail_summary"
        ],self.g.settings_data["sail_rephrase"],self.g.settings_data["sail_rephrase_prompt"],self.g.settings_data["sail_gen_rephrase"
        ],self.g.settings_data["sail_sinus"],self.g.settings_data["sail_sinus_freq"],self.g.settings_data["sail_sinus_range"
        ],self.g.settings_data["sail_add_style"],self.g.settings_data["sail_style"],self.g.settings_data["sail_add_search"
        ],self.g.settings_data["sail_search"],self.g.settings_data["sail_max_gallery_size"],self.g.settings_data["sail_filter_text"
        ],self.g.settings_data["sail_filter_not_text"],self.g.settings_data["sail_filter_context"],self.g.settings_data["sail_filter_prompt"]

    def get_prompt_template(self):
        self.interface.prompt_template = self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]
        return self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]
    
    
    def set_prompt_template_select(self, value):
        self.g.settings_data['selected_template'] = value
        self.settings_io.write_settings(self.g.settings_data)
        return self.g.settings_data["prompt_templates"][value]
    
    def set_neg_prompt(self, value):
        self.g.settings_data['negative_prompt'] = value
        self.settings_io.write_settings(self.g.settings_data)

    def set_rephrase_instruction(self, value):
        self.g.settings_data['rephrase_instruction'] = value
        self.settings_io.write_settings(self.g.settings_data)

    def set_prompt_template(self, selection, prompt_text):
        return_data = self.interface.set_prompt(prompt_text)
        self.g.settings_data["prompt_templates"][selection] = prompt_text
        self.settings_io.write_settings(self.g.settings_data)
        return return_data
    

    def run_hordeai_generation(self, prompt, negative_prompt, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
        self.set_hordeai_settings(api_key, model, sampler, steps, cfg, width, heigth, clipskip)
        client = hordeai_client()
        return client.request_generation(api_key=api_key, prompt=prompt, negative_prompt=negative_prompt,
                                         sampler=sampler, model=model, steps=steps, cfg=cfg, width=width, heigth=heigth,
                                         clipskip=clipskip)

    def timestamp(self):
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    
    def run_automatics_generation(self, prompt, negative_prompt, sampler,checkpoint, steps, cfg, width, heigth, batch,n_iter, url, save,save_api):
        self.g.running = True
        self.set_automa_settings(prompt, negative_prompt, sampler, checkpoint, steps, cfg, width, heigth, batch,n_iter, url, save, save_api)
        self.g.last_prompt = prompt
        self.g.last_negative_prompt = negative_prompt

        response = self.automa_client.request_generation(prompt, negative_prompt, self.g.settings_data)
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
        response = self.automa_client.request_interrogation(base64_image,url)
        self.g.context_prompt = response
        return response


    def automa_switch_size(self,automa_Width,automa_Height):
        self.g.settings_data['automa_Width'] = automa_Height
        self.g.settings_data['automa_Height'] = automa_Width
        return automa_Height,automa_Width

    def automa_refresh(self):
        self.g.settings_data['automa_checkpoints'] = self.get_automa_checkpoints()
        self.g.settings_data['automa_samplers'] = self.get_automa_sampler()
        return gr.update(choices=self.g.settings_data['automa_samplers'], value=self.g.settings_data['automa_Sampler']), gr.update(choices=self.g.settings_data['automa_checkpoints'], value=self.g.settings_data['automa_Checkpoint'])

    def run_automa_interrogation_batch(self, image_filenames,url, save):
    
        all_response = ''
        output = ''
        n = 0
        for file in image_filenames:
            response = self.run_automa_interrogation(file[0],url)
            if all_response == '':
                all_response = response
            else:
                all_response = f'{all_response}\n{response}'
            output = f'{output}{response}\n{n} ---------\n'
            n += 1
            yield output
    
        if save:
            import time
            filename = f'{time.strftime("%Y%m%d-%H%M%S")}.txt'
            outfile = os.path.join(out_dir_i2t,filename)
            f = open(outfile,'a',encoding='utf8',errors='ignore')
            f.write(f'{all_response}\n')
            f.close()


    def get_automa_sampler(self):
        samplers = self.automa_client.get_samplers(self.g.settings_data['automa_url'])
        if samplers != -1:
            if self.g.settings_data['automa_Sampler'] == '':
                self.g.settings_data['automa_Sampler'] = samplers[0]
            return samplers
        else:
            return []

    def get_automa_checkpoints(self):
        checkpoints = self.automa_client.get_checkpoints(self.g.settings_data['automa_url'])
        if checkpoints != -1:
            if self.g.settings_data['automa_Checkpoint'] == '':
                self.g.settings_data['automa_Checkpoint'] = checkpoints[0]
            return checkpoints
        else:
            return []

    def get_next_target_new(self, nodes):

        if len(nodes) < self.g.settings_data['sail_depth']:
            self.g.settings_data['sail_depth'] = self.sail_depth_start + len(self.g.sail_history)

        if self.g.settings_data['sail_sinus']:
            self.sinus = int(math.sin(self.sail_sinus_count/10.0)*self.g.settings_data['sail_sinus_range'])
            self.sail_sinus_count += self.g.settings_data['sail_sinus_freq']
            self.g.settings_data['sail_depth'] += self.sinus
            if self.g.settings_data['sail_depth'] < 0:
                self.g.settings_data['sail_depth'] = 1

        if len(nodes) > 0:

            if self.g.settings_data['sail_target']:
                node = nodes[len(nodes)-1]
                payload = json.loads(node.payload['_node_content'])
                out = payload['text']
                self.g.sail_history.append(out)
                return out
            else:
                node = nodes[0]
                payload = json.loads(node.payload['_node_content'])
                out = payload['text']
                self.g.sail_history.append(out)
                return out
        else:
            return -1
    def check_api_avail(self):
        return self.automa_client.check_avail(self.g.settings_data['automa_url'])

    def sail_automa_gen(self, query):

        negative_prompt = self.g.settings_data['negative_prompt']

        if self.g.settings_data['sail_dyn_neg']:
            if len(self.g.negative_prompt_list) > 0:
                negative_prompt = shared.get_negative_prompt()


        if self.g.settings_data['sail_add_neg']:
            negative_prompt = f"{self.g.settings_data['sail_neg_prompt']}, {negative_prompt}"

        if len(negative_prompt) < 30:
            negative_prompt = self.g.settings_data['negative_prompt']

        return self.automa_client.request_generation(query,
                                                     negative_prompt,
                                                     self.g.settings_data)


    def log_prompt(self, filename, prompt, orig_prompt, n, sail_log):

        if self.g.settings_data['sail_sinus']:
            self.interface.log_raw(filename,f'{prompt} \nsinus {self.sinus} {n} ----------')
            if self.g.settings_data['sail_rephrase']:
                self.interface.log_raw(filename,f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------')
                sail_log = sail_log + f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------\n'

            sail_log = sail_log + f'{prompt} \nsinus {self.sinus} {n} ----------\n'
        else:
            self.interface.log_raw(filename,f'{prompt}\n{n} ----------')
            if self.g.settings_data['sail_rephrase']:
                self.interface.log_raw(filename,f'original prompt: {orig_prompt} \n{n} ----------')
                sail_log = sail_log + f'original prompt: {orig_prompt}\n{n} ----------\n'

            sail_log = sail_log + f'{prompt}\n{n} ----------\n'

        return sail_log



    def automa_gen(self, prompt, images):

        response = self.sail_automa_gen(prompt)

        for index, image in enumerate(response.get('images')):
            img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
            save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
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

    def get_new_prompt(self,query,n,prompt_discard_count,sail_steps,filename):
        prompt = ''
        query = self.prepare_query(query)
        if self.g.settings_data['sail_filter_prompt']:
            while 1:

                if self.g.sail_running is False:
                    break

                prompt = self.interface.retrieve_llm_completion(query)

                not_check = False
                check = False
                if len(self.g.settings_data['sail_filter_not_text']) > 0:
                    not_check = True
                    search = set(word.strip().lower() for word in self.g.settings_data['sail_filter_not_text'].split(","))
                    for word in search:
                        if word in prompt:
                            not_check = False
                            break

                if len(self.g.settings_data['sail_filter_text']) > 0:
                    search = set(word.strip().lower() for word in self.g.settings_data['sail_filter_text'].split(","))
                    for word in search:
                        if word in prompt:
                            check = True
                            break

                if not check and not not_check and prompt not in self.g.sail_history:
                    self.g.sail_history.append(prompt)
                    break
                n += 1
                new_nodes = self.interface.direct_search(self.g.settings_data['sail_text'],self.g.settings_data['sail_depth'],n)
                query = self.get_next_target_new(new_nodes)
                prompt_discard_count += 1
                sail_steps += 1

        else:
            prompt = self.interface.retrieve_llm_completion(query)

        prompt = shared.clean_llm_artefacts(prompt)


        if self.g.settings_data['sail_summary']:
            prompt = extractive_summary(prompt)

        orig_prompt = prompt
        if self.g.settings_data['sail_rephrase']:
            prompt = self.interface.rephrase(prompt, self.g.settings_data['sail_rephrase_prompt'])

        if self.g.settings_data['sail_add_style']:
            prompt = f'{self.g.settings_data["sail_style"]}, {prompt}'
            orig_prompt = f'{self.g.settings_data["sail_style"]}, {orig_prompt}'


        self.sail_log = self.log_prompt(filename, prompt, orig_prompt, n, self.sail_log)

        return prompt,orig_prompt,n,prompt_discard_count,sail_steps


    def prepare_query(self,query):
        if self.g.settings_data['sail_add_search']:
            query = f'{self.g.settings_data["sail_search"]}, {query}'

        if len(query) > 1000:
            query = extractive_summary(query,num_sentences=2)
            if len(query) > 1000:
                query = self.shorten_string(query)

        return query

    def run_t2t_sail(self):

        """
        Runs a Text-to-Text (T2T) SAIL (possibly referring to a creative text generation process) loop based on user-provided settings.

        This function iterates through a specified number of "sails" (iterations) as defined by the `sail_width` setting.
        In each sail, it performs the following actions:

            1. Initializes variables based on user settings:
                * Sets a flag indicating SAIL is running.
                * Initializes an empty history list.
                * Records starting depth for text generation.
                * Initializes counters and variables for sinusoidal variations (purpose unclear based on provided code).
                * Initializes variables for logging and image storage.

            2. Prepares the query based on settings:
                * Fetches the query text from settings.
                * Optionally translates the query if enabled in settings.
                * Optionally adds a search prefix to the query if enabled.

            3. Iterates through each "sail" step:
                * Retrieves a query based on settings (potentially using an interface).
                * Cleans potential artefacts from the retrieved query using the `clean_llm_artefacts` function.
                * Optionally summarizes the query using an external `extractive_summary` function (if enabled).
                * Optionally adds a style prefix to the query if enabled.
                * Logs the query with additional information (sinusoidal value, step number) based on settings.
                * Retrieves top-k most relevant nodes based on the current query and depth (using an interface).

                * (Optional) Generates creative text:
                    * If generation is enabled, calls the `sail_automa_gen` function to generate text (implementation not provided).
                    * Processes and saves any generated images from the response.

            4. Yields results after each sail:
                * Yields the accumulated log and a list of generated images (if any) for each sail.

            5. Updates query for next sail:
                * Extracts the next target text from retrieved nodes using the `get_next_target` function (implementation not provided).
                * Handles potential early termination due to context rotation or user interruption.

        Args:
            self: Reference to the class instance (likely holds configuration and state).

        Yields:
            tuple: A tuple containing the accumulated log for the sail and a list of generated images (if any).
        """

        self.g.sail_running = True
        self.g.sail_history = []
        self.sail_depth_start = self.g.settings_data['sail_depth']
        self.sail_sinus_count = 1.0
        self.sinus = 0
        self.sail_log = ''
        self.images_done = 1
        query = self.g.settings_data['sail_text']
        images = deque(maxlen=int(self.g.settings_data['sail_max_gallery_size']))
        filename = os.path.join(out_dir_t2t, f'journey_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')


        if self.g.settings_data['translate']:
            query = self.interface.translate(self.g.settings_data['sail_text'])

        prompt_discard_count = 0
        n = 0
        sail_steps = self.g.settings_data['sail_width']
        while n < sail_steps:

            try:

                prompt,orig_prompt,n,prompt_discard_count,sail_steps = self.get_new_prompt(query,n,prompt_discard_count,sail_steps,filename)

                new_nodes = self.interface.direct_search(self.g.settings_data['sail_text'],self.g.settings_data['sail_depth'],n)



                if self.g.settings_data['sail_generate']:
                    if self.g.settings_data['sail_gen_rephrase']:
                        images = self.automa_gen(orig_prompt, images)
                        yield self.sail_log,list(images),f'{self.images_done} image(s) done\n{prompt_discard_count} prompts filtered'
                    images = self.automa_gen(prompt, images)
                    yield self.sail_log,list(images),f'{self.images_done} image(s) done\n{prompt_discard_count} prompts filtered'
                else:
                    yield self.sail_log,[],f'{self.images_done} image(s) done\n{prompt_discard_count} prompts filtered'

                query = self.get_next_target_new(new_nodes)

                if query == -1:
                    self.interface.log_raw(filename,f'{n} sail is finished early due to rotating context')
                    yield self.sail_log,list(images),f'after {self.images_done} image(s), sail is finished early due to no more context\n{prompt_discard_count} prompts filtered'
                    break

                if self.g.sail_running is False:
                    break

            except Exception as e:
                n += 1
                sail_steps += 1
                new_nodes = self.interface.direct_search(self.g.settings_data['sail_text'],self.g.settings_data['sail_depth'],n)
                query = self.get_next_target_new(new_nodes)
                print('some error happened: ',str(e))
                time.sleep(5)
            finally:
                n += 1
                self.images_done += 1

    def run_t2t_show_sail(self):
        self.g.sail_running = True
        self.g.settings_data['automa_batch'] = 1
        self.g.settings_data['automa_n_iter'] = 1
        self.g.sail_history = []
        self.sail_depth_start = self.g.settings_data['sail_depth']
        self.sail_sinus_count = 1.0
        sail_log = ''
        query = self.g.settings_data['sail_text']

        filename = os.path.join(out_dir_t2t, f'journey_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')

        if self.g.settings_data['translate']:
            query = self.interface.translate(query)


        prompt_discard_count = 0
        n = 0
        sail_steps = self.g.settings_data['sail_width']
        while n < sail_steps:

            try:



                prompt,orig_prompt,n,prompt_discard_count,sail_steps = self.get_new_prompt(query,n,prompt_discard_count,sail_steps,filename)

                new_nodes = self.interface.direct_search(self.g.settings_data['sail_text'],self.g.settings_data['sail_depth'],n)

                if self.g.settings_data['sail_generate']:
                    response = self.sail_automa_gen(prompt)

                    for index, image in enumerate(response.get('images')):
                        img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                        save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                        self.automa_client.decode_and_save_base64(image, save_path)
                        yield prompt,img
                else:
                    yield prompt,None

                query = self.get_next_target_new(new_nodes)
                if query == -1:
                    self.interface.log_raw(filename,f'{n} sail is finished early due to rotating context')
                    break
                if self.g.sail_running is False:
                    break
            except Exception as e:
                new_nodes = self.interface.direct_search(self.g.settings_data['sail_text'],self.g.settings_data['sail_depth'],n)
                query = self.get_next_target_new(new_nodes)
                print('some error happened: ',str(e))
                time.sleep(5)
    def stop_t2t_sail(self):
        self.g.sail_running = False
    def stop_all(self):
        self.g.running = False
    def variable_outputs(self, k):
        self.g.settings_data['top_k'] = int(k)
        self.interface.set_top_k(self.g.settings_data['top_k'])
        k = int(k)
        out = [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (self.max_top_k - k)
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
    
        if len(context) < self.max_top_k - 1:
            x = range(len(context), self.max_top_k - 1)
            for n in x:
                context.append('')
    
        return context  # .append(text)



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

    def load_preset(self,name):
        self.g.settings_data = settings_io().load_preset(name)
        return 'OK'

    def save_preset(self, name):
        status = settings_io().save_preset(name,self.g.settings_data)
        return status

    def load_preset_list(self):
        self.g.settings_data['preset_list'] = settings_io().load_preset_list()
        return gr.Dropdown(choices=self.g.settings_data['preset_list'])



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

        self.LLM = gr.Dropdown(
            self.g.settings_data['model_list'].keys(), value=self.g.settings_data['LLM Model'], label="LLM Model",
            info="Will add more LLMs later!"
        )
        self.embedding_model = gr.Dropdown(
            self.g.settings_data['embedding_model_list'], value=self.g.settings_data['embedding_model'], label="Embedding Model",
            info="If you dont know better, use all-MiniLM-L12-v2!"
        )

        self.collection = gr.Dropdown(
            self.g.settings_data['collections_list'], value=self.g.settings_data['collection'], label="Collection",
            info="If you got more than one collection!"
        )
        self.Temperature = gr.Slider(0, 1, step=0.1, value=self.g.settings_data['Temperature'], label="Temperature",
                                info="Choose between 0 and 1")
        self.Context = gr.Slider(0, 8192, step=1, value=self.g.settings_data['Context Length'], label="Context Length",
                            info="Choose between 1 and 8192")
        self.GPU = gr.Slider(0, 1024, step=1, value=self.g.settings_data['GPU Layers'], label="GPU Layers",
                        info="Choose between 1 and 1024")
        self.max = gr.Slider(0, 1024, step=1, value=self.g.settings_data['max output Tokens'], label="max output Tokens",
                        info="Choose between 1 and 1024")
        self.top_k = gr.Slider(0, self.max_top_k, step=1, value=self.g.settings_data['top_k'],
                          label="how many entrys to be fetched from the vector store",
                          info="Choose between 1 and 50 be careful not to overload the context window of the LLM")
        self.Instruct = gr.Checkbox(label='Instruct Model', value=self.g.settings_data['Instruct Model'])


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
            choices=self.g.settings_data['automa_samplers'], value=self.g.settings_data['automa_Sampler'], label='Sampler')
        self.automa_Checkpoint = gr.Dropdown(
            choices=self.g.settings_data['automa_checkpoints'], value=self.g.settings_data['automa_Checkpoint'], label='Checkpoint')
        self.automa_Steps = gr.Slider(1, 100, step=1, value=self.g.settings_data['automa_Steps'], label="Steps",
                                 info="Choose between 1 and 100")
        self.automa_CFG = gr.Slider(0, 20, step=0.1, value=self.g.settings_data['automa_CFG Scale'], label="CFG Scale",
                               info="Choose between 1 and 20")
        self.automa_Width = gr.Slider(1, 2048, step=1, value=self.g.settings_data['automa_Width'], label="Width",
                                 info="Choose between 1 and 2048")
        self.automa_Height = gr.Slider(1, 2048, step=1, value=self.g.settings_data['automa_Height'], label="Height",
                                  info="Choose between 1 and 2048")
        self.automa_Batch = gr.Slider(1, 250, step=1, value=self.g.settings_data['automa_batch'], label="Batch",
                                       info="The number of simultaneous images in each batch, range from 1-50.")
        self.automa_n_iter = gr.Slider(1, 500, step=1, value=self.g.settings_data['automa_n_iter'], label="Iterations",
                                      info="The number of sequential batches to be run, range from 1-500.")
        self.automa_save = gr.Checkbox(label="Save", info="Save the image?", value=self.g.settings_data['automa_save'])
        self.automa_save_on_api_host = gr.Checkbox(label="Save", info="Save the image on API host?", value=self.g.settings_data['automa_save_on_api_host'])

        self.automa_stop_button = gr.Button('Stop')

        self.prompt_template = gr.TextArea(self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]], lines=20)
        self.prompt_template_select = gr.Dropdown(choices=self.g.settings_data["prompt_templates"].keys(),
                                             value=self.g.settings_data["selected_template"], label='Template', interactive=True)
        self.prompt_template_status = gr.TextArea(lines=1, label="Refresh Status", placeholder='Status')

        self.sail_result_last_image = gr.Image(label='last Image')




