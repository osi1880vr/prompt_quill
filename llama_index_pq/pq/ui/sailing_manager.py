# sailing_manager.py
import globals
import os
import gradio as gr
from collections import deque
from PIL import Image
from io import BytesIO
import time
import json
import base64
import random
import math
import shared
from post_process.summary import extractive_summary
from generators.automatics.client import automa_client
from settings.io import settings_io
from prompt_iteration import prompt_iterator
from llm_fw import llm_interface_qdrant

class SailingManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.interface = llm_interface_qdrant.get_interface()
        self.settings_io = settings_io()
        self.automa_client = automa_client()
        self.prompt_iterator = prompt_iterator()

        # Sailing-specific state
        self.sail_depth_start = 0
        self.sail_sinus_count = 1.0
        self.sinus = 0
        self.sail_log = ''
        self.images_done = 1
        self.gen_step = 0
        self.gen_step_select = 0
        self.prompt_array_index = {}

        # Output directories
        self.out_dir = '../api_out'
        self.out_dir_t2t = os.path.join(self.out_dir, 'txt2txt')
        self.out_dir_t2i = os.path.join(self.out_dir, 'txt2img')
        os.makedirs(self.out_dir_t2t, exist_ok=True)
        os.makedirs(self.out_dir_t2i, exist_ok=True)

    # Utility methods
    def timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def shorten_string(self, text, max_bytes=1000):
        if len(text) <= max_bytes:
            return text
        encoded_text = text.encode('utf-8')
        while len(encoded_text) > max_bytes:
            words = text.rsplit()
            text = ' '.join(words[:-1])
            encoded_text = text.encode('utf-8')
        return text

    def prepare_query(self, query):
        if self.g.settings_data['sailing']['sail_add_search']:
            query = f'{self.g.settings_data["sailing"]["sail_search"]}, {query}'
        if len(query) > 1000:
            query = extractive_summary(query, num_sentences=2)
            if len(query) > 1000:
                query = self.shorten_string(query)
        return query

    def log_prompt(self, filename, prompt, orig_prompt, n, sail_log):
        if self.g.settings_data['sailing']['sail_generate']:
            if self.g.settings_data['sailing']['sail_sinus']:
                self.interface.log_raw(filename, f'{prompt} \nsinus {self.sinus} {n} ----------')
                if self.g.settings_data['sailing']['sail_rephrase']:
                    self.interface.log_raw(filename, f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------')
                    sail_log += f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------\n'
                sail_log += f'{prompt} \nsinus {self.sinus} {n} ----------\n'
            else:
                self.interface.log_raw(filename, f'{prompt}\n{n} ----------')
                if self.g.settings_data['sailing']['sail_rephrase']:
                    self.interface.log_raw(filename, f'original prompt: {orig_prompt} \n{n} ----------')
                    sail_log += f'original prompt: {orig_prompt}\n{n} ----------\n'
                sail_log += f'{prompt}\n{n} ----------\n'
        else:
            self.interface.log_raw(filename, prompt.replace('\n', '').strip())
            if self.g.settings_data['sailing']['sail_rephrase']:
                sail_log += f'original prompt: {orig_prompt} \nsinus {self.sinus} {n} ----------\n'
            sail_log += f'{prompt} \nsinus {self.sinus} {n} ----------\n'
        return sail_log

    def get_next_target_new(self, nodes):
        if len(nodes) < self.g.settings_data['sailing']['sail_depth']:
            self.g.settings_data['sailing']['sail_depth'] = self.sail_depth_start + len(self.g.sail_history)

        if self.g.settings_data['sailing']['sail_sinus']:
            self.sinus = int(math.sin(self.sail_sinus_count / 10.0) * self.g.settings_data['sailing']['sail_sinus_range'])
            self.sail_sinus_count += self.g.settings_data['sailing']['sail_sinus_freq']
            self.g.settings_data['sailing']['sail_depth'] += self.sinus
            if self.g.settings_data['sailing']['sail_depth'] < 0:
                self.g.settings_data['sailing']['sail_depth'] = 1

        if len(nodes) > 0:
            node = nodes[-1] if self.g.settings_data['sailing']['sail_target'] else nodes[0]
            payload = json.loads(node.payload['_node_content'])
            out = payload['text']
            self.g.sail_history.append(out)
            return out
        return -1

    def sail_automa_gen(self, query):
        negative_prompt = self.g.settings_data['negative_prompt']
        if self.g.settings_data['sailing']['sail_dyn_neg'] and self.g.negative_prompt_list:
            negative_prompt = shared.get_negative_prompt()
        if self.g.settings_data['sailing']['sail_add_neg']:
            negative_prompt = f"{self.g.settings_data['sailing']['sail_neg_embed']},{self.g.settings_data['sailing']['sail_neg_prompt']}, {negative_prompt}"
        if len(negative_prompt) < 30:
            negative_prompt = self.g.settings_data['negative_prompt']
        self.g.act_neg_prompt = negative_prompt
        return self.automa_client.request_generation(query, negative_prompt, self.g.settings_data)

    def run_sail_automa_gen(self, prompt, images, folder=None):
        if self.g.settings_data['sailing']['sail_gen_enabled']:
            if folder:
                folder = shared.sanitize_path_component(folder)
            if self.g.settings_data['sailing']['sail_unload_llm']:
                self.interface.del_llm_model()

            gen_array = [
                self.g.settings_data['sailing']['sail_dimensions'],
                self.g.settings_data['sailing']['sail_checkpoint'],
                self.g.settings_data['sailing']['sail_sampler'],
                self.g.settings_data['sailing']['sail_vae'],
                self.g.settings_data['sailing']['sail_scheduler'],
            ]
            combinations = self.prompt_iterator.combine_all_arrays_to_arrays(gen_array)
            if self.g.settings_data['sailing']['sail_gen_type'] == 'Linear':
                if self.gen_step == self.g.settings_data['sailing']['sail_gen_steps']:
                    self.gen_step_select = (self.gen_step_select + 1) % len(combinations)
                    self.gen_step = 0
                step_gen_data = combinations[self.gen_step_select]
            else:
                step_gen_data = combinations[random.randint(0, len(combinations) - 1)]
            self.gen_step += 1
            if step_gen_data:
                self.g.settings_data['automa']['automa_width'], self.g.settings_data['automa']['automa_height'] = step_gen_data[0].split(',')
                self.g.settings_data['automa']['automa_checkpoint'] = step_gen_data[1]
                self.g.settings_data['automa']['automa_sampler'] = step_gen_data[2]
                self.g.settings_data['automa']['automa_vae'] = step_gen_data[3]
                self.g.settings_data['automa']['automa_scheduler'] = step_gen_data[4]
                print('Sail Gen done with: ', step_gen_data)

        if not folder and self.g.settings_data['sailing']['sail_store_folders']:
            folder = shared.sanitize_path_component(self.g.settings_data['automa']['automa_checkpoint'])

        response = self.sail_automa_gen(prompt)
        if response == '':
            self.g.job_running = False
            print('Sailing stopped due to no response from image generator')
        if self.g.settings_data['sailing']['sail_unload_llm']:
            self.automa_client.unload_checkpoint()

        if response:
            for index, image in enumerate(response.get('images')):
                img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                save_path = os.path.join(self.out_dir_t2i, folder or '', f'txt2img-{self.timestamp()}-{index}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.automa_client.decode_and_save_base64(image, save_path)
                images.append(img)
        return images

    def get_new_prompt(self, query, n, prompt_discard_count, sail_steps, filename, sail_keep_text=False, retry_count=0):
        prompt = ''
        query = self.prepare_query(query)
        if self.g.settings_data['sailing']['sail_filter_prompt']:
            while True:
                if not self.g.job_running:
                    break
                prompt = self.interface.retrieve_llm_completion(query, sail_keep_text=sail_keep_text)
                filtered = shared.check_filtered(query)
                if not filtered and prompt not in self.g.sail_history:
                    self.g.sail_history.append(prompt)
                    break
                n += 1
                new_nodes = self.interface.direct_search(self.g.settings_data['sailing']['sail_text'], self.g.settings_data['sailing']['sail_depth'], n)
                query = self.get_next_target_new(new_nodes)
                prompt_discard_count += 1
                sail_steps += 1
        else:
            prompt = self.interface.retrieve_llm_completion(query, sail_keep_text=sail_keep_text)

        prompt = shared.clean_llm_artefacts(prompt)
        if self.g.settings_data['sailing']['sail_summary']:
            prompt = extractive_summary(prompt)
        orig_prompt = prompt
        if self.g.settings_data['sailing']['sail_rephrase']:
            prompt = self.interface.rephrase(prompt, self.g.settings_data['sailing']['sail_rephrase_prompt'])

        style_prompt = ''
        if self.g.settings_data['sailing']['sail_add_style']:
            processor = shared.WildcardResolver()
            style_prompt = processor.resolve_prompt(self.g.settings_data['sailing']["sail_style"])
            if '%%PROMPT%%' in style_prompt:
                prompt = style_prompt.replace('%%PROMPT%%', prompt)
                prompt = f'{self.g.settings_data["sailing"]["sail_pos_embed"]},{prompt}'
                orig_prompt = style_prompt.replace('%%PROMPT%%', orig_prompt)
                orig_prompt = f'{self.g.settings_data["sailing"]["sail_pos_embed"]}, {orig_prompt}'
            else:
                prompt = f'{self.g.settings_data["sailing"]["sail_pos_embed"]}, {style_prompt}, {prompt}'
                orig_prompt = f'{self.g.settings_data["sailing"]["sail_pos_embed"]}, {style_prompt}, {orig_prompt}'

        if not prompt or len(prompt) < 10 + len(style_prompt) + len(self.g.settings_data['sailing']["sail_pos_embed"]) and retry_count < 10:
            print(f'empty or too short prompt will retry {10 - retry_count} times, each retry fetches a new query and so dives deeper into the data')
            n += 1
            retry_count += 1
            new_nodes = self.interface.direct_search(self.g.settings_data['sailing']['sail_text'], self.g.settings_data['sailing']['sail_depth'], n)
            query = self.get_next_target_new(new_nodes)
            return self.get_new_prompt(query, n, prompt_discard_count, sail_steps, filename, sail_keep_text, retry_count)

        self.sail_log = self.log_prompt(filename, prompt, orig_prompt, n, self.sail_log)
        return prompt.strip(), orig_prompt.strip(), n, prompt_discard_count, sail_steps

    def check_black_images(self, prompt, images, filename):
        checkimages = deque(maxlen=int(self.g.settings_data['sailing']['sail_max_gallery_size']))
        batch = self.g.settings_data['automa']['automa_batch']
        iter = self.g.settings_data['automa']['automa_n_iter']
        vae = self.g.settings_data['automa']['automa_vae']

        self.g.settings_data['automa']['automa_batch'] = 1
        self.g.settings_data['automa']['automa_n_iter'] = 1

        for image in list(images):
            if shared.is_image_black(image):
                self.log_prompt(filename, prompt, self.g.act_neg_prompt, 0, '')
                self.g.settings_data['automa']['automa_vae'] = self.g.settings_data['automa']['automa_alt_vae']
                images = self.run_sail_automa_gen(prompt, [])
                self.g.settings_data['automa']['automa_vae'] = vae
                checkimages = self.run_sail_automa_gen(prompt, checkimages)
            else:
                checkimages.append(image)

        self.g.settings_data['automa']['automa_vae'] = vae
        self.g.settings_data['automa']['automa_batch'] = batch
        self.g.settings_data['automa']['automa_n_iter'] = iter
        return checkimages

    # UI-specific methods
    def setting_dropdown(self, label, choices, initial_value):
        return gr.Dropdown(label=label, choices=choices, value=initial_value, interactive=True)

    def set_sailing_settings(self, sail_text, sail_keep_text, sail_width, sail_depth, sail_generate, sail_summary,
                             sail_rephrase, sail_rephrase_prompt, sail_gen_rephrase, sail_sinus, sail_sinus_freq,
                             sail_sinus_range, sail_add_style, sail_style, sail_add_search, sail_search,
                             sail_max_gallery_size, sail_dyn_neg, sail_add_neg, sail_neg_prompt, sail_filter_text,
                             sail_filter_not_text, sail_filter_context, sail_filter_prompt, sail_neg_filter_text,
                             sail_neg_filter_not_text, sail_neg_filter_context, automa_alt_vae, sail_checkpoint,
                             sail_sampler, sail_vae, sail_dimensions, sail_gen_type, sail_gen_any_combination,
                             sail_gen_steps, sail_gen_enabled, sail_override_settings_restore, sail_store_folders,
                             sail_depth_preset, sail_scheduler, sail_unload_llm, sail_neg_embed, sail_pos_embed):
        if self.g.job_running:
            self.sail_depth_start = sail_depth

        self.g.settings_data['sailing'].update({
            'sail_text': sail_text, 'sail_keep_text': sail_keep_text, 'sail_width': sail_width,
            'sail_depth': sail_depth, 'sail_generate': sail_generate, 'sail_summary': sail_summary,
            'sail_rephrase': sail_rephrase, 'sail_rephrase_prompt': sail_rephrase_prompt,
            'sail_gen_rephrase': sail_gen_rephrase, 'sail_sinus': sail_sinus, 'sail_sinus_freq': sail_sinus_freq,
            'sail_sinus_range': sail_sinus_range, 'sail_add_style': sail_add_style, 'sail_style': sail_style,
            'sail_add_search': sail_add_search, 'sail_search': sail_search, 'sail_max_gallery_size': sail_max_gallery_size,
            'sail_dyn_neg': sail_dyn_neg, 'sail_add_neg': sail_add_neg, 'sail_neg_prompt': sail_neg_prompt,
            'sail_filter_text': sail_filter_text, 'sail_filter_not_text': sail_filter_not_text,
            'sail_filter_context': sail_filter_context, 'sail_filter_prompt': sail_filter_prompt,
            'sail_neg_filter_text': sail_neg_filter_text, 'sail_neg_filter_not_text': sail_neg_filter_not_text,
            'sail_neg_filter_context': sail_neg_filter_context, 'sail_checkpoint': sail_checkpoint,
            'sail_sampler': sail_sampler, 'sail_vae': sail_vae, 'sail_dimensions': sail_dimensions,
            'sail_gen_type': sail_gen_type, 'sail_gen_any_combination': sail_gen_any_combination,
            'sail_gen_steps': sail_gen_steps, 'sail_gen_enabled': sail_gen_enabled,
            'sail_override_settings_restore': sail_override_settings_restore, 'sail_store_folders': sail_store_folders,
            'sail_depth_preset': sail_depth_preset, 'sail_scheduler': sail_scheduler, 'sail_unload_llm': sail_unload_llm,
            'sail_pos_embed': sail_pos_embed, 'sail_neg_embed': sail_neg_embed
        })
        self.g.settings_data['automa']['automa_alt_vae'] = automa_alt_vae
        self.settings_io.write_settings(self.g.settings_data)

    def get_sailing_settings(self):
        if not self.g.settings_data['automa']['automa_checkpoints']:
            self.g.settings_data['automa']['automa_checkpoints'] = self.automa_client.get_checkpoints(self.g.settings_data['automa']['automa_url'])
            self.g.settings_data['automa']['automa_samplers'] = self.automa_client.get_samplers(self.g.settings_data['automa']['automa_url'])
            self.g.settings_data['automa']['automa_vaes'] = self.automa_client.get_vaes(self.g.settings_data['automa']['automa_url'])
        for key, fallback in [
            ('automa_sampler', self.g.settings_data['automa']['automa_samplers']),
            ('automa_checkpoint', self.g.settings_data['automa']['automa_checkpoints']),
            ('automa_vae', self.g.settings_data['automa']['automa_vaes']),
            ('automa_scheduler', self.g.settings_data['automa']['automa_schedulers'] if self.g.settings_data['automa']['automa_new_forge'] else [])
        ]:
            if not self.g.settings_data['automa'][key] and fallback:
                self.g.settings_data['automa'][key] = fallback[0]

        sailing = self.g.settings_data['sailing']
        return (
            sailing["sail_text"], sailing['sail_width'], sailing['sail_depth'], sailing["sail_generate"],
            sailing["sail_summary"], sailing["sail_rephrase"], sailing["sail_rephrase_prompt"],
            sailing["sail_gen_rephrase"], sailing["sail_sinus"], sailing["sail_sinus_freq"],
            sailing["sail_sinus_range"], sailing["sail_add_style"], sailing["sail_style"],
            sailing["sail_add_search"], sailing["sail_search"], sailing["sail_max_gallery_size"],
            sailing["sail_filter_text"], sailing["sail_filter_not_text"], sailing["sail_filter_context"],
            sailing["sail_filter_prompt"], sailing["sail_neg_filter_text"], sailing["sail_neg_filter_not_text"],
            sailing["sail_neg_filter_context"], gr.update(choices=self.g.settings_data['automa']['automa_vaes'], value=self.g.settings_data['automa']['automa_alt_vae']),
            sailing["sail_checkpoint"], sailing["sail_sampler"], sailing["sail_vae"], sailing["sail_dimensions"],
            sailing["sail_gen_type"], sailing["sail_gen_steps"], sailing["sail_gen_enabled"],
            sailing["sail_override_settings_restore"], sailing["sail_store_folders"], sailing["sail_depth_preset"],
            sailing['sail_scheduler'], sailing["sail_neg_embed"], sailing['sail_pos_embed']
        )

    def automa_sail_refresh(self):
        self.g.settings_data['automa']['automa_checkpoints'] = self.automa_client.get_checkpoints(self.g.settings_data['automa']['automa_url'])
        self.g.settings_data['automa']['automa_samplers'] = self.automa_client.get_samplers(self.g.settings_data['automa']['automa_url'])
        self.g.settings_data['automa']['automa_schedulers'] = self.automa_client.get_schedulers(self.g.settings_data['automa']['automa_url'])
        self.g.settings_data['automa']['automa_vaes'] = self.automa_client.get_vaes(self.g.settings_data['automa']['automa_url'])
        return (
            gr.update(choices=self.g.settings_data['automa']['automa_samplers'], value=self.g.settings_data['sailing']['sail_sampler']),
            gr.update(choices=self.g.settings_data['automa']['automa_checkpoints'], value=self.g.settings_data['sailing']['sail_checkpoint']),
            gr.update(choices=self.g.settings_data['automa']['automa_vaes'], value=self.g.settings_data['sailing']['sail_vae']),
            gr.update(choices=self.g.settings_data['automa']['automa_schedulers'], value=self.g.settings_data['sailing']['sail_scheduler']),
            self.g.settings_data['automa']['automa_checkpoints']
        )

    def run_t2t_sail(self):
        self.g.settings_data['sailing']['sail_target'] = True
        self.g.job_running = True
        self.g.sail_history = []
        self.sail_depth_start = self.g.settings_data['sailing']['sail_depth']
        self.sail_sinus_count = 1.0
        self.sinus = 0
        self.sail_log = ''
        self.images_done = 1
        self.g.act_neg_prompt = ''
        query = self.g.settings_data['sailing']['sail_text']
        images = deque(maxlen=int(self.g.settings_data['sailing']['sail_max_gallery_size']))
        filename = os.path.join(self.out_dir_t2t, f'journey_log_{self.timestamp()}.txt')
        black_images_filename = os.path.join(self.out_dir_t2t, f'black_images_{self.timestamp()}.txt')

        if self.g.settings_data['translate']:
            query = self.interface.translate(self.g.settings_data['sailing']['sail_text'])

        prompt_discard_count = 0
        n = 1
        sail_steps = self.g.settings_data['sailing']['sail_width']
        context_count = self.interface.count_context().count
        possible_images = int(context_count / self.g.settings_data['sailing']['sail_depth']) - int(self.g.settings_data['sailing']['sail_depth_preset'] / self.g.settings_data['sailing']['sail_depth'])

        yield self.sail_log, [], f"Sailing for {sail_steps} steps has started please be patient for the first result to arrive, there is {context_count} possible context entries in the ocean based on your filter settings, based on your distance setting there might be {possible_images} images possible"
        query = shared.WildcardResolver().resolve_prompt(query)
        new_nodes = self.interface.direct_search(query, self.g.settings_data['sailing']['sail_depth'], 0)
        query = self.get_next_target_new(new_nodes)

        while n < sail_steps + 1:
            try:
                if query == -1:
                    self.g.job_running = False
                    yield self.sail_log, [], 'Something went wrong, there is no valid query anymore'
                    break

                prompt, orig_prompt, n, prompt_discard_count, sail_steps = self.get_new_prompt(query, n, prompt_discard_count, sail_steps, filename, self.g.settings_data['sailing']['sail_keep_text'])
                new_nodes = self.interface.direct_search(self.g.settings_data['sailing']['sail_text'], self.g.settings_data['sailing']['sail_depth'], n)

                if self.g.settings_data['sailing']['sail_generate']:
                    if self.g.settings_data['sailing']['sail_gen_rephrase']:
                        images = self.run_sail_automa_gen(orig_prompt, images)
                        images = self.check_black_images(prompt, images, black_images_filename)
                        yield self.sail_log, list(images), f'{self.images_done} image(s) done\n{prompt_discard_count} prompts filtered'
                    images = self.run_sail_automa_gen(prompt, images)
                    images = self.check_black_images(prompt, images, black_images_filename)





                    yield self.sail_log, list(images), f'{self.images_done} image(s) done\n{prompt_discard_count} prompts filtered'
                else:
                    yield self.sail_log, [], f'{self.images_done} prompts(s) done\n{prompt_discard_count} prompts filtered'

                query = self.get_next_target_new(new_nodes)
                if query == -1:
                    self.interface.log_raw(filename, f'{n} sail is finished early due to no more context')
                    yield self.sail_log, list(images), f'after {self.images_done} image(s), sail is finished early due to no more context\n{prompt_discard_count} prompts filtered'
                    break

            except Exception as e:
                n += 1
                sail_steps += 1
                new_nodes = self.interface.direct_search(self.g.settings_data['sailing']['sail_text'], self.g.settings_data['sailing']['sail_depth'], n)
                query = self.get_next_target_new(new_nodes)
                print('some error happened: ', str(e))
                time.sleep(5)
            finally:
                if not self.g.job_running:
                    yield self.sail_log, list(images) if self.g.settings_data['sailing']['sail_generate'] else [], "Journey interrupted"
                    break
                n += 1
                self.images_done += 1

        if query != -1:
            stop_reason = 'Finished' if self.g.job_running else 'Stopped'
            yield self.sail_log, list(images) if self.g.settings_data['sailing']['sail_generate'] else [], f'{stop_reason}\n{self.images_done-1} image(s) done\n{prompt_discard_count} prompts filtered'


    def stop_job(self):
        self.g.job_running = False

    def check_api_avail(self):
        return self.automa_client.check_avail(self.g.settings_data['automa']['automa_url'])

    def count_context(self):
        result = self.interface.count_context()
        return f'{result.count} entries are in the ocean'

    def run_t2t_show_sail(self):
        self.g.job_running = True
        self.g.settings_data['automa']['automa_batch'] = 1
        self.g.settings_data['automa']['automa_n_iter'] = 1
        self.g.sail_history = []
        self.sail_depth_start = self.g.settings_data['sailing']['sail_depth']
        self.sail_sinus_count = 1.0
        self.sail_log = ''
        query = self.g.settings_data['sailing']['sail_text']
        filename = os.path.join(self.out_dir_t2t, f'journey_log_{self.timestamp()}.txt')

        if self.g.settings_data['translate']:
            query = self.interface.translate(query)

        prompt_discard_count = 0
        n = 0
        sail_steps = self.g.settings_data['sailing']['sail_width']
        while n < sail_steps:
            try:
                prompt, orig_prompt, n, prompt_discard_count, sail_steps = self.get_new_prompt(query, n, prompt_discard_count, sail_steps, filename)
                new_nodes = self.interface.direct_search(self.g.settings_data['sailing']['sail_text'], self.g.settings_data['sailing']['sail_depth'], n)

                if self.g.settings_data['sailing']['sail_generate']:
                    response = self.sail_automa_gen(prompt)
                    if response:
                        for index, image in enumerate(response.get('images')):
                            img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                            save_path = os.path.join(self.out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                            self.automa_client.decode_and_save_base64(image, save_path)
                            yield prompt, img
                    else:
                        yield prompt, None
                else:
                    yield prompt, None

                query = self.get_next_target_new(new_nodes)
                if query == -1:
                    self.interface.log_raw(filename, f'{n} sail is finished early due to rotating context')
                    break

            except Exception as e:
                new_nodes = self.interface.direct_search(self.g.settings_data['sailing']['sail_text'], self.g.settings_data['sailing']['sail_depth'], n)
                query = self.get_next_target_new(new_nodes)
                print('some error happened: ', str(e))
                time.sleep(5)
            finally:
                if not self.g.job_running:
                    break
                n += 1
