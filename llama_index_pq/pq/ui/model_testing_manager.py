# model_testing_manager.py
import globals
import gradio as gr
from collections import deque
from PIL import Image
from io import BytesIO
import time
import os
import shared
from post_process.summary import extractive_summary
from generators.automatics.client import automa_client
from settings.io import settings_io
from prompt_iteration import prompt_iterator
from llm_fw import llm_interface_qdrant

class ModelTestingManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.settings_io = settings_io()
        self.interface = llm_interface_qdrant.get_interface()
        self.automa_client = automa_client()
        self.prompt_iterator = prompt_iterator()
        self.out_dir_t2i = os.path.join('../api_out', 'txt2img')
        os.makedirs(self.out_dir_t2i, exist_ok=True)

    #to be shared later
    def stop_job(self):
        self.g.job_running = False


    # Wrapper for prompt_iterator methods
    def setting_dropdown(self, choices, label, initial_value):
        return gr.Dropdown(label=label, choices=choices, value=initial_value, interactive=True)

    def data_dropdown(self, data, label, initial_value):
        return gr.Dropdown(label=label, choices=list(data.keys()), value=initial_value, interactive=True)

    def save_test_data(self, model_test_list, model_test_inst_prompt, model_test_type, model_test_steps,
                       model_test_dimensions, model_test_gen_type, model_test_cfg):
        self.g.settings_data['model_test']['model_test_list'] = model_test_list
        self.g.settings_data['prompt_templates']['model_test_instruction'] = model_test_inst_prompt
        self.g.settings_data['model_test']['model_test_type'] = model_test_type
        self.g.settings_data['model_test']['model_test_steps'] = model_test_steps
        self.g.settings_data['model_test']['model_test_dimensions'] = model_test_dimensions
        self.g.settings_data['model_test']['model_test_gen_type'] = model_test_gen_type
        self.g.settings_data['model_test']['model_test_cfg'] = model_test_cfg
        self.settings_io.write_settings(self.g.settings_data)

    def get_sample(self):
        sample = self.prompt_iterator.get_sample()
        return sample, "Sample generated"

    def get_all_samples(self):
        samples = self.prompt_iterator.get_all_samples()
        return "\n".join(samples), f"{len(samples)} samples generated"

    # Moved from ui_actions
    def run_test(self):
        self.g.job_running = True
        filename = os.path.join('../api_out/txt2txt', f'model_test_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')
        images = deque(maxlen=int(self.g.settings_data['sailing']['sail_max_gallery_size']))

        yield [], 'Preparing the test data'

        combinations = self.prompt_iterator.get_combinations()
        n = 0
        automa_steps = self.g.settings_data['automa']["automa_steps"]
        automa_width = self.g.settings_data['automa']["automa_width"]
        automa_height = self.g.settings_data['automa']["automa_height"]
        automa_CFG = self.g.settings_data['automa']["automa_cfg_scale"]

        if len(combinations) > 0:
            yield [], 'Test data ready, start image generation'
            self.images_done = 0
            for test_query in combinations:
                self.g.settings_data['automa']["automa_steps"] = test_query[0][0]
                self.g.settings_data['automa']["automa_width"] = test_query[0][1].split(',')[0]
                self.g.settings_data['automa']["automa_height"] = test_query[0][1].split(',')[1]
                self.g.settings_data['automa']["automa_cfg_scale"] = test_query[0][2]

                query = self.prepare_query(test_query[1])
                prompt = self.interface.retrieve_model_test_llm_completion(query)

                self.interface.log_raw(filename, f'Gen params: CFG = {test_query[0][2]} Steps = {test_query[0][0]} Width = {test_query[0][1].split(",")[0]} Height = {test_query[0][1].split(",")[1]} \n{prompt}\n{n} ----------')

                images = self.automa_gen(prompt, images, folder=f'{test_query[0][2]}_{test_query[0][0]}_{test_query[0][1][0]}_{test_query[0][1][1]}')
                self.images_done += 1
                yield list(images), f'{self.images_done} image(s) done'
                if not self.g.job_running:
                    yield list(images), f'Stopped.\n{self.images_done} image(s) done'
                    break
                n += 1
        else:
            yield [], 'Nothing to do'

        self.g.settings_data['automa']["automa_steps"] = automa_steps
        self.g.settings_data['automa']["automa_width"] = automa_width
        self.g.settings_data['automa']["automa_height"] = automa_height
        self.g.settings_data['automa']["automa_cfg_scale"] = automa_CFG

    # Helpers needed for run_test (copied from ui_actions)
    def prepare_query(self, query):
        if self.g.settings_data['sailing']['sail_add_search']:
            query = f'{self.g.settings_data["sailing"]["sail_search"]}, {query}'
        if len(query) > 1000:
            query = extractive_summary(query, num_sentences=2)
            if len(query) > 1000:
                query = self.shorten_string(query)
        return query

    def automa_gen(self, prompt, images, folder=None):
        response = self.automa_client.request_generation(prompt, self.g.settings_data['negative_prompt'], self.g.settings_data)
        if response != '':
            for index, image in enumerate(response.get('images')):
                img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                if folder is None:
                    save_path = os.path.join(self.out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                else:
                    save_path = os.path.join(self.out_dir_t2i, folder)
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f'txt2img-{self.timestamp()}-{index}.png')
                self.automa_client.decode_and_save_base64(image, save_path)
                images.append(img)
        return images

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