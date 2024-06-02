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

from datetime import datetime
import urllib.request
import base64
import json
import time
import os




webui_server_url = 'http://192.168.0.127:7860'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
#os.makedirs(out_dir_i2i, exist_ok=True)



class automa_client:

    def __init__(self):
        self.webui_server_url = 'http://localhost:7860'
    def timestamp(self):
        return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


    def encode_file_to_base64(self,path):
        with open(path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')


    def decode_and_save_base64(self,base64_str, save_path):
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str))


    def call_api(self,api_endpoint, **payload):
        data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            f'{self.webui_server_url}/{api_endpoint}',
            headers={'Content-Type': 'application/json'},
            data=data,
        )
        try:
            response = urllib.request.urlopen(request)
            return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(e)
            return ''

    def call_txt2img_api(self,**payload):
        response = self.call_api('sdapi/v1/txt2img', **payload)
        if response != '':
            return response
        else:
            return ''


    def call_img2img_api(self,**payload):
        response = self.call_api('sdapi/v1/img2img', **payload)
        for index, image in enumerate(response.get('images')):
            save_path = os.path.join(out_dir_i2i, f'img2img-{self.timestamp()}-{index}.png')
            self.decode_and_save_base64(image, save_path)


    def call_interrogation_api(self,**payload):
        response = self.call_api('sdapi/v1/interrogate', **payload)
        return response['caption']


    def request_generation(self,prompt, negative_prompt,settings_data):


        self.webui_server_url=settings_data["automa_url"]
        self.save = settings_data["automa_save"]


        ADetailer = {}
        alwayson_scripts = {}
        if settings_data['automa_adetailer_enable']:
            ADetailer['args'] = [{
                'ad_model': settings_data['automa_ad_model'],
                'ad_use_inpaint_width_height': settings_data['automa_ad_use_inpaint_width_height'],
                'ad_denoising_strength': settings_data['automa_ad_denoising_strength'],
                "ad_clip_skip": settings_data['automa_ad_clip_skip'],
                "ad_confidence": settings_data['automa_ad_confidence'],
            }]

        if len(ADetailer) > 0:
            alwayson_scripts["ADetailer"] = ADetailer

        if type(prompt) == str:
            prompt = (str(prompt).encode('utf-8')).decode('utf-8')
        elif type(prompt) == bytes:
            prompt = prompt.decode('utf-8')

        if type(negative_prompt) == str:
            negative_prompt = (str(negative_prompt).encode('utf-8')).decode('utf-8')
        elif type(negative_prompt) == bytes:
            negative_prompt = negative_prompt.decode('utf-8')




        override_settings = {
            "sd_model_checkpoint": settings_data['automa_Checkpoint'],
        }

        if settings_data['automa_vae'] != '':
            override_settings["sd_vae"] = settings_data['automa_vae']

        if settings_data['automa_clip_skip'] > 0:
            override_settings["CLIP_stop_at_last_layers"] = settings_data['automa_clip_skip']

        payload = {
            "alwayson_scripts": alwayson_scripts,
            "prompt": prompt,  # extra networks also in prompts
            "negative_prompt": negative_prompt,
            "seed": -1,
            "steps": settings_data["automa_Steps"],
            "width": settings_data["automa_Width"],
            "height": settings_data["automa_Height"],
            "cfg_scale": settings_data["automa_CFG Scale"],
            "sampler_name": settings_data['automa_Sampler'],
            "n_iter": settings_data["automa_n_iter"],
            "batch_size": settings_data["automa_batch"],
            "save_images":settings_data["automa_save_on_api_host"],
            "override_settings": override_settings,
            "override_settings_restore_afterwards": True,
        }

        return self.call_txt2img_api(**payload)



    def request_interrogation(self, image,url):
        self.webui_server_url=url
        payload = {
            "image": image,
            "model": "clip"
        }
        return self.call_interrogation_api(**payload)


    def get_api_endpoint(self,api_endpoint):
        request = urllib.request.Request(
            f'{self.webui_server_url}/{api_endpoint}',
            headers={'Content-Type': 'application/json'}
        )

        try:
            response = urllib.request.urlopen(request)
            return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(e)
            return ''


    def get_samplers(self, url):
        self.webui_server_url = url
        samplers = self.get_api_endpoint('sdapi/v1/samplers')
        if samplers != '':
            sampler_array = []
            for sampler in samplers:
                sampler_array.append(sampler['name'])

            return sampler_array
        else:
            return -1

    def get_checkpoints(self, url):
        self.webui_server_url = url
        models = self.get_api_endpoint('sdapi/v1/sd-models')
        if models != '':
            model_array = []
            for model in models:
                model_array.append(model['model_name'])

            return model_array
        else:
            return -1

    def get_vaes(self, url):
        self.webui_server_url = url
        vaes = self.get_api_endpoint('sdapi/v1/sd-vae')
        if vaes != '':
            vae_array = []
            for model in vaes:
                vae_array.append(model['model_name'])

            return vae_array
        else:
            return -1
    def check_avail(self, url):
        self.webui_server_url = url
        vaes = self.get_api_endpoint('sdapi/v1/sd-vae')
        if vaes != '':
            return 'API OK'
        else:
            return 'API NOT OK'