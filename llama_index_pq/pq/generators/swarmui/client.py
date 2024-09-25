# Copyright 2024 osiworx

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
import requests
import base64
import json
import time
import os



class swarmui_client:

    def __init__(self, url="http://localhost:7801"):
        self.webui_server_url = url
        self.session_id = None
        self.get_session()

    def get_session(self):

        try:
            endpoint = '/API/GetNewSession'
            url = self.webui_server_url + endpoint
            data = {}
            response = requests.post(url, json=data)
            res_json = response.json()
            self.session_id = res_json['session_id']
        except Exception as e:
            self.session_id = None



    def get_checkpoints(self):
        endpoint = '/API/ListModels'
        url = self.webui_server_url + endpoint
        data = {"session_id": self.session_id,
                "path": "",
                "depth": 1}
        response = requests.post(url, json=data)
        res_json = response.json()
        model_list = []
        for model in res_json['files']:
            model_list.append(model['name'])

        return model_list


    def get_loras(self):
        endpoint = '/API/ListModels'
        url = self.webui_server_url + endpoint
        data = {"session_id": self.session_id,
                "path": "",
                "subtype": "LoRA",
                "depth": 3}
        response = requests.post(url, json=data)
        res_json = response.json()
        model_list = []
        for model in res_json['files']:
            model_list.append(model['name'])

        return model_list

    def get_embeddings(self):
        endpoint = '/API/ListModels'
        url = self.webui_server_url + endpoint
        data = {"session_id": self.session_id,
                "path": "",
                "subtype": "Embedding",
                "depth": 3}
        response = requests.post(url, json=data)
        res_json = response.json()
        model_list = []
        for model in res_json['files']:
            model_list.append(model['name'])

        return model_list

    def get_wildcards(self):
        endpoint = '/API/ListModels'
        url = self.webui_server_url + endpoint
        data = {"session_id": self.session_id,
                "path": "",
                "subtype": "Wildcards",
                "depth": 3}
        response = requests.post(url, json=data)
        res_json = response.json()
        model_list = []
        for model in res_json['files']:
            model_list.append(model['name'])

        return model_list

    def get_vaes(self):
        endpoint = '/API/ListModels'
        url = self.webui_server_url + endpoint
        data = {"session_id": self.session_id,
                "path": "",
                "subtype": "VAE",
                "depth": 3}
        response = requests.post(url, json=data)
        res_json = response.json()
        model_list = []
        for model in res_json['files']:
            model_list.append(model['name'])

        return model_list

    def check_avail(self):
        check = self.extract_from_t2i_params("Scheduler")
        if check:
            return 'API OK'
        else:
            return 'API NOT OK'

    def get_image(self, data_json):
        endpoint = '/API/GenerateText2Image'
        url = self.webui_server_url + endpoint
        response = requests.post(url, json=data_json)
        res_json = response.json()

        return res_json['images']

    def extract_from_t2i_params(self, name):
        endpoint = '/API/ListT2IParams'
        url = self.webui_server_url + endpoint
        data = {"session_id": self.session_id}
        response = requests.post(url, json=data)
        res_json = response.json()

        out_array = None
        for item in res_json['list']:
            if item['name'] == name:
                out_array = item["values"]
                break
        return out_array

    def get_scheduler(self):
        return self.extract_from_t2i_params("Scheduler")

    def get_samplers(self):
        return self.extract_from_t2i_params("Sampler")


    def request_generation(self, query, negative_prompt, settings_data):
        endpoint = '/API/GenerateText2Image'
        url = self.webui_server_url + endpoint
        clipstop = settings_data["swarmui_clip_skip"] * -1
        if clipstop > -1:
            clipstop = -1

        data = {
            "session_id": self.session_id,
            "images": 1,
            "donotsave": True,
            "prompt": query,
            "negativeprompt": negative_prompt,
            "model": settings_data["swarmui_checkpoint"],
            "width": settings_data["swarmui_width"],
            "height": settings_data["swarmui_height"],
            "cfgscale": settings_data["swarmui_cfg_scale"],
            "steps": settings_data["swarmui_steps"],
            "sampler": settings_data['swarmui_sampler'],
            "seed": -1,
            "clipstopatlayer": clipstop,
            "modelspecificenhancements": True,
            "colordepth": "8bit",
            "imageformat": "PNG",
            "aspectratio": "1:1",
            "automaticvae": True,
            "regionalobjectcleanupfactor": "0",
            "maskcompositeunthresholded": False,
            "savesegmentmask": False,
            "gligenmodel": "None",
            "cascadelatentcompression": "32",
            "removebackground": False,
            "shiftedlatentaverageinit": False,
            "internalbackendtype": "Any",
            "noseedincrement": False,
            "personalnote": "",
            "scheduler": "beta",
            "zeronegative": False,
            "seamlesstileable": "false",
            "batchsize": "1",
            "saveintermediateimages": False,
            "nopreviews": True
        }

        response = requests.post(url, json=data)
        res_json = response.json()

        return res_json


    def decode_and_save_base64(self,base64_str, save_path):
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(base64_str.split(",")[1]))