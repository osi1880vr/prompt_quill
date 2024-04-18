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


    def request_generation(self,prompt, negative_prompt,
                           sampler, steps, cfg, width, heigth, url, save, batch, n_iter,save_api):

        self.webui_server_url=url
        self.save = save

        payload = {
            "prompt": prompt,  # extra networks also in prompts
            "negative_prompt": negative_prompt,
            "seed": -1,
            "steps": steps,
            "width": width,
            "height": heigth,
            "cfg_scale": cfg,
            "sampler_name": sampler,
            "n_iter": n_iter,
            "batch_size": batch,
            "save_images":save_api,
            #"override_settings": {
            # "sd_model_checkpoint": "v1-5-pruned-emaonly.safetensors",
            # "sd_vae": "sd-vae-ft-mse.safetensors"
            #
            # },
            #"override_settings_restore_afterwards": true,
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


    def check_avail(self, url):
        self.webui_server_url = url
        vaes = self.get_api_endpoint('sdapi/v1/sd-vae')
        if vaes != '':
            return 'API OK'
        else:
            return 'API NOT OK'