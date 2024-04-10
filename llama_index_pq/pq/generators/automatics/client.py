from datetime import datetime
import urllib.request
import base64
import json
import time
import os
from PIL import Image
from io import BytesIO



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
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))


    def call_txt2img_api(self,**payload):
        response = self.call_api('sdapi/v1/txt2img', **payload)
        for index, image in enumerate(response.get('images')):
            img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
            if self.save:
                save_path = os.path.join(out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                self.decode_and_save_base64(image, save_path)
            return img


    def call_img2img_api(self,**payload):
        response = self.call_api('sdapi/v1/img2img', **payload)
        for index, image in enumerate(response.get('images')):
            save_path = os.path.join(out_dir_i2i, f'img2img-{self.timestamp()}-{index}.png')
            self.decode_and_save_base64(image, save_path)



    def request_generation(self,prompt, negative_prompt,
                           sampler, steps, cfg, width, heigth, url, save):

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
            "n_iter": 1,
            "batch_size": 1,
        }

        return self.call_txt2img_api(**payload)











