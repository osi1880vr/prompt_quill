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

import json
import time

import requests
import requests
from PIL import Image
from io import BytesIO


civitai_host = 'https://sdk.civitai.com'
headers = {'Content-Type' : 'application/json',
		   'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
		   'Referer' : 'https://sdk.civitai.com/',
		   'Origin' :  'https://sdk.civitai.com'
		   }
class civitai_client:

	def get_generation_dict(self,air="", prompt="", negative_prompt="", steps=20, cfg=7, width=512, heigth=512, clipskip=2):
		return {
			"model": air,
			"prompt": prompt,
			"negativePrompt": negative_prompt,
			"scheduler": 'EulerA',
			"steps": steps,
			"cfgScale": cfg,
			"width": width,
			"height": heigth,
			"clipSkip": clipskip,
			"additionalNetworks": [
				{
					"model": "",
					"type": "Lora",
					"strength": None,
					"triggerWord": ""
				}
			]
		}

	def wait_for_image(self, token):
		while 1:
			image_url = self.poll_status(token)
			if image_url != -1:


				# Fetch the image data
				response = requests.get(image_url)

				# Open the image data as an RGB PIL image
				img = Image.open(BytesIO(response.content)).convert('RGB')
				return img

			else:
				break


	def poll_status(self, token):

		url = civitai_host + '/api/poll'+f'/{token}'
		image_url = ''
		while 1:

			response = requests.get(url,headers=headers)

			if response.status_code == 200:
				res = response.json()
				if res['jobs'][0]['result']['available'] is True:
					image_url = res['jobs'][0]['result']['blobUrl']
					break
				if res['jobs'][0]['scheduled'] is False:
					image_url = -1
					break
			else:
				image_url = -1
				break
			time.sleep(1)


		return image_url

	def request_generation(self, air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip):

		url = civitai_host + '/api/generate'

		req_dict = self.get_generation_dict(air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip)

		response = requests.post(url, data = json.dumps(req_dict),headers=headers)

		if response.status_code == 200:
			return self.wait_for_image(response.json()['token'])
		else:
			return -1






