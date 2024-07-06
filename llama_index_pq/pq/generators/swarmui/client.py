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
import urllib.parse
import base64
import json
import time
import os



webui_server_url = 'http://localhost:7801'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)



class swarm_client:

	def __init__(self):
		self.webui_server_url = webui_server_url

	def timestamp(self):
		return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

	def convert_byte_array_to_base64(self, byte_array):
		# Encode the byte array to a Base64 string
		base64_string = base64.b64encode(byte_array).decode('utf-8')
		return base64_string


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
			method='POST'
		)
		try:
			response = urllib.request.urlopen(request)
			return json.loads(response.read().decode('utf-8'))
		except Exception as e:
			print(e)
			return ''



	def call_txt2img_api(self,**payload):
		response = self.call_api('API/GenerateText2Image', **payload)
		if response != '':
			return response
		else:
			return ''


	def get_session_id(self):
		response = self.call_api('API/GetNewSession',**{})
		return response['session_id']

	def download_image(self, image):

		with urllib.request.urlopen(f'{self.webui_server_url}/{urllib.parse.quote(image)}') as response:
			file_content = response.read()  # This will hold the file's content as bytes
			return self.convert_byte_array_to_base64(file_content)

	def download_images(self, images):
		img_array = []
		for image in images:
			img = self.download_image(image)
			img_array.append(img)

		return img_array

	def request_generation(self,prompt, negative_prompt,settings_data):

		session_id = self.get_session_id()

		self.webui_server_url=settings_data["swarm_url"]

	
		if type(prompt) == str:
			prompt = (str(prompt).encode('utf-8')).decode('utf-8')
		elif type(prompt) == bytes:
			prompt = prompt.decode('utf-8')
	
		if type(negative_prompt) == str:
			negative_prompt = (str(negative_prompt).encode('utf-8')).decode('utf-8')
		elif type(negative_prompt) == bytes:
			negative_prompt = negative_prompt.decode('utf-8')


		payload = {
			"prompt": prompt,
			"negativeprompt": negative_prompt,
			"controlnetpreviewonly": False,
			"debugregionalprompting": False,
			"model": settings_data['swarm_checkpoint'],
			"images": settings_data["swarm_n_iter"],
			"seed": "-1",
			"steps": settings_data["swarm_steps"],
			"cfgscale": settings_data["swarm_cfg_scale"],
			"aspectratio": "1:1",
			"width": settings_data["swarm_width"],
			"height": settings_data["swarm_height"],
			"sampler": settings_data['swarm_sampler'],
			"scheduler": settings_data['swarm_scheduler'],
			"seamlesstileable": "false",
			"batchsize": settings_data["swarm_batch"],
			"saveintermediateimages": False,
			"donotsave": False,
			"nopreviews": False,
			"internalbackendtype": "Any",
			"noseedincrement": False,
			"personalnote": "",
			"modelspecificenhancements": True,
			"regionalobjectcleanupfactor": "0",
			"savesegmentmask": False,
			"gligenmodel": "None",
			"cascadelatentcompression": "32",
			"removebackground": False,
			"shiftedlatentaverageinit": False,
			"automaticvae": True,
			"presets": [],
			"session_id": session_id
		}


		response = self.call_txt2img_api(**payload)


		img_array = []
		if 'images' in response:
			img_array = self.download_images(response['images'])

		output = {
			'images': img_array,
			'parameters': payload
		}

		return output
	
