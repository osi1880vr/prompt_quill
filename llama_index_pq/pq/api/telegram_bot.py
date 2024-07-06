import globals
from generators.automatics.client import automa_client
from generators.swarmui.client import swarm_client

class Telegram:

	def __init__(self):
		self.interface = None
		self.gen_client = swarm_client()
		self.g = globals.get_globals()


	def image_gen(self, query):
		negative_prompt = self.g.settings_data['negative_prompt']
		return self.gen_client.request_generation(query,
												  negative_prompt,
												  self.g.settings_data)


	def get_image(self, data):


		if not data['reuse_prompt'] or data['artist'] != '' or data['style'] != '':

			if data['artist'] != '':
				data['query'] = f"((artwork by {data['artist']})) {data['query']}"
			if data['style'] != '':
				data['query'] = f"((in the style of {data['style']})) {data['query']} "

			try:
				prompt = self.interface.run_api_llm_response(query=data['query'], api=True)
			except Exception as e:
				print(e)
		else:
			prompt = {'prompt':data['query']}

		if data['image_type'] == 'transparent':
			self.g.settings_data['automa_layerdiffuse_enable'] = True
			self.g.settings_data["automa_width"] = 1024
			self.g.settings_data["automa_height"] = 1024
		else:
			self.g.settings_data['automa_layerdiffuse_enable'] = False
			self.g.settings_data["automa_width"] = data['w']
			self.g.settings_data["automa_height"] = data['h']

		self.g.settings_data["automa_checkpoint"] = data['model']
		self.g.settings_data["automa_steps"] = data['steps']
		self.g.settings_data["automa_cfg_scale"] = data['cfg']

		image = self.image_gen(prompt['prompt'])

		return {"prompt": prompt['prompt'],
				"negative_prompt": self.g.settings_data['negative_prompt'],
				"image": image}


	def get_prompt(self, data):
		if 'artist' in data:
			if data['artist'] != '':
				data['query'] = f"artwork by {data['artist']} {data['query']}"
		if 'style' in data:
			if data['style'] != '':
				data['query'] = f"{data['query']} in the style of {data['style']}"
		prompt = self.interface.run_api_llm_response(data['query'], True)
		return {"prompt": prompt['prompt'],
				"negative_prompt": self.g.settings_data['negative_prompt']}




