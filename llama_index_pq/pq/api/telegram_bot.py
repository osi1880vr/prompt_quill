import globals
from generators.automatics.client import automa_client

class Telegram:

	def __init__(self):
		self.interface = None
		self.automa_client = automa_client()
		self.g = globals.get_globals()


	def automa_gen(self, query):
		negative_prompt = self.g.settings_data['negative_prompt']
		return self.automa_client.request_generation(query,
													 negative_prompt,
													 self.g.settings_data)


	def get_image(self, data):

		prompt = self.interface.run_api_llm_response(data['query'])
		if "layerdiffuse" in data:
			if data['layerdiffuse']:
				self.g.settings_data['automa_layerdiffuse_enable'] = True
				self.g.settings_data["automa_width"] = 1024
				self.g.settings_data["automa_height"] = 1024

		image = self.automa_gen(prompt['prompt'])
		if "layerdiffuse" in data:
			if data['layerdiffuse']:
				self.g.settings_data['automa_layerdiffuse_enable'] = False

		return {"prompt": prompt['prompt'],
				"negative_prompt": self.g.settings_data['negative_prompt'],
				"image": image}



	def get_prompt(self, data):

		prompt = self.interface.run_api_llm_response(data['query'])
		return {"prompt": prompt['prompt'],
				"negative_prompt": self.g.settings_data['negative_prompt']}




