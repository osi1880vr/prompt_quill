import globals
from generators.automatics.client import automa_client

class Telegram:

	def __init__(self):
		self.interface = None
		self.automa_client = automa_client()
		self.g = globals.get_globals()


	def automa_gen(self, query, negative_prompt):
		return self.automa_client.request_generation(query,
													 negative_prompt,
													 self.g.settings_data)


	def get_image(self, data):


		if not data['reuse_prompt'] or data['style'] != '':


			if data['style'] != '':
				data['query'] = data['style_prompt'].format(prompt=data['query'])

			try:
				prompt = self.interface.run_api_llm_response(query=data['query'], api=True)
			except Exception as e:
				print(e)
		else:
			prompt = {'prompt':data['query']}

		if data['image_type'] == 'transparent':
			self.g.settings_data['automa']['automa_layerdiffuse_enable'] = True
			self.g.settings_data['automa']["automa_width"] = 1024
			self.g.settings_data['automa']["automa_height"] = 1024
		else:
			self.g.settings_data['automa']['automa_layerdiffuse_enable'] = False
			self.g.settings_data['automa']["automa_width"] = data['w']
			self.g.settings_data['automa']["automa_height"] = data['h']

		self.g.settings_data['automa']["automa_checkpoint"] = data['model']
		self.g.settings_data['automa']["automa_steps"] = data['steps']
		self.g.settings_data['automa']["automa_cfg_scale"] = data['cfg']


		image = self.automa_gen(prompt['prompt'], self.g.settings_data['negative_prompt'] if data['style'] == '' else data['style_neg_prompt'])


		return {"prompt": prompt['prompt'],
				"negative_prompt": self.g.settings_data['negative_prompt'] if data['style'] == '' else data['style_neg_prompt'],
				"image": image}


	def get_prompt(self, data):

		if data['style'] != '':
			data['query'] = data['style_prompt'].format(prompt=data['query'])
		prompt = self.interface.run_api_llm_response(data['query'], True)
		return {"prompt": prompt['prompt'],
				"negative_prompt": self.g.settings_data['negative_prompt'] if data['style'] == '' else data['style_neg_prompt']}




