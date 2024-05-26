from settings.io import settings_io
import gradio as gr
import globals
from itertools import product


class prompt_iterator:

	def __init__(self):
		self.g = globals.get_globals()
		self.prompt_data = settings_io().load_prompt_data()

		self.character_adj = self.prompt_data['adjectives']['charadjs']
		self.character_objects = self.prompt_data['characters']['charobjs']
		self.character = self.prompt_data['characters']['chars']
		self.creature_air = self.prompt_data['creatures']['aircreas']
		self.creature_land = self.prompt_data['creatures']['landcreas']
		self.creature_sea = self.prompt_data['creatures']['seacreas']

		self.vehicles_air = self.prompt_data['vehicles']['airvehicles']
		self.vehicles_land = self.prompt_data['vehicles']['landvehicles']
		self.vehicles_sea = self.prompt_data['vehicles']['seavehicles']
		self.vehicles_space = self.prompt_data['vehicles']['spacevehicles']

		self.moving_relation = list(set(
			self.prompt_data['relations']['aircrearels'] + self.prompt_data['relations']['airobjrels'] +
			self.prompt_data['relations']['spaceobjrels']))
		self.still_relation = self.prompt_data['relations']['landrels']

		self.object_adj = self.prompt_data['adjectives']['objadjs']
		self.visual_adj = self.prompt_data['adjectives']['visadjs']
		self.visual_qualities = self.prompt_data['visualmodifiers']['qualities']

		self.settings = self.prompt_data['settings']['allsets']

		self.colors = self.prompt_data['adjectives']['colors']

		self.artists = self.prompt_data['style']['artists']
		self.styles = list(set(self.prompt_data['style']['styles'] + self.prompt_data['visualmodifiers']['styles']))

		self.selectors = ['Visual Qualities',
						  'Visual Adjectives',
						  'Styles',
						  'Colors',
						  'Character',
						  'Air Creatures',
						  'Land Cratures',
						  'Sea Creatures',
						  'Character Objects',
						  'Character Adjectives',
						  'Moving relation',
						  'Still relation',
						  'Air Vehicle',
						  'Land Vehicle',
						  'Sea Vehicle',
						  'Space Vehicle',
						  'Object Adjectives',
						  'Setup',
						  'Artists',
						  ]


		print('ok')


	def get_test_data(self):
		return {
			'Character': self.g.settings_data['model_test_setup']['Character'],
			'Air Creatures': self.g.settings_data['model_test_setup']['Air Creatures'],
			'Land Cratures': self.g.settings_data['model_test_setup']['Land Creatures'],
			'Sea Creatures': self.g.settings_data['model_test_setup']['Sea Creatures'],
			'Character Objects': self.g.settings_data['model_test_setup']['Character Objects'],
			'Character Adjectives': self.g.settings_data['model_test_setup']['Character Adjectives'],
			'Air Vehicle': self.g.settings_data['model_test_setup']['Air Vehicle'],
			'Land Vehicle': self.g.settings_data['model_test_setup']['Land Vehicle'],
			'Sea Vehicle': self.g.settings_data['model_test_setup']['Sea Vehicle'],
			'Space Vehicle': self.g.settings_data['model_test_setup']['Space Vehicle'],
			'Moving relation': self.g.settings_data['model_test_setup']['Moving relation'],
			'Still relation': self.g.settings_data['model_test_setup']['Still relation'],
			'Object Adjectives': self.g.settings_data['model_test_setup']['Object Adjectives'],
			'Visual Adjectives': self.g.settings_data['model_test_setup']['Visual Adjectives'],
			'Visual Qualities': self.g.settings_data['model_test_setup']['Visual Qualities'],
			'Setup': self.g.settings_data['model_test_setup']['Setup'],
			'Colors': self.g.settings_data['model_test_setup']['Colors'],
			'Styles': self.g.settings_data['model_test_setup']['Styles'],
			'Artists': self.g.settings_data['model_test_setup']['Artists']}

	def get_sample(self):

		test_data = self.get_test_data()

		test_output = ''
		for entry in self.g.settings_data['model_test_list']:
			test_output = f'{test_output} {test_data[entry][0]}'

		yield test_output, 'OK'

	def get_all_samples(self):

		test_data = self.get_test_data()

		yield '', 'Preparing the test data'
		work_list = []

		if self.g.settings_data['model_test_list'] is not None and len(self.g.settings_data['model_test_list']) > 0:
			for entry in self.g.settings_data['model_test_list']:
				work_list.append(test_data[entry])

			combinations = self.combine_limited(work_list)
			yield "\n".join(combinations), len(combinations)


	def save_test_data(self,model_test_list,
					   model_test_inst_prompt):
		self.g.settings_data['model_test_list'] = model_test_list
		self.g.settings_data['prompt_templates']['model_test_instruction'] = model_test_inst_prompt
		settings_io().write_settings(self.g.settings_data)

	def dropdown(self, choices, label, initial_value=None):
		with gr.Row():
			with gr.Column(scale=1):
				is_all_selected = gr.Checkbox(label="Select All", value=False)
			with gr.Column(scale=3):
				dropdown = gr.Dropdown(label=label,choices=choices, value=initial_value, multiselect=True,allow_custom_value=True)

		def select_all_dropdown(is_all_selected_value):
			self.g.settings_data['model_test_setup'][label] = choices
			settings_io().write_settings(self.g.settings_data)
			return gr.update(choices=choices, value=choices.copy() if is_all_selected_value else [])

		def update_dropdown(dropdown):
			self.g.settings_data['model_test_setup'][label] = dropdown
			settings_io().write_settings(self.g.settings_data)
			return gr.update(choices=choices, value=dropdown)

		gr.on(
			triggers=[is_all_selected.change],
			fn=select_all_dropdown,
			inputs=[is_all_selected],
			outputs=[dropdown])
		gr.on(
			triggers=[dropdown.change],
			fn=update_dropdown,
			inputs=[dropdown],
			outputs=[dropdown])

		dropdown.interactive = True

		return dropdown

	def combine_all_arrays_to_strings(self, data):
		string_combinations = []
		for element in product(*data):
			# Join elements with a separator to form a string
			combination_string = " ".join(element)
			string_combinations.append(combination_string)
		return string_combinations


	def combine_limited(self, data):

		longest_array  = max(data, key=len)
		formatted_lines = []

		# Iterate for the length of the longest array
		for i in range(len(longest_array)):
			line = ""
			# Loop through each sub-array
			for arr in data:
				# Calculate the effective index for the current sub-array
				index = i % len(arr)
				# Add element if it exists, otherwise pad with a space
				if index < len(arr):
					line += str(arr[index]) + " "
				else:
					line += "  "  # Add two spaces for padding
			# Remove trailing space
			formatted_lines.append(line.rstrip())

		return formatted_lines