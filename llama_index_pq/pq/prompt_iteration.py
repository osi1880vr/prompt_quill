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

	def get_combinations(self):

		work_list = []
		gen_list_out = []
		test_data = self.get_test_data()

		gen_list = [self.g.settings_data['model_test_steps'] if len(self.g.settings_data['model_test_steps']) > 0 else [25],
					self.g.settings_data['model_test_dimensions'] if len(self.g.settings_data['model_test_dimensions']) > 0 else [[1024,1024]],
					self.g.settings_data['model_test_cfg'] if len(self.g.settings_data['model_test_cfg']) > 0 else [7],
					]
		# remove empty arrays
		gen_list = [sub_array for sub_array in gen_list if sub_array]

		if self.g.settings_data['model_test_gen_type'] == 'Largest List':
			gen_list_out = self.combine_limited_arrays(gen_list)
		elif self.g.settings_data['model_test_gen_type'] == 'Full Run':
			gen_list_out = self.combine_all_arrays_to_arrays(gen_list)


		if self.g.settings_data['model_test_list'] is not None and len(self.g.settings_data['model_test_list']) > 0:
			for entry in self.g.settings_data['model_test_list']:
				if entry == 'Artists':
					artist_array = list(map(lambda s: 'in the style of ' + s, test_data[entry]))
					work_list.append(artist_array)
				else:
					work_list.append(test_data[entry])

			# remove empty arrays
			work_list = [sub_array for sub_array in work_list if sub_array]

			if self.g.settings_data['model_test_type'] == 'Largest List':
				work_list = self.combine_limited(work_list)
			elif self.g.settings_data['model_test_type'] == 'Full Run':
				work_list = self.combine_all_arrays_to_strings(work_list)


		gen_job = self.combine_limited_arrays([gen_list_out,work_list])

		return gen_job

	def convert_to_strings(self, array_of_arrays):
		string_array = []
		for sub_array in array_of_arrays:
			# Join elements of the sub-array with a space separator
			string_array.append(" ".join(str(element) for element in sub_array))
		return string_array

	def get_all_samples(self):
		yield '', 'Preparing the test data'
		combinations = self.get_combinations()
		combinations = self.convert_to_strings(combinations)
		yield "\n".join(combinations), len(combinations)


	def save_test_data(self,
					   model_test_list,
					   model_test_inst_prompt,
					   model_test_type,
					   model_test_steps,
					   model_test_dimensions,
					   model_test_gen_type,
					   model_test_cfg):
		self.g.settings_data['model_test_list'] = model_test_list
		self.g.settings_data['model_test_type'] = model_test_type
		self.g.settings_data['model_test_steps'] = model_test_steps
		self.g.settings_data['model_test_dimensions'] = model_test_dimensions
		self.g.settings_data['model_test_gen_type'] = model_test_gen_type
		self.g.settings_data['model_test_cfg'] = model_test_cfg
		self.g.settings_data['prompt_templates']['model_test_instruction'] = model_test_inst_prompt
		settings_io().write_settings(self.g.settings_data)


	def data_dropdown(self, choices, label, initial_value=None):
		with gr.Row():
			with gr.Column(scale=1):
				is_all_selected = gr.Checkbox(label="Select All", value=False)
			with gr.Column(scale=3):
				dropdown = gr.Dropdown(label=label, choices=choices, value=initial_value, multiselect=True,
									   allow_custom_value=True)


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


	def setting_dropdown(self, choices, label, initial_value=None):
		with gr.Row():
			with gr.Column(scale=1):
				is_all_selected = gr.Checkbox(label="Select All", value=False)
			with gr.Column(scale=3):
				dropdown = gr.Dropdown(label=label, choices=choices, value=initial_value, multiselect=True,
									   allow_custom_value=True)


		def select_all_dropdown(is_all_selected_value):
			self.g.settings_data[label] = choices
			settings_io().write_settings(self.g.settings_data)
			return gr.update(choices=choices, value=choices.copy() if is_all_selected_value else [])


		def update_dropdown(dropdown):
			self.g.settings_data[label] = dropdown
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


	def combine_all_arrays_to_arrays(self, data):
		string_combinations = []
		for element in product(*data):
			string_combinations.append(element)
		return string_combinations

	def combine_all_arrays_to_strings(self, data):
		string_combinations = []
		for element in product(*data):
			# Join elements with a separator to form a string
			combination_string = " ".join(element)
			string_combinations.append(combination_string)
		return string_combinations


	def combine_limited_arrays(self, data):

		longest_array = max(data, key=len)
		formatted_lines = []

		# Iterate for the length of the longest array
		for i in range(len(longest_array)):
			line = []
			# Loop through each sub-array
			for arr in data:
				# Calculate the effective index for the current sub-array
				index = i % len(arr)
				# Add element if it exists, otherwise append an empty list for padding
				line.append(arr[index] if index < len(arr) else [])
			# No need to remove trailing space for arrays
			formatted_lines.append(line)

		return formatted_lines

	def combine_limited(self, data):

		longest_array = max(data, key=len)
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
