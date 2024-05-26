from settings.io import settings_io
import gradio as gr

class prompt_iterator:

	def __init__(self):
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

		self.test_data = {
			'Character': [],
			'Air Creatures': [],
			'Land Cratures': [],
			'Sea Creatures': [],
			'Character Objects': [],
			'Character Adjectives': [],
			'Air Vehicle': [],
			'Land Vehicle': [],
			'Sea Vehicle': [],
			'Space Vehicle': [],
			'Moving relation': [],
			'Still relation': [],
			'Object Adjectives': [],
			'Visual Adjectives': [],
			'Visual Qualities': [],
			'Setup': [],
			'Colors': [],
			'Styles': [],
			'Artists': []}

		print('ok')

	def get_sample(self, model_test_list,
				   model_test_character,
				   model_test_creature_air,
				   model_test_creature_land,
				   model_test_creature_sea,
				   model_test_character_objects,
				   model_test_character_adj,
				   model_test_vehicles_air,
				   model_test_vehicles_land,
				   model_test_vehicles_sea,
				   model_test_vehicles_space,
				   model_test_moving_relation,
				   model_test_still_relation,
				   model_test_object_adj,
				   model_test_visual_adj,
				   model_test_visual_qualities,
				   model_test_settings,
				   model_test_colors,
				   model_test_styles,
				   model_test_artists):
		test_data = {
			'Character': model_test_character,
			'Air Creatures': model_test_creature_air,
			'Land Cratures': model_test_creature_land,
			'Sea Creatures': model_test_creature_sea,
			'Character Objects': model_test_character_objects,
			'Character Adjectives': model_test_character_adj,
			'Air Vehicle': model_test_vehicles_air,
			'Land Vehicle': model_test_vehicles_land,
			'Sea Vehicle': model_test_vehicles_sea,
			'Space Vehicle': model_test_vehicles_space,
			'Moving relation': model_test_moving_relation,
			'Still relation': model_test_still_relation,
			'Object Adjectives': model_test_object_adj,
			'Visual Adjectives': model_test_visual_adj,
			'Visual Qualities': model_test_visual_qualities,
			'Setup': model_test_settings,
			'Colors': model_test_colors,
			'Styles': model_test_styles,
			'Artists': model_test_artists}

		test_output = ''
		for entry in model_test_list:
			test_output = f'{test_output} {test_data[entry][0]}'

		return test_output



	def dropdown(self, choices, label, initial_value=None):
		"""
		Custom component with dropdown and "Select All" checkbox with event listener.

		Args:
			choices (list): List of options for the dropdown.
			initial_value (list, optional): Initial selection. Defaults to None.

		Returns:
			tuple: Tuple containing selected options and a boolean for "Select All" state.
		"""
		with gr.Row():
			with gr.Column(scale=1):
				is_all_selected = gr.Checkbox(label="Select All", value=False)
			with gr.Column(scale=3):
				dropdown = gr.Dropdown(label=label,choices=choices, value=initial_value, multiselect=True,allow_custom_value=True)

		self.test_data[label] = dropdown.value


		def update_dropdown(is_all_selected_value):
			return gr.update(choices=choices, value=choices.copy() if is_all_selected_value else [])


		# Trigger update on checkbox change using event listener
		gr.on(
			triggers=[is_all_selected.change],
			fn=update_dropdown,
			inputs=is_all_selected,
			outputs=[dropdown])

		dropdown.interactive = True

		return dropdown

