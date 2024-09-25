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

import os
import globals
from settings.io import settings_io

g = globals.get_globals()
g.settings_data = settings_io().load_settings()
g.settings_data['automa_checkpoints'] = []  # bad hack for now, this should later be updateable via a button
g.settings_data['automa_samplers'] = []

import gradio as gr

from llm_fw.llama_cpp_hijack import llama_cpp_hijack

hijack = llama_cpp_hijack()

from ui import ui_actions, ui_staff
from generators.aesthetic import score
from style import style

css = style

ui = ui_staff()
ui_code = ui_actions()
image_score = score.aestetic_score()

max_top_k = 50
textboxes = []



with gr.Blocks(css=css, title='Prompt Quill') as pq_ui:
	with gr.Row():
		# Image element (adjust width as needed)
		gr.Image(os.path.join(os.getcwd(), "logo/pq_v_small.jpg"), width="20vw", show_label=False,
				 show_download_button=False, container=False, elem_classes="gr-image", )

		# Title element (adjust font size and styling with CSS if needed)
		gr.Markdown("**Prompt Quill 2.0**", elem_classes="app-title")  # Add unique ID for potential CSS styling

	with gr.Tab("Chat") as chat:
		with gr.Row():
			translate = gr.Checkbox(label="Translate", info="Translate your native language to english?",
									value=g.settings_data['translate'])
			batch = gr.Checkbox(label="Batch", info="Run every entry from the context as a input prompt?",
								value=g.settings_data['batch'])
			summary = gr.Checkbox(label="Summary", info="Create a summary from the LLM prompt?",
								  value=g.settings_data['summary'])
		gr.ChatInterface(
			ui_code.run_llm_response,
			textbox=ui.prompt_input,
			chatbot=gr.Chatbot(height=500, render=False),
			theme="soft",
			retry_btn="🔄  Retry",
			undo_btn="↩️ Undo",
			clear_btn="Clear",
		)
		chat.select(ui_code.set_prompt_input, None, ui.prompt_input)
		translate.change(ui_code.set_translate, translate, None)
		batch.change(ui_code.set_batch, batch, None)
		summary.change(ui_code.set_summary, summary, None)

	with gr.Tab("Sail the data ocean") as sailor:

		with gr.Tab('Sailing using Forge/Auto1111') as sailing:
			with gr.Tab('Main view'):
				with gr.Row():
					with gr.Column(scale=3):
						sail_text = gr.Textbox(g.settings_data['sail_text'],
											   label=f'Start your journey with this search. This will be used all along. Change it during sailing to change course.',
											   placeholder="Where do we set our sails", elem_id='sail-input-text')
						keep_sail_text = gr.Checkbox(label="Use the input during the whole trip",
												 info="if set to true there is no dynamic prompting, only the context is changed by the sailing.",
												 value=g.settings_data['keep_sail_text'])
						with gr.Row():
							sail_width = gr.Slider(1, 10000, step=1, value=g.settings_data['sail_width'],
												   label="Sail steps", info="Choose between 1 and 10000")
							sail_depth = gr.Slider(1, 10000, step=1, value=g.settings_data['sail_depth'],
												   label="Sail distance", info="Choose between 1 and 10000")
							sail_depth_preset = gr.Slider(0, 1000000, step=1, value=g.settings_data['sail_depth_preset'],
												   label="Sail distance preset", info="Choose between 1 and 1000000")

						with gr.Row():
							sail_sinus = gr.Checkbox(label="Add a sinus to the distance",
													 info="This will create a sinus wave based movement along the distance.",
													 value=g.settings_data['sail_sinus'])
							sail_sinus_freq = gr.Slider(0.1, 10, step=0.1, value=g.settings_data['sail_sinus_freq'],
														label="Sinus Frequency", info="Choose between 0.1 and 10")
							sail_sinus_range = gr.Slider(1, 500, step=1, value=g.settings_data['sail_sinus_range'],
														 label="Sinus Multiplier", info="Choose between 1 and 500")
						with gr.Row():
							sail_generate = gr.Checkbox(label="Generate with A1111 / SD Forge",
														info="Do you want to directly generate the images?",
														value=g.settings_data['sail_generate'])

							sail_generate_swarm = gr.Checkbox(label="Generate with SwarmUI",
															  info="Do you want to directly generate the images?",
															  value=g.settings_data['sail_generate_swarm'])

							automa_alt_vae = gr.Dropdown(
								choices=g.settings_data['automa_vaes'],
								value=g.settings_data['automa_alt_vae'],
								label='Alternate VAE for fixing black images')
					with gr.Column(scale=1):
						with gr.Row():
							sail_status = gr.Textbox('', label=f'Status', placeholder="status")
						with gr.Row():
							sail_submit_button = gr.Button('Start your journey')
							sail_stop_button = gr.Button('Interrupt your journey')
							sail_count_button = gr.Button('Count possible results')
							sail_check_connect_button = gr.Button('Check API Available')

						with gr.Row():
							sail_max_gallery_size = gr.Slider(1, 500, step=1,
															  value=g.settings_data['sail_max_gallery_size'],
															  label="Max Gallery size",
															  info="Limit the number of images keept in the gallery choose between 1 and 500")

				with gr.Row():
					sail_result_images = gr.Gallery(label='output images', height=300, rows=1, columns=6, format='png',interactive=True)
				with gr.Row():
					sail_result = gr.Textbox("", label=f'Your journey journal', placeholder="Your journey logs",interactive=True,autoscroll=True)
			with gr.Tab('Filters'):
				with gr.Row():
					with gr.Column(scale=1):
						sail_filter_count_button = gr.Button('Count possible results')
					with gr.Column(scale=3):
						sail_filter_status = gr.Textbox('', label=f'Count', placeholder="0")
				with gr.Row():
					with gr.Column(scale=1):
						sail_filter_prompt = gr.Checkbox(label="Filter on prompt Level?",
														 info="With this you filter entries from the prompt generation. It may lead to long wait time until a prompt will match.",
														 value=g.settings_data['sail_filter_prompt'])
						sail_filter_context = gr.Checkbox(label="Filter on context Level?",
														  info="With this you filter entries from the context prior to prompt generation. It may lead to empty context.",
														  value=g.settings_data['sail_filter_context'])

					with gr.Column(scale=3):
						sail_filter_text = gr.Textbox(g.settings_data['sail_filter_text'],
													  label=f'List of negative words, words that are not allowed to be in context.',
													  placeholder="Comma separated list of words you dont want in your prompt")
						sail_filter_not_text = gr.Textbox(g.settings_data['sail_filter_not_text'],
														  label=f'List of positive words, words that must be in context.',
														  placeholder="Comma separated list of words that must be part of the prompt")
				with gr.Row():
					with gr.Column(scale=1):
						sail_neg_filter_context = gr.Checkbox(label="Filter on negative prompt context Level?",
															  info="With this you filter entries from the context prior to prompt generation. It may lead to empty context.",
															  value=g.settings_data['sail_filter_context'])
					with gr.Column(scale=3):
						sail_neg_filter_text = gr.Textbox(g.settings_data['sail_filter_text'],
														  label=f'List of negative words, words that are not allowed to be in negative prompt context.',
														  placeholder="Comma separated list of words you dont want in your negative prompt ")
						sail_neg_filter_not_text = gr.Textbox(g.settings_data['sail_filter_not_text'],
															  label=f'List of positive words, words that must be in negative prompt context.',
															  placeholder="Comma separated list of words that must be part of the negative prompt ")

				with gr.Row():
					with gr.Column(scale=1):
						sail_add_search = gr.Checkbox(label="Add search specification",
													  info="Add a text to each vector search.",
													  value=g.settings_data['sail_add_search'])
					with gr.Column(scale=3):
						sail_search = gr.Textbox(g.settings_data['sail_search'], label=f'Search Spec',
												 placeholder="Enter your additional search")
			with gr.Tab('Prompt manipulation'):
				with gr.Row():
					sail_dyn_neg = gr.Checkbox(label="Use dynamic Negative Prompt",
											   info="Uses the negative if we find one, or the default. Be warned this can cause black images or other troubles.",
											   value=g.settings_data['sail_dyn_neg'])

				with gr.Row(equal_height=True):
					with gr.Column(scale=1):
						sail_add_neg = gr.Checkbox(label="Add to negative prompt",
												   info="Add a text to each negative prompt",
												   value=g.settings_data['sail_add_neg'])
					with gr.Column(scale=3):
						sail_neg_prompt = gr.Textbox(g.settings_data['sail_neg_prompt'], label=f'Negative Prompt addon',
													 placeholder="Enter your negative prompt addon")
				with gr.Row(equal_height=True):
					with gr.Column(scale=1):
						sail_add_style = gr.Checkbox(label="Add style specification", info="Add a text to each prompt",
													 value=g.settings_data['sail_add_style'])
					with gr.Column(scale=3):
						sail_style = gr.Textbox(g.settings_data['sail_style'], label=f'Style Spec',
												placeholder="Enter your hardcoded style")

				with gr.Row():
					sail_summary = gr.Checkbox(label="Do summary of LLM prompt",
											   info="The prompt will get reduced to a summary",
											   value=g.settings_data['sail_summary'])

				with gr.Row(equal_height=True):
					with gr.Column(scale=1):
						sail_rephrase = gr.Checkbox(label="Rephrase LLM prompt",
													info="The prompt gets rephrased based on the rephrase prompt",
													value=g.settings_data['sail_rephrase'])
						sail_gen_rephrase = gr.Checkbox(label="Generate the input Prompt too",
														info="To see the effect of the rephrasing you can check here to get both prompts generated",
														value=g.settings_data['sail_gen_rephrase'])
					with gr.Column(scale=3):
						sail_rephrase_prompt = gr.Textbox(g.settings_data['sail_rephrase_prompt'],
														  label=f'Rephrase Prompt',
														  placeholder="Enter your rephrase prompt",
														  lines=4)
			with gr.Tab('Generation Sailing') as gen_sail:
				with gr.Row():
					with gr.Column(scale=3):
						with gr.Row():
							with gr.Column():
								sail_gen_enabled = gr.Checkbox(label="Enable generation parameters", info="Enable dynamic generation parameters?",
														  value=g.settings_data['sail_gen_enabled'])
								sail_override_settings_restore = gr.Checkbox(label="Restore overriden Settings after each image", info="If set to true the Checkpoint and VAE will be set to the settings in SD Forge/Auto1111. It will slow down the process, but might heping with black images.",
															   value=g.settings_data['sail_override_settings_restore'])
								sail_store_folders = gr.Checkbox(label="store the images in folders per model?", info="Should the images be stored in different Folders per model?",
															   value=g.settings_data['sail_store_folders'])
							with gr.Column():
								sail_gen_any_combination = gr.Checkbox(label="Generate any combination selected for any single prompt?", info="Should any prompt be generated for any possible combination?",
																 value=g.settings_data['sail_gen_any_combination'])
								sail_gen_type = gr.Radio(['Random', 'Linear'],
														 label='Select type of change, Ranodm or linear after n steps',
														 value=g.settings_data['sail_gen_type'], interactive=True)

								sail_gen_steps = gr.Slider(1, 100, step=1, value=g.settings_data['sail_gen_steps'],
														   label="Steps",
														   info="Rotate after n steps")
						with gr.Row():
							sail_dimensions = ui_code.prompt_iterator.setting_dropdown(label='Dimensions',
														  choices=g.settings_data['model_test_dimensions_list'],
														initial_value=g.settings_data['sail_dimensions'])
						with gr.Row() as automa_model_settings:


							sail_sampler = ui_code.prompt_iterator.setting_dropdown(
								choices=g.settings_data['automa_samplers'],
								initial_value=g.settings_data['sail_sampler'],
								label='Sampler')
							sail_checkpoint = ui_code.prompt_iterator.setting_dropdown(
								choices=g.settings_data['automa_checkpoints'],
								initial_value=g.settings_data['sail_checkpoint'],
								label='Checkpoint')
							sail_vae = ui_code.prompt_iterator.setting_dropdown(
								choices=g.settings_data['automa_vaes'],
								initial_value=g.settings_data['sail_vae'],
								label='VAE')

					with gr.Column(scale=1):
						sail_gen_refresh_button = gr.Button('Refresh')

		with gr.Tab('Sailing using SwarmUi') as sailing_swarmui:
			with gr.Tab('Main view'):
				with gr.Row():
					with gr.Column(scale=3):
						swarmui_sail_text = gr.Textbox(g.settings_data['swarmui_sail_text'],
											   label=f'Start your journey with this search. This will be used all along. Change it during sailing to change course.',
											   placeholder="Where do we set our sails", elem_id='sail-input-text')
						swarmui_keep_sail_text = gr.Checkbox(label="Use the input during the whole trip",
													 info="if set to true there is no dynamic prompting, only the context is changed by the sailing.",
													 value=g.settings_data['swarmui_keep_sail_text'])
						with gr.Row():
							swarmui_sail_width = gr.Slider(1, 10000, step=1, value=g.settings_data['swarmui_sail_width'],
												   label="Sail steps", info="Choose between 1 and 10000")
							swarmui_sail_depth = gr.Slider(1, 10000, step=1, value=g.settings_data['swarmui_sail_depth'],
												   label="Sail distance", info="Choose between 1 and 10000")
							swarmui_sail_depth_preset = gr.Slider(0, 1000000, step=1, value=g.settings_data['swarmui_sail_depth_preset'],
														  label="Sail distance preset", info="Choose between 1 and 1000000")

						with gr.Row():
							swarmui_sail_sinus = gr.Checkbox(label="Add a sinus to the distance",
													 info="This will create a sinus wave based movement along the distance.",
													 value=g.settings_data['swarmui_sail_sinus'])
							swarmui_sail_sinus_freq = gr.Slider(0.1, 10, step=0.1, value=g.settings_data['swarmui_sail_sinus_freq'],
														label="Sinus Frequency", info="Choose between 0.1 and 10")
							swarmui_sail_sinus_range = gr.Slider(1, 500, step=1, value=g.settings_data['swarmui_sail_sinus_range'],
														 label="Sinus Multiplier", info="Choose between 1 and 500")
						with gr.Row():

							swarmui_sail_generate = gr.Checkbox(label="Generate with SwarmUI",
															  info="Do you want to directly generate the images?",
															  value=g.settings_data['swarmui_sail_generate'])

							swarmui_alt_vae = gr.Dropdown(
								choices=g.settings_data['swarmui_vaes'],
								value=g.settings_data['swarmui_alt_vae'],
								label='Alternate VAE for fixing black images')
					with gr.Column(scale=1):
						with gr.Row():
							swarmui_sail_status = gr.Textbox('', label=f'Status', placeholder="status")
						with gr.Row():
							swarmui_sail_submit_button = gr.Button('Start your journey')
							swarmui_sail_stop_button = gr.Button('Interrupt your journey')
							swarmui_sail_count_button = gr.Button('Count possible results')
							swarmui_sail_check_connect_button = gr.Button('Check API Available')

						with gr.Row():
							swarmui_sail_max_gallery_size = gr.Slider(1, 500, step=1,
															  value=g.settings_data['swarmui_sail_max_gallery_size'],
															  label="Max Gallery size",
															  info="Limit the number of images keept in the gallery choose between 1 and 500")

				with gr.Row():
					swarmui_sail_result_images = gr.Gallery(label='output images', height=300, rows=1, columns=6, format='png',interactive=True)
				with gr.Row():
					swarmui_sail_result = gr.Textbox("", label=f'Your journey journal', placeholder="Your journey logs",interactive=True,autoscroll=True)
			with gr.Tab('Filters'):
				with gr.Row():
					with gr.Column(scale=1):
						swarmui_sail_filter_count_button = gr.Button('Count possible results')
					with gr.Column(scale=3):
						swarmui_sail_filter_status = gr.Textbox('', label=f'Count', placeholder="0")
				with gr.Row():
					with gr.Column(scale=1):
						swarmui_sail_filter_prompt = gr.Checkbox(label="Filter on prompt Level?",
														 info="With this you filter entries from the prompt generation. It may lead to long wait time until a prompt will match.",
														 value=g.settings_data['swarmui_sail_filter_prompt'])
						swarmui_sail_filter_context = gr.Checkbox(label="Filter on context Level?",
														  info="With this you filter entries from the context prior to prompt generation. It may lead to empty context.",
														  value=g.settings_data['swarmui_sail_filter_context'])

					with gr.Column(scale=3):
						swarmui_sail_filter_text = gr.Textbox(g.settings_data['swarmui_sail_filter_text'],
													  label=f'List of negative words, words that are not allowed to be in context.',
													  placeholder="Comma separated list of words you dont want in your prompt")
						swarmui_sail_filter_not_text = gr.Textbox(g.settings_data['swarmui_sail_filter_not_text'],
														  label=f'List of positive words, words that must be in context.',
														  placeholder="Comma separated list of words that must be part of the prompt")
				with gr.Row():
					with gr.Column(scale=1):
						swarmui_sail_neg_filter_context = gr.Checkbox(label="Filter on negative prompt context Level?",
															  info="With this you filter entries from the context prior to prompt generation. It may lead to empty context.",
															  value=g.settings_data['swarmui_sail_neg_filter_context'])
					with gr.Column(scale=3):
						swarmui_sail_neg_filter_text = gr.Textbox(g.settings_data['swarmui_sail_neg_filter_text'],
														  label=f'List of negative words, words that are not allowed to be in negative prompt context.',
														  placeholder="Comma separated list of words you dont want in your negative prompt ")
						swarmui_sail_neg_filter_not_text = gr.Textbox(g.settings_data['swarmui_sail_neg_filter_not_text'],
															  label=f'List of positive words, words that must be in negative prompt context.',
															  placeholder="Comma separated list of words that must be part of the negative prompt ")

				with gr.Row():
					with gr.Column(scale=1):
						swarmui_sail_add_search = gr.Checkbox(label="Add search specification",
													  info="Add a text to each vector search.",
													  value=g.settings_data['swarmui_sail_add_search'])
					with gr.Column(scale=3):
						swarmui_sail_search = gr.Textbox(g.settings_data['swarmui_sail_search'], label=f'Search Spec',
												 placeholder="Enter your additional search")
			with gr.Tab('Prompt manipulation'):
				with gr.Row():
					swarmui_sail_dyn_neg = gr.Checkbox(label="Use dynamic Negative Prompt",
											   info="Uses the negative if we find one, or the default. Be warned this can cause black images or other troubles.",
											   value=g.settings_data['swarmui_sail_dyn_neg'])

				with gr.Row(equal_height=True):
					with gr.Column(scale=1):
						swarmui_sail_add_neg = gr.Checkbox(label="Add to negative prompt",
												   info="Add a text to each negative prompt",
												   value=g.settings_data['swarmui_sail_add_neg'])
					with gr.Column(scale=3):
						swarmui_sail_neg_prompt = gr.Textbox(g.settings_data['swarmui_sail_neg_prompt'], label=f'Negative Prompt addon',
													 placeholder="Enter your negative prompt addon")
				with gr.Row(equal_height=True):
					with gr.Column(scale=1):
						swarmui_sail_add_style = gr.Checkbox(label="Add style specification", info="Add a text to each prompt",
													 value=g.settings_data['swarmui_sail_add_style'])
					with gr.Column(scale=3):
						swarmui_sail_style = gr.Textbox(g.settings_data['swarmui_sail_style'], label=f'Style Spec',
												placeholder="Enter your hardcoded style")

				with gr.Row():
					swarmui_sail_summary = gr.Checkbox(label="Do summary of LLM prompt",
											   info="The prompt will get reduced to a summary",
											   value=g.settings_data['swarmui_sail_summary'])

				with gr.Row(equal_height=True):
					with gr.Column(scale=1):
						swarmui_sail_rephrase = gr.Checkbox(label="Rephrase LLM prompt",
													info="The prompt gets rephrased based on the rephrase prompt",
													value=g.settings_data['swarmui_sail_rephrase'])
						swarmui_sail_gen_rephrase = gr.Checkbox(label="Generate the input Prompt too",
														info="To see the effect of the rephrasing you can check here to get both prompts generated",
														value=g.settings_data['swarmui_sail_gen_rephrase'])
					with gr.Column(scale=3):
						swarmui_sail_rephrase_prompt = gr.Textbox(g.settings_data['swarmui_sail_rephrase_prompt'],
														  label=f'Rephrase Prompt',
														  placeholder="Enter your rephrase prompt",
														  lines=4)
			with gr.Tab('Generation Sailing') as swarmui_gen_sail:
				with gr.Row():
					with gr.Column(scale=3):
						with gr.Row():
							with gr.Column():
								swarmui_sail_gen_enabled = gr.Checkbox(label="Enable generation parameters", info="Enable dynamic generation parameters?",
															   value=g.settings_data['swarmui_sail_gen_enabled'])
								swarmui_sail_override_settings_restore = gr.Checkbox(label="Restore overriden Settings after each image", info="If set to true the Checkpoint and VAE will be set to the settings in SD Forge/Auto1111. It will slow down the process, but might heping with black images.",
																			 value=g.settings_data['swarmui_sail_override_settings_restore'])
								swarmui_sail_store_folders = gr.Checkbox(label="store the images in folders per model?", info="Should the images be stored in different Folders per model?",
																 value=g.settings_data['swarmui_sail_store_folders'])
							with gr.Column():
								swarmui_sail_gen_any_combination = gr.Checkbox(label="Generate any combination selected for any single prompt?", info="Should any prompt be generated for any possible combination?",
																	   value=g.settings_data['swarmui_sail_gen_any_combination'])
								swarmui_sail_gen_type = gr.Radio(['Random', 'Linear'],
														 label='Select type of change, Ranodm or linear after n steps',
														 value=g.settings_data['swarmui_sail_gen_type'], interactive=True)

								swarmui_sail_gen_steps = gr.Slider(1, 100, step=1, value=g.settings_data['swarmui_sail_gen_steps'],
														   label="Steps",
														   info="Rotate after n steps")
						with gr.Row():
							swarmui_sail_dimensions = ui_code.prompt_iterator.setting_dropdown(label='Dimensions',
																					   choices=g.settings_data['model_test_dimensions_list'],
																					   initial_value=g.settings_data['swarmui_sail_dimensions'])
						with gr.Row() as swarmui_model_settings:


							swarmui_sail_sampler = ui_code.prompt_iterator.setting_dropdown(
								choices=g.settings_data['swarmui_samplers'],
								initial_value=g.settings_data['swarmui_sail_sampler'],
								label='Sampler')
							swarmui_sail_checkpoint = ui_code.prompt_iterator.setting_dropdown(
								choices=g.settings_data['swarmui_checkpoints'],
								initial_value=g.settings_data['swarmui_sail_checkpoint'],
								label='Checkpoint')
							swarmui_sail_vae = ui_code.prompt_iterator.setting_dropdown(
								choices=g.settings_data['swarmui_vaes'],
								initial_value=g.settings_data['swarmui_sail_vae'],
								label='VAE')

					with gr.Column(scale=1):
						swarmui_sail_gen_refresh_button = gr.Button('Refresh')

		# swarm ui sailing

		gr.on(
			triggers=[swarmui_sail_text.change,
					  swarmui_keep_sail_text.change,
					  swarmui_sail_width.change,
					  swarmui_sail_depth.change,
					  swarmui_sail_generate.change,
					  swarmui_sail_summary.change,
					  swarmui_sail_rephrase.change,
					  swarmui_sail_rephrase_prompt.change,
					  swarmui_sail_gen_rephrase.change,
					  swarmui_sail_sinus.change,
					  swarmui_sail_sinus_freq.change,
					  swarmui_sail_sinus_range.change,
					  swarmui_sail_add_style.change,
					  swarmui_sail_style.change,
					  swarmui_sail_add_search.change,
					  swarmui_sail_search.change,
					  swarmui_sail_max_gallery_size.change,
					  swarmui_sail_dyn_neg.change,
					  swarmui_sail_add_neg.change,
					  swarmui_sail_neg_prompt.change,
					  swarmui_sail_filter_text.change,
					  swarmui_sail_filter_not_text.change,
					  swarmui_sail_filter_context.change,
					  swarmui_sail_filter_prompt.change,
					  swarmui_sail_neg_filter_text.change,
					  swarmui_sail_neg_filter_not_text.change,
					  swarmui_sail_neg_filter_context.change,
					  swarmui_alt_vae.change,
					  swarmui_sail_checkpoint.change,
					  swarmui_sail_sampler.change,
					  swarmui_sail_vae.change,
					  swarmui_sail_dimensions.change,
					  swarmui_sail_gen_type.change,
					  swarmui_sail_gen_any_combination.change,
					  swarmui_sail_gen_steps.change,
					  swarmui_sail_gen_enabled.change,
					  swarmui_sail_override_settings_restore.change,
					  swarmui_sail_store_folders.change,
					  swarmui_sail_depth_preset.change
					  ],
			fn=ui_code.set_swarmui_sailing_settings,
			inputs=[swarmui_sail_text,
					swarmui_keep_sail_text,
					swarmui_sail_width,
					swarmui_sail_depth,
					swarmui_sail_generate,
					swarmui_sail_summary,
					swarmui_sail_rephrase,
					swarmui_sail_rephrase_prompt,
					swarmui_sail_gen_rephrase,
					swarmui_sail_sinus,
					swarmui_sail_sinus_freq,
					swarmui_sail_sinus_range,
					swarmui_sail_add_style,
					swarmui_sail_style,
					swarmui_sail_add_search,
					swarmui_sail_search,
					swarmui_sail_max_gallery_size,
					swarmui_sail_dyn_neg,
					swarmui_sail_add_neg,
					swarmui_sail_neg_prompt,
					swarmui_sail_filter_text,
					swarmui_sail_filter_not_text,
					swarmui_sail_filter_context,
					swarmui_sail_filter_prompt,
					swarmui_sail_neg_filter_text,
					swarmui_sail_neg_filter_not_text,
					swarmui_sail_neg_filter_context,
					swarmui_alt_vae,
					swarmui_sail_checkpoint,
					swarmui_sail_sampler,
					swarmui_sail_vae,
					swarmui_sail_dimensions,
					swarmui_sail_gen_type,
					swarmui_sail_gen_any_combination,
					swarmui_sail_gen_steps,
					swarmui_sail_gen_enabled,
					swarmui_sail_override_settings_restore,
					swarmui_sail_store_folders,
					swarmui_sail_depth_preset
					],
			outputs=None)


		gr.on(
			triggers=[swarmui_gen_sail.select],
			fn=ui_code.swarmui_refresh,
			inputs=None,
			outputs=[swarmui_sail_sampler, swarmui_sail_checkpoint, swarmui_sail_vae]
		)

		# forge sailing
		gr.on(
			triggers=[sail_text.change,
					  keep_sail_text.change,
					  sail_width.change,
					  sail_depth.change,
					  sail_generate.change,
					  sail_summary.change,
					  sail_rephrase.change,
					  sail_rephrase_prompt.change,
					  sail_gen_rephrase.change,
					  sail_sinus.change,
					  sail_sinus_freq.change,
					  sail_sinus_range.change,
					  sail_add_style.change,
					  sail_style.change,
					  sail_add_search.change,
					  sail_search.change,
					  sail_max_gallery_size.change,
					  sail_dyn_neg.change,
					  sail_add_neg.change,
					  sail_neg_prompt.change,
					  sail_filter_text.change,
					  sail_filter_not_text.change,
					  sail_filter_context.change,
					  sail_filter_prompt.change,
					  sail_neg_filter_text.change,
					  sail_neg_filter_not_text.change,
					  sail_neg_filter_context.change,
					  automa_alt_vae.change,
					  sail_checkpoint.change,
					  sail_sampler.change,
					  sail_vae.change,
					  sail_dimensions.change,
					  sail_gen_type.change,
					  sail_gen_any_combination.change,
					  sail_gen_steps.change,
					  sail_gen_enabled.change,
					  sail_override_settings_restore.change,
					  sail_store_folders.change,
					  sail_depth_preset.change
					  ],
			fn=ui_code.set_sailing_settings,
			inputs=[sail_text,
					keep_sail_text,
					sail_width,
					sail_depth,
					sail_generate,
					sail_summary,
					sail_rephrase,
					sail_rephrase_prompt,
					sail_gen_rephrase,
					sail_sinus,
					sail_sinus_freq,
					sail_sinus_range,
					sail_add_style,
					sail_style,
					sail_add_search,
					sail_search,
					sail_max_gallery_size,
					sail_dyn_neg,
					sail_add_neg,
					sail_neg_prompt,
					sail_filter_text,
					sail_filter_not_text,
					sail_filter_context,
					sail_filter_prompt,
					sail_neg_filter_text,
					sail_neg_filter_not_text,
					sail_neg_filter_context,
					automa_alt_vae,
					sail_checkpoint,
					sail_sampler,
					sail_vae,
					sail_dimensions,
					sail_gen_type,
					sail_gen_any_combination,
					sail_gen_steps,
					sail_gen_enabled,
					sail_override_settings_restore,
					sail_store_folders,
					sail_depth_preset
					],
			outputs=None)

		gr.on(
			triggers=[sailor.select],
			fn=ui_code.get_sailing_settings,
			inputs=None,
			outputs=[sail_text,
					 sail_width,
					 sail_depth,
					 sail_generate,
					 sail_summary,
					 sail_rephrase,
					 sail_rephrase_prompt,
					 sail_gen_rephrase,
					 sail_sinus,
					 sail_sinus_freq,
					 sail_sinus_range,
					 sail_add_style,
					 sail_style,
					 sail_add_search,
					 sail_search,
					 sail_max_gallery_size,
					 sail_filter_text,
					 sail_filter_not_text,
					 sail_filter_context,
					 sail_filter_prompt,
					 sail_neg_filter_text,
					 sail_neg_filter_not_text,
					 sail_neg_filter_context,
					 automa_alt_vae,
					 sail_checkpoint,
					 sail_sampler,
					 sail_vae,
					 sail_dimensions,
					 sail_gen_type,
					 sail_gen_steps,
					 sail_gen_enabled,
					 sail_override_settings_restore,
					 sail_store_folders])


		gr.on(
			triggers=[gen_sail.select],
			fn=ui_code.automa_refresh,
			inputs=None,
			outputs=[sail_sampler, sail_checkpoint, sail_vae]
		)

		start_sail = sail_submit_button.click(fn=ui_code.run_t2t_sail,
											  inputs=[],
											  outputs=[sail_result,
													   sail_result_images,
													   sail_status])
		sail_stop_button.click(fn=ui_code.stop_job,
							   inputs=None,
							   outputs=None,
							   cancels=[start_sail])

		swarmui_start_sail = swarmui_sail_submit_button.click(fn=ui_code.run_swarmui_t2t_sail,
											  inputs=[],
											  outputs=[swarmui_sail_result,
													   swarmui_sail_result_images,
													   swarmui_sail_status])
		swarmui_sail_stop_button.click(fn=ui_code.stop_job,
							   inputs=None,
							   outputs=None,
							   cancels=[swarmui_start_sail])

		sail_check_connect_button.click(fn=ui_code.check_api_avail,
										inputs=None,
										outputs=sail_status)

		swarmui_sail_check_connect_button.click(fn=ui_code.swarmui_check_api_avail,
										inputs=None,
										outputs=swarmui_sail_status)


		sail_count_button.click(fn=ui_code.count_context,
								inputs=None,
								outputs=sail_status)

		swarmui_sail_count_button.click(fn=ui_code.count_context,
								inputs=None,
								outputs=swarmui_sail_status)

		sail_filter_count_button.click(fn=ui_code.count_context,
									   inputs=None,
									   outputs=sail_filter_status)

		swarmui_sail_filter_count_button.click(fn=ui_code.count_context,
									   inputs=None,
									   outputs=swarmui_sail_filter_status)

		sail_gen_refresh_button.click(fn=ui_code.automa_refresh,
									inputs=None,
									outputs=[sail_sampler, sail_checkpoint, sail_vae])

		swarmui_sail_gen_refresh_button.click(fn=ui_code.swarmui_refresh,
									  inputs=None,
									  outputs=[swarmui_sail_sampler, swarmui_sail_checkpoint, swarmui_sail_vae])


		with gr.Tab('Show'):
			with gr.Row():
				sail_show_submit_button = gr.Button('Start your journey')
			with gr.Row():
				sail_show_image = gr.Image(height=800, width=1300)
			with gr.Row():
				sail_show_result = gr.Textbox("", label=f'Your journey journal', placeholder="Your journey logs",
											  lines=4)

			start_sail_show = sail_show_submit_button.click(ui_code.run_t2t_show_sail, None,
															[sail_show_result, sail_show_image])

	with gr.Tab("Generator") as generator:

		with gr.Tab("Automatic 1111 / Forge") as automatic1111:
			with gr.Tab('Generate') as generate:
				with gr.Row():
					with gr.Column(scale=3):
						automa_prompt_input = gr.TextArea(g.last_prompt, lines=5, label="Prompt")
						automa_negative_prompt_input = gr.TextArea(g.last_negative_prompt, lines=3,
																   label="Negative Prompt")
						with gr.Row():
							with gr.Column(scale=2):
								automa_sampler = gr.Dropdown(
									choices=g.settings_data['automa_samplers'], value=g.settings_data['automa_sampler'],
									label='Sampler')
							with gr.Column(scale=2):
								automa_checkpoint = gr.Dropdown(
									choices=g.settings_data['automa_checkpoints'],
									value=g.settings_data['automa_checkpoint'],
									label='Checkpoint')
							with gr.Column(scale=2):
								automa_vae = gr.Dropdown(
									choices=g.settings_data['automa_vaes'],
									value=g.settings_data['automa_vae'],
									label='VAE')

						with gr.Row():
							automa_steps = gr.Slider(1, 100, step=1, value=g.settings_data['automa_steps'],
													 label="Steps",
													 info="Choose between 1 and 100")
							automa_CFG = gr.Slider(0, 20, step=0.1, value=g.settings_data['automa_cfg_scale'],
												   label="CFG Scale",
												   info="Choose between 1 and 20")
							automa_clip_skip = gr.Slider(0, 12, step=1, value=g.settings_data['automa_clip_skip'],
														 label="Clip Skip",
														 info="Choose between 1 and 12")
						with gr.Row():
							with gr.Column(scale=3):
								automa_width = gr.Slider(1, 2048, step=1, value=g.settings_data['automa_width'],
														 label="Width",
														 info="Choose between 1 and 2048")
								automa_height = gr.Slider(1, 2048, step=1, value=g.settings_data['automa_height'],
														  label="Height",
														  info="Choose between 1 and 2048")
							with gr.Column(scale=0):
								automa_size_button = gr.Button('Switch size', elem_classes="gr-small-button")
							with gr.Column(scale=1):
								automa_Batch = gr.Slider(1, 250, step=1, value=g.settings_data['automa_batch'],
														 label="Batch",
														 info="The number of simultaneous images in each batch, range from 1-250.")
								automa_n_iter = gr.Slider(1, 500, step=1, value=g.settings_data['automa_n_iter'],
														  label="Iterations",
														  info="The number of sequential batches to be run, range from 1-500.")
						with gr.Row():
							automa_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa_url'])
							automa_save = gr.Checkbox(label="Save", info="Save the image?",
													  value=g.settings_data['automa_save'])
							automa_save_on_api_host = gr.Checkbox(label="Save", info="Save the image on API host?",
																  value=g.settings_data['automa_save_on_api_host'])
					with gr.Column(scale=1):
						automa_refresh_button = gr.Button('Refresh')
						automa_start_button = gr.Button('Generate')
						automa_result_images = gr.Gallery(label='output images', )

				automa_size_button.click(fn=ui_code.automa_switch_size,
										 inputs=[automa_width, automa_height],
										 outputs=[automa_width, automa_height])

				automa_refresh_button.click(fn=ui_code.automa_refresh,
											inputs=None,
											outputs=[automa_sampler, automa_checkpoint, automa_vae])

				automa_start_button.click(fn=ui_code.run_automatics_generation,
										  inputs=[automa_prompt_input,
												  automa_negative_prompt_input,
												  automa_sampler,
												  automa_checkpoint,
												  automa_steps,
												  automa_CFG,
												  automa_width,
												  automa_height,
												  automa_Batch,
												  automa_n_iter,
												  automa_url,
												  automa_save,
												  automa_save_on_api_host,
												  automa_vae,
												  automa_clip_skip],
										  outputs=automa_result_images)

				gr.on(
					triggers=[automa_prompt_input.change,
							  automa_negative_prompt_input.change,
							  automa_sampler.change,
							  automa_steps.change,
							  automa_CFG.change,
							  automa_width.change,
							  automa_height.change,
							  automa_Batch.change,
							  automa_n_iter.change,
							  automa_url.change,
							  automa_save.change,
							  automa_save_on_api_host.change,
							  automa_checkpoint.change,
							  automa_vae.change,
							  automa_clip_skip.change, ],
					fn=ui_code.set_automa_settings,
					inputs=[automa_prompt_input,
							automa_negative_prompt_input,
							automa_sampler,
							automa_checkpoint,
							automa_steps,
							automa_CFG,
							automa_width,
							automa_height,
							automa_Batch,
							automa_n_iter,
							automa_url,
							automa_save,
							automa_save_on_api_host,
							automa_vae,
							automa_clip_skip],
					outputs=None)

			with gr.Tab('Interrogate') as interrogate:
				input_image = gr.Image(type='filepath', height=300)
				output_interrogation = gr.Textbox()
				interrogate_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa_url'])
				button_interrogate = gr.Button('Interrogate')
				button_interrogate.click(ui_code.run_automa_interrogation, [input_image, interrogate_url],
										 output_interrogation)
				input_image.change(ui_code.run_automa_interrogation, [input_image, interrogate_url],
								   output_interrogation)
			with gr.Tab('Interrogate Batch') as interrogate_batch:
				input_image_gallery = gr.Gallery()
				output_interrogation = gr.Textbox()
				with gr.Row():
					save = gr.Checkbox(label="Save to File", info="Save the whole to a file?", value=True)
					interrogate_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa_url'])
				button_interrogate = gr.Button('Interrogate')
				button_interrogate.click(ui_code.run_automa_interrogation_batch,
										 [input_image_gallery, interrogate_url, save], output_interrogation)
			with gr.Tab('Extensions') as extensions:
				with gr.Tab('Adetailer') as adetailer:
					automa_adetailer_enable = gr.Checkbox(label="Enable Adetailer",
														  value=g.settings_data['automa_adetailer_enable'])

					automa_adetailer_render_both = gr.Checkbox(label="Render with and without Adetailer",
														  value=g.settings_data['automa_adetailer_render_both'])

					automa_ad_use_inpaint_width_height = gr.Checkbox(label="Use inpaint with height",
																	 value=g.settings_data[
																		 'automa_ad_use_inpaint_width_height'])

					automa_ad_model = gr.Dropdown(
						choices=["face_yolov8n.pt",
								 "face_yolov8s.pt",
								 "hand_yolov8n.pt",
								 "person_yolov8n-seg.pt",
								 "yolov8x-worldv2-seg.pt",
								 "mediapipe_face_full.pt",
								 "mediapipe_face_short.pt",
								 "mediapipe_face_mesh.pt",
								 "mediapipe_face_mesh_eyes_only.pt"], value=g.settings_data['automa_ad_model'], label='Model')

					automa_ad_denoising_strength = gr.Slider(0, 1, step=0.1,
															 value=g.settings_data['automa_ad_denoising_strength'],
															 label="Denoising strength",
															 info="Denoising strength 0-1.")

					automa_ad_clip_skip = gr.Slider(1, 5, step=1, value=g.settings_data['automa_ad_clip_skip'],
													label="Clipskip",
													info="Clipskip 1-5.")

					automa_ad_confidence = gr.Slider(0, 1, step=0.1, value=g.settings_data['automa_ad_confidence'],
													 label="Confidence",
													 info="Level of confidence 0-1.")

					gr.on(
						triggers=[automa_adetailer_enable.change,
								  automa_adetailer_render_both.change,
								  automa_ad_use_inpaint_width_height.change,
								  automa_ad_model.change,
								  automa_ad_denoising_strength.change,
								  automa_ad_clip_skip.change,
								  automa_ad_confidence.change],
						fn=ui_code.set_automa_adetailer,
						inputs=[automa_adetailer_enable,
								automa_adetailer_render_both,
								automa_ad_use_inpaint_width_height,
								automa_ad_model,
								automa_ad_denoising_strength,
								automa_ad_clip_skip,
								automa_ad_confidence],
						outputs=None
					)
				with gr.Tab('Layer Diffusion') as layerdiffuse:
					automa_layerdiffuse_enable = gr.Checkbox(label="Enable Layer Diffusion",
														  value=g.settings_data['automa_layerdiffuse_enable'])

					gr.on(
						triggers=[automa_layerdiffuse_enable.change],
						fn=ui_code.set_automa_layerdiffuse,
						inputs=[automa_layerdiffuse_enable],
						outputs=None
					)

		with gr.Tab("HordeAI") as hordeai:
			gr.on(
				triggers=[hordeai.select],
				fn=ui_code.hordeai_get_last_prompt,
				inputs=None,
				outputs=[ui.hordeai_prompt_input,
						 ui.hordeai_negative_prompt_input,
						 ui.horde_api_key,
						 ui.horde_model,
						 ui.horde_sampler,
						 ui.horde_steps,
						 ui.horde_CFG,
						 ui.horde_width,
						 ui.horde_height,
						 ui.horde_clipskip]
			)
			gr.Interface(
				ui_code.run_hordeai_generation,
				[
					ui.hordeai_prompt_input,
					ui.hordeai_negative_prompt_input,
					ui.horde_api_key,
					ui.horde_model,
					ui.horde_sampler,
					ui.horde_steps,
					ui.horde_CFG,
					ui.horde_width,
					ui.horde_height,
					ui.horde_clipskip,
				]
				, outputs=gr.Image(label="Generated Image"),  # "text",
				allow_flagging='never',
				flagging_options=None,
				# live=True
			)

		with gr.Tab("SwarmUI") as swarmui:
			with gr.Tab('Generate') as swarm_generate:
				with gr.Row():
					with gr.Column(scale=3):
						swarmui_prompt_input = gr.TextArea(g.last_prompt, lines=5, label="Prompt")
						swarmui_negative_prompt_input = gr.TextArea(g.last_negative_prompt, lines=3,
															   label="Negative Prompt")
						with gr.Row():
							with gr.Column(scale=2):
								swarmui_sampler = gr.Dropdown(
									choices=g.settings_data['swarmui_samplers'], value=g.settings_data['swarmui_sampler'],
									label='Sampler')
							with gr.Column(scale=2):
								swarmui_checkpoint = gr.Dropdown(
									choices=g.settings_data['swarmui_checkpoints'],
									value=g.settings_data['swarmui_checkpoint'],
									label='Checkpoint')
							with gr.Column(scale=2):
								swarmui_vae = gr.Dropdown(
									choices=g.settings_data['swarmui_vaes'],
									value=g.settings_data['swarmui_vae'],
									label='VAE')
						with gr.Row():
							swarmui_steps = gr.Slider(1, 100, step=1, value=g.settings_data['swarmui_steps'],
													 label="Steps",
													 info="Choose between 1 and 100")
							swarmui_CFG = gr.Slider(0, 20, step=0.1, value=g.settings_data['swarmui_cfg_scale'],
												   label="CFG Scale",
												   info="Choose between 1 and 20")
							swarmui_clip_skip = gr.Slider(0, 12, step=1, value=g.settings_data['swarmui_clip_skip'],
														 label="Clip Skip",
														 info="Choose between 1 and 12")
						with gr.Row():
							with gr.Column(scale=3):
								swarmui_width = gr.Slider(1, 2048, step=1, value=g.settings_data['swarmui_width'],
														 label="Width",
														 info="Choose between 1 and 2048")
								swarmui_height = gr.Slider(1, 2048, step=1, value=g.settings_data['swarmui_height'],
														  label="Height",
														  info="Choose between 1 and 2048")
							with gr.Column(scale=0):
								swarmui_size_button = gr.Button('Switch size', elem_classes="gr-small-button")
							with gr.Column(scale=1):
								swarmui_Batch = gr.Slider(1, 250, step=1, value=g.settings_data['swarmui_batch'],
													 label="Batch",
													 info="The number of simultaneous images in each batch, range from 1-250.")
								swarmui_n_iter = gr.Slider(1, 500, step=1, value=g.settings_data['swarmui_n_iter'],
													  label="Iterations",
													  info="The number of sequential batches to be run, range from 1-500.")
						with gr.Row():
							swarmui_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['swarmui_url'])
							swarmui_save = gr.Checkbox(label="Save", info="Save the image?",
													  value=g.settings_data['swarmui_save'])
							swarmui_save_on_api_host = gr.Checkbox(label="Save", info="Save the image on API host?",
																  value=g.settings_data['swarmui_save_on_api_host'])
					with gr.Column(scale=1):
						swarmui_refresh_button = gr.Button('Refresh')
						swarmui_start_button = gr.Button('Generate')
						swarmui_result_images = gr.Gallery(label='output images', )


				swarmui_size_button.click(fn=ui_code.swarmui_switch_size,
										 inputs=[swarmui_width, swarmui_height],
										 outputs=[swarmui_width, swarmui_height])

				swarmui_refresh_button.click(fn=ui_code.swarmui_refresh,
											inputs=None,
											outputs=[automa_sampler, automa_checkpoint, automa_vae])

				swarmui_start_button.click(fn=ui_code.run_swarmui_generation,
										  inputs=[swarmui_prompt_input,
												  swarmui_negative_prompt_input,
												  swarmui_sampler,
												  swarmui_checkpoint,
												  swarmui_steps,
												  swarmui_CFG,
												  swarmui_width,
												  swarmui_height,
												  swarmui_Batch,
												  swarmui_n_iter,
												  swarmui_url,
												  swarmui_save,
												  swarmui_save_on_api_host,
												  swarmui_vae,
												  swarmui_clip_skip],
										  outputs=swarmui_result_images)


				gr.on(
					triggers=[swarmui_prompt_input.change,
							  swarmui_negative_prompt_input.change,
							  swarmui_sampler.change,
							  swarmui_steps.change,
							  swarmui_CFG.change,
							  swarmui_width.change,
							  swarmui_height.change,
							  swarmui_Batch.change,
							  swarmui_n_iter.change,
							  swarmui_url.change,
							  swarmui_save.change,
							  swarmui_save_on_api_host.change,
							  swarmui_checkpoint.change,
							  swarmui_vae.change,
							  swarmui_clip_skip.change, ],
					fn=ui_code.set_swarmui_settings,
					inputs=[swarmui_prompt_input,
							swarmui_negative_prompt_input,
							swarmui_sampler,
							swarmui_checkpoint,
							swarmui_steps,
							swarmui_CFG,
							swarmui_width,
							swarmui_height,
							swarmui_Batch,
							swarmui_n_iter,
							swarmui_url,
							swarmui_save,
							swarmui_save_on_api_host,
							swarmui_vae,
							swarmui_clip_skip],
					outputs=None)


	with gr.Tab("Interrogation") as interrogation:
		with gr.Tab("Moondream"):
			with gr.Tab("Single"):
				with gr.Row():
					with gr.Column(scale=3):
						moon_prompt = gr.Textbox(label="Input Prompt", value='Describe this image.', placeholder="Type here...", scale=3)
					with gr.Column(scale=1):
						with gr.Row():
							moon_max_tokens = gr.Dropdown(label='Max Tokens', choices=[16,32,64,128,256,512], value=512)
							moon_low_mem = gr.Checkbox(label="Low Mem", value=g.settings_data["moon_low_mem"])
							moon_submit = gr.Button("Submit")
							moon_get_prompt = gr.Button("Get Prompt")
						with gr.Row():
							moon_unload_model = gr.Button("Unload Moondream")
				with gr.Row():
					moon_img = gr.Image(type="pil", label="Upload an Image")
					with gr.Column():
						moon_output = gr.Markdown(label="Response")
						#ann = gr.Image(visible=False, label="Annotated Image")

				moon_submit.click(ui_code.moon_answer_question, [moon_img, moon_prompt, moon_max_tokens], moon_output)
				moon_prompt.submit(ui_code.moon_answer_question, [moon_img, moon_prompt, moon_max_tokens], moon_output)
				moon_get_prompt.click(ui_code.moon_get_prompt, [moon_img, moon_prompt, moon_max_tokens], moon_output)
				moon_unload_model.click(ui_code.moon_unload,None,moon_output)
				#moon_output.change(ui_code.moon_process_answer, [moon_img, moon_output], ann, show_progress=False)
			with gr.Tab("Batch"):
				with gr.Row():
					with gr.Column(scale=3):
						moon_batch_prompt = gr.Textbox(label="Input Prompt", value='Describe this image.', placeholder="Type here...", scale=3)
					with gr.Column(scale=1):
						with gr.Row():
							moon_batch_max_tokens = gr.Dropdown(label='Max Tokens', choices=[16,32,64,128,256,512], value=512)
							moon_batch_low_mem = gr.Checkbox(label="Low Mem", value=g.settings_data["moon_low_mem"])
							moon_batch_submit = gr.Button("Submit")
							moon_batch_get_prompt = gr.Button("Get Prompt")
						with gr.Row():
							moon_batch_unload_model = gr.Button("Unload Moondream")
				with gr.Row():
						moon_batch_gallery = gr.Gallery(type="pil", label="Upload Images")
						moon_batch_output = gr.Markdown(label="Response")
				moon_batch_submit.click(ui_code.moon_batch_answer_question, [moon_batch_gallery, moon_batch_prompt, moon_batch_max_tokens], moon_batch_output)
				moon_batch_prompt.submit(ui_code.moon_batch_answer_question, [moon_batch_gallery, moon_batch_prompt, moon_batch_max_tokens], moon_batch_output)
				moon_batch_get_prompt.click(ui_code.moon_batch_get_prompt, [moon_batch_gallery, moon_batch_prompt, moon_batch_max_tokens], moon_batch_output)
				moon_batch_unload_model.click(ui_code.moon_unload,None,moon_output)
			with gr.Tab("Image improver"):
				with gr.Row():
					with gr.Column(scale=3):
						moon_improver_prompt = gr.Textbox(label="Input Prompt", value='How to improve this image?', placeholder="Type here...", scale=3)
					with gr.Column(scale=1):
						with gr.Row():
							moon_improver_max_tokens = gr.Dropdown(label='Max Tokens', choices=[16,32,64,128,256,512], value=512)
							moon_improver_low_mem = gr.Checkbox(label="Low Mem", value=g.settings_data["moon_low_mem"])
							moon_improver_submit = gr.Button("Submit")
						with gr.Row():
							moon_improver_unload_model = gr.Button("Unload Moondream")
				with gr.Row():
					moon_improver_gallery = gr.Gallery(type="pil", label="Upload Images")
					moon_improver_output = gr.Markdown(label="Response")
					moon_improver_output_gallery = gr.Gallery()


				moon_improver_submit.click(fn=ui_code.moon_improver,
										   inputs=[moon_improver_gallery, moon_improver_prompt, moon_improver_max_tokens],
										   outputs=[moon_improver_output,moon_improver_output_gallery])
			with gr.Tab("File Renamer"):
				with gr.Tab("Rename"):
					with gr.Row():
						with gr.Column(scale=3):
							moon_folder_name = gr.Textbox(label="The path and all its substructures will be processed?'", value='F:\\test', placeholder="give a path...", scale=3)
						with gr.Column(scale=1):
							moon_folder_status = gr.Textbox('', label=f'Status', placeholder="status")
							moon_folder_submit = gr.Button("Submit")
							moon_folder_stop = gr.Button("Stop")

						moon_folder_submit.click(fn=ui_code.moon_file_rename,
												 inputs=moon_folder_name,
												 outputs=moon_folder_status)
						moon_folder_stop.click(fn=ui_code.moon_file_rename_stop,
												 inputs=None,
												 outputs=moon_folder_status)

				with gr.Tab("Story teller"):
					gr.Markdown("""<span style="color: red; font-weight: bold;">To use this feature you have to install a Ollama Server</span>""")
					with gr.Row():
						with gr.Column(scale=3):
							story_teller_enabled = gr.Checkbox(label="Enable", info="Enable Story telling?",
													  value=g.settings_data['story_teller_enabled'])

							story_teller_seconds_step_enabled = gr.Checkbox(label="Enable second step", info="Enable advanced Story telling?",
															   value=g.settings_data['story_teller_seconds_step_enabled'])

							image_description_model = gr.Dropdown(
								choices=g.settings_data['image_description_models'],
								value=g.settings_data['image_description_model'],
								label='Image description Model')


							image_description_system_context = gr.Textbox(label="Image description System Context", value=g.settings_data['image_description_system_context'], placeholder="You are an assistant who describes the content and composition of images. Describe only what you see in the image, not what you think the image is about.Be factual and literal. Do not use metaphors or similes. Be concise.", scale=3)
							image_description_prompt = gr.Textbox(label="Image description prompt", value=g.settings_data['image_description_prompt'], placeholder="Make it describe the image", scale=3)

							story_teller_model = gr.Dropdown(
								choices=g.settings_data['story_teller_models'],
								value=g.settings_data['story_teller_model'],
								label='Story Teller Model')

							story_teller_system_context = gr.Textbox(label="Story telling System Context", value=g.settings_data['story_teller_system_context'], placeholder="You are an vibrant story teller, you tell fantastic detailed storys about a image description thats given.Be factual and literal. Do not use metaphors or similes. Be concise.", scale=3)
							story_teller_prompt = gr.Textbox(label="Story telling prompt", value=g.settings_data['story_teller_prompt'], placeholder="Make it tell a story", scale=3)


							story_teller_host = gr.Textbox(label="URL for Ollama Server", value=g.settings_data['story_teller_host'], placeholder="http://localhost:11434", scale=3)
							story_teller_timeout = gr.Textbox(label="Timeout Ollama Server", value=g.settings_data['story_teller_timeout'], placeholder="300", scale=3)


							story_teller_temperature = gr.Slider(0.1, 1, step=0.1, value=g.settings_data['story_teller_temperature'],
														 label="Temperature",
														 info="Choose between 0.1 and 1")
							story_teller_max_tokens = gr.Textbox(label="Max Tokens", value=g.settings_data['story_teller_max_tokens'], placeholder="300", scale=3)


							gr.on(
								triggers=[story_teller_enabled.change,
										  story_teller_seconds_step_enabled.change,
										  story_teller_model.change,
										  image_description_model.change,
										  image_description_system_context.change,
										  image_description_prompt.change,
										  story_teller_system_context.change,
										  story_teller_prompt.change,
										  story_teller_host.change,
										  story_teller_timeout.change,
										  story_teller_temperature.change,
										  story_teller_max_tokens.change
										  ],
								fn=ui_code.set_story_teller,
								inputs=[story_teller_enabled,
										story_teller_seconds_step_enabled,
										story_teller_model,
										image_description_model,
										image_description_system_context,
										image_description_prompt,
										story_teller_system_context,
										story_teller_prompt,
										story_teller_host,
										story_teller_timeout,
										story_teller_temperature,
										story_teller_max_tokens],
								outputs=None
							)





						with gr.Column(scale=1):
							story_teller_status = gr.Textbox('', label=f'Status', placeholder="status")






		moon_improver_low_mem.change(fn=ui_code.moon_set_low_mem,
									 inputs=moon_improver_low_mem,
									 outputs=[moon_batch_low_mem,moon_low_mem])
		moon_batch_low_mem.change(fn=ui_code.moon_set_low_mem,
									 inputs=moon_batch_low_mem,
									 outputs=[moon_improver_low_mem,moon_low_mem])
		moon_low_mem.change(fn=ui_code.moon_set_low_mem,
									 inputs=moon_low_mem,
									 outputs=[moon_batch_low_mem,moon_improver_low_mem])



		with gr.Tab("PNG Info"):
			with gr.Row():
				with gr.Column(scale=3):
					png_info_img = gr.Image(type="pil", label="Upload an Image", height=300)
				with gr.Column(scale=1):
					png_info_output = gr.Markdown(label="Response")
			png_info_img.change(fn=ui_code.png_info_get,
								inputs=png_info_img,
								outputs=png_info_output)

	with gr.Tab('Model testing')as model_test:
		with gr.Tab('Setup test'):
			with gr.Row():
				with gr.Column(scale=3):
					model_test_list = gr.Dropdown(label='Select the List(s) you want to use',
												  choices=ui_code.prompt_iterator.selectors,
												  multiselect=True, value=g.settings_data['model_test_list'],
												  interactive=True)
					with gr.Row():
						model_test_type = gr.Radio(['Largest List', 'Full Run'],
												   label='Select type of test, Full run may take very long time',
												   value=g.settings_data['model_test_type'], interactive=True)
						model_test_gen_type = gr.Radio(['Largest List', 'Full Run'],
													   label='Select type of test generation params, Full run may take very long time',
													   value=g.settings_data['model_test_gen_type'], interactive=True)
					model_test_result_images = gr.Gallery(label='output images', height=300, rows=1, columns=6,
														  format='png')
					model_test_sample = gr.Textbox(label=f'A sample of your selection', placeholder="Sample", lines=5)

				with gr.Column(scale=1):
					model_test_status = gr.Textbox(label=f'Status', placeholder="Status", lines=1)
					model_test_sample_button = gr.Button('Get a sample')
					model_test_all_sample_button = gr.Button('Get all samples')
					model_test_run_button = gr.Button('Run test')
					model_test_stop_button = gr.Button('Stop test')

		with gr.Tab('Generation Settings'):
			model_test_steps = ui_code.prompt_iterator.setting_dropdown(g.settings_data['model_test_steps_list'],
																		'Steps',
																		g.settings_data['model_test_steps'])
			model_test_cfg = ui_code.prompt_iterator.setting_dropdown(g.settings_data['model_test_cfg_list'], 'CFG',
																	  g.settings_data['model_test_cfg'])
			model_test_dimensions = ui_code.prompt_iterator.setting_dropdown(
				g.settings_data['model_test_dimensions_list'],
				'Image Dimension',
				g.settings_data['model_test_dimensions'])

		with gr.Tab('Characters'):
			model_test_character = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.character, 'Character',
																		 g.settings_data['model_test_setup'][
																			 'Character'])

			model_test_celebrities = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.celebrities,
																		   'Celebrities',
																		   g.settings_data['model_test_setup'][
																			   'Celebrities'])
			model_test_creature_air = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.creature_air,
																			'Air Creatures',
																			g.settings_data['model_test_setup'][
																				'Air Creatures'])
			model_test_creature_land = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.creature_land,
																			 'Land Creatures',
																			 g.settings_data['model_test_setup'][
																				 'Land Creatures'])
			model_test_creature_sea = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.creature_sea,
																			'Sea Creatures',
																			g.settings_data['model_test_setup'][
																				'Sea Creatures'])
			model_test_character_objects = ui_code.prompt_iterator.data_dropdown(
				ui_code.prompt_iterator.character_objects,
				'Character Objects',
				g.settings_data['model_test_setup'][
					'Character Objects'])
			model_test_character_adj = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.character_adj,
																			 'Character Adjectives',
																			 g.settings_data['model_test_setup'][
																				 'Character Adjectives'])

		with gr.Tab('Vehicles'):
			model_test_vehicles_air = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.vehicles_air,
																			'Air Vehicle',
																			g.settings_data['model_test_setup'][
																				'Air Vehicle'])
			model_test_vehicles_land = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.vehicles_land,
																			 'Land Vehicle',
																			 g.settings_data['model_test_setup'][
																				 'Land Vehicle'])
			model_test_vehicles_sea = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.vehicles_sea,
																			'Sea Vehicle',
																			g.settings_data['model_test_setup'][
																				'Sea Vehicle'])
			model_test_vehicles_space = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.vehicles_space,
																			  'Space Vehicle',
																			  g.settings_data['model_test_setup'][
																				  'Space Vehicle'])
		with gr.Tab('Relations'):
			model_test_moving_relation = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.moving_relation,
																			   'Moving relation',
																			   g.settings_data['model_test_setup'][
																				   'Moving relation'])
			model_test_still_relation = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.still_relation,
																			  'Still relation',
																			  g.settings_data['model_test_setup'][
																				  'Still relation'])
		with gr.Tab('Adjectives'):
			model_test_object_adj = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.object_adj,
																		  'Object Adjectives',
																		  g.settings_data['model_test_setup'][
																			  'Object Adjectives'])
			model_test_visual_adj = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.visual_adj,
																		  'Visual Adjectives',
																		  g.settings_data['model_test_setup'][
																			  'Visual Adjectives'])
			model_test_visual_qualities = ui_code.prompt_iterator.data_dropdown(
				ui_code.prompt_iterator.visual_qualities,
				'Visual Qualities',
				g.settings_data['model_test_setup'][
					'Visual Qualities'])
		with gr.Tab('Settings'):
			model_test_settings = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.settings, 'Setup',
																		g.settings_data['model_test_setup']['Setup'])
			model_test_things = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.things, 'Things',
																	  g.settings_data['model_test_setup']['Things'])
		with gr.Tab('Style'):
			model_test_colors = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.colors, 'Colors',
																	  g.settings_data['model_test_setup']['Colors'])
			model_test_styles = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.styles, 'Styles',
																	  g.settings_data['model_test_setup']['Styles'])
			model_test_artists = ui_code.prompt_iterator.data_dropdown(ui_code.prompt_iterator.artists, 'Artists',
																	   g.settings_data['model_test_setup']['Artists'])

		with gr.Tab('Instruction Prompt'):
			model_test_inst_prompt = gr.Textbox(label=f'Prompt instruction for Model test',
												value=g.settings_data['prompt_templates']['model_test_instruction'],
												lines=20)

		gr.on(
			triggers=[model_test_list.change,
					  model_test_inst_prompt.change,
					  model_test_type.change,
					  model_test_steps.change,
					  model_test_dimensions.change,
					  model_test_gen_type.change,
					  model_test_cfg.change],
			fn=ui_code.prompt_iterator.save_test_data,
			inputs=[model_test_list,
					model_test_inst_prompt,
					model_test_type,
					model_test_steps,
					model_test_dimensions,
					model_test_gen_type,
					model_test_cfg],
			outputs=None)

		model_test_sample_button.click(fn=ui_code.prompt_iterator.get_sample,
									   inputs=None,
									   outputs=[model_test_sample, model_test_status])

		model_test_all_sample_button.click(fn=ui_code.prompt_iterator.get_all_samples,
										   inputs=None,
										   outputs=[model_test_sample, model_test_status])

		model_test_run = model_test_run_button.click(fn=ui_code.run_test,
													 inputs=None,
													 outputs=[model_test_result_images, model_test_status])

		model_test_stop_button.click(fn=ui_code.stop_job,
									 inputs=None,
									 outputs=None,
									 cancels=[model_test_run])

	with gr.Tab('Deep Dive') as deep_dive:
		top_k_slider = gr.Slider(1, max_top_k, value=g.settings_data['top_k'], step=1,
								 label="How many entries to retrieve:")
		search = gr.Textbox(f"", label=f'Context search')
		textboxes = []
		visible = True
		for i in range(max_top_k - 1):
			if i + 1 > g.settings_data['top_k']:
				visible = False
			t = gr.Textbox(f"Retrieved context {i}", label=f'Context {i + 1}', visible=visible)
			textboxes.append(t)

		output = textboxes
		# output.append(prompt_input)

		for text in textboxes:
			text.focus(ui_code.dive_into, text, output)

		deep_dive.select(ui_code.get_context_details, textboxes, textboxes)
		top_k_slider.change(ui_code.variable_outputs, top_k_slider, textboxes)
		search.change(ui_code.dive_into, search, textboxes)

	with gr.Tab("File2File") as batch_run:

		with gr.Row():
			with gr.Column(scale=3):
				f2f_input_file = gr.Files()

			with gr.Column(scale=1):
				f2f_status = gr.Textbox(f"", label=f'Status',
										placeholder='Status')
				f2f_summary = gr.Checkbox(label="Summary", info="Create a summary from the LLM prompt?",
										  value=g.settings_data['summary'])
				f2f_submit_button = gr.Button('Run Batch')

		f2f_submit_button.click(fn=ui_code.run_batch, inputs=f2f_input_file, outputs=f2f_status)
		f2f_summary.change(fn=ui_code.set_summary, inputs=f2f_summary, outputs=None)

	gr.on(
		triggers=[generator.select],
		fn=ui_code.all_get_last_prompt,
		inputs=None,
		outputs=[ui.hordeai_prompt_input,
				 ui.hordeai_negative_prompt_input,
				 ui.horde_api_key,
				 ui.horde_model,
				 ui.horde_sampler,
				 ui.horde_steps,
				 ui.horde_CFG,
				 ui.horde_width,
				 ui.horde_height,
				 ui.horde_clipskip,
				 automa_prompt_input,
				 automa_negative_prompt_input,
				 automa_sampler,
				 automa_steps,
				 automa_CFG,
				 automa_width,
				 automa_height,
				 automa_Batch,
				 automa_n_iter,
				 automa_url,
				 automa_save,
				 automa_save_on_api_host,
				 automa_checkpoint,
				 automa_vae,
				 automa_clip_skip
				 ]
	)
	gr.on(
		triggers=[automatic1111.select, generate.select],
		fn=ui_code.automa_get_last_prompt,
		inputs=None,
		outputs=[automa_prompt_input,
				 automa_negative_prompt_input,
				 automa_sampler,
				 automa_steps,
				 automa_CFG,
				 automa_width,
				 automa_height,
				 automa_Batch,
				 automa_n_iter,
				 automa_url,
				 automa_save,
				 automa_save_on_api_host,
				 automa_checkpoint]
	)

	gr.on(
		triggers=[swarmui.select],
		fn=ui_code.swarmui_get_last_prompt,
		inputs=None,
		outputs=[swarmui_prompt_input,
				 swarmui_negative_prompt_input,
				 swarmui_sampler,
				 swarmui_steps,
				 swarmui_CFG,
				 swarmui_width,
				 swarmui_height,
				 swarmui_Batch,
				 swarmui_n_iter,
				 swarmui_url,
				 swarmui_save,
				 swarmui_save_on_api_host,
				 swarmui_checkpoint]
	)

	with gr.Tab('Image Scoring'):
		with gr.Tab('Single Image'):
			with gr.Row():
				with gr.Column(scale=3):
					score_image = gr.Image(label='Image', type='pil')
				with gr.Column(scale=1):
					score_result = gr.Textbox("", label=f'Image Score', placeholder="The Score of your Image", lines=1)
					score_button = gr.Button('Score Image')

			score_button.click(image_score.get_single_aestetics_score, score_image, score_result)

		with gr.Tab('Image Folder'):
			with gr.Row():
				with gr.Column(scale=3):
					score_min_aestetics_level = gr.Slider(0, 10, step=0.1, value=7, label="Minimum Score",
														  info="Choose between 1 and 10")
					score_keep_structure = gr.Checkbox(label="Create new Folder", value=False)
					score_output_folder = gr.Textbox("", label=f'Where to store the scored images', lines=1)
					score_images = gr.File(file_count='directory')

				with gr.Column(scale=1):
					score_images_result = gr.Textbox("", label=f'Status', placeholder="Status", lines=1)
					score_images_button = gr.Button('Score Image')

			score_images_button.click(fn=image_score.run_aestetic_prediction,
									  inputs=[score_images, score_min_aestetics_level, score_keep_structure,
											  score_output_folder],
									  outputs=score_images_result)

	with gr.Tab('Settings'):
		with gr.Tab("Presets") as presets:
			with gr.Row():
				with gr.Column(scale=3):
					with gr.Row():
						with gr.Column(scale=3):
							preset_select = gr.Dropdown(choices=g.settings_data['preset_list'],
														value=g.settings_data['selected_preset'], label='Preset')
						with gr.Column(scale=1):
							preset_reload_button = gr.Button('Reload presets')
							preset_load_button = gr.Button('Load preset')
							preset_save_button = gr.Button('Save preset')
					with gr.Row():
						with gr.Column(scale=3):
							preset_name = gr.TextArea('', lines=1, label="Filename", placeholder='Enter preset name')
						with gr.Column(scale=1):
							preset_create_button = gr.Button('Create new preset')
				with gr.Column(scale=1):
					with gr.Row():
						preset_status = gr.TextArea(lines=1, label="Status", placeholder="Status")

			gr.on(
				triggers=[presets.select],
				fn=ui_code.load_preset_list,
				inputs=None,
				outputs=[preset_select]
			)
			preset_load_button.click(fn=ui_code.load_preset, inputs=preset_select, outputs=preset_status)
			preset_save_button.click(fn=ui_code.save_preset, inputs=preset_select, outputs=preset_status)
			preset_create_button.click(fn=ui_code.save_preset, inputs=preset_name, outputs=preset_status)
			preset_reload_button.click(fn=ui_code.load_preset_list, inputs=None, outputs=[preset_select, preset_status])

		with gr.Tab("Model Settings") as llm_settings:
			with gr.Row():
				with gr.Column(scale=3):
					llm = gr.Dropdown(
						g.settings_data['model_list'].keys(), value=g.settings_data['LLM Model'], label="LLM Model",
						info="Will add more LLMs later!"
					)
					collection = gr.Dropdown(
						g.settings_data['collections_list'], value=g.settings_data['collection'], label="Collection",
						info="If you got more than one collection!"
					)
					embedding_model = gr.Dropdown(
						g.settings_data['embedding_model_list'], value=g.settings_data['embedding_model'],
						label="Embedding Model",
						info="If you dont know better, use all-MiniLM-L12-v2!"
					)
					temperature = gr.Slider(0, 1, step=0.1, value=g.settings_data['Temperature'], label="Temperature",
											info="Choose between 0 and 1")
					context = gr.Slider(0, 8192, step=1, value=g.settings_data['Context Length'],
										label="Context Length",
										info="Choose between 1 and 8192")
					gpu_layers = gr.Slider(0, 1024, step=1, value=g.settings_data['GPU Layers'], label="GPU Layers",
										   info="Choose between 1 and 1024")
					max_out_token = gr.Slider(0, 1024, step=1, value=g.settings_data['max output Tokens'],
											  label="max output Tokens",
											  info="Choose between 1 and 1024")
					top_k = gr.Slider(0, g.settings_data['max_top_k'], step=1, value=g.settings_data['top_k'],
									  label="how many entrys to be fetched from the vector store",
									  info="Choose between 1 and 50 be careful not to overload the context window of the LLM")

				with gr.Column(scale=1):
					model_status = gr.Textbox("", label=f'Status', placeholder="Status")
					model_save_button = gr.Button('Save model settings')

			gr.on(
				triggers=[llm_settings.select],
				fn=ui_code.get_llm_settings,
				inputs=None,
				outputs=[collection,
						 llm,
						 embedding_model,
						 temperature,
						 context,
						 gpu_layers,
						 max_out_token,
						 top_k
						 ]
			)

			model_save_button.click(fn=ui_code.set_model,
									inputs=[collection,
											llm,
											embedding_model,
											temperature,
											context,
											gpu_layers,
											max_out_token,
											top_k
											],
									outputs=model_status
									)

		with gr.Tab("Prompting Settings") as defaults:
			with gr.Tab('Negative Prompt'):
				with gr.Row():
					with gr.Column(scale=3):
						neg_prompt_text = gr.Textbox(g.settings_data['negative_prompt'],
													 label=f'Default Negative Prompt',
													 lines=10)
					with gr.Column(scale=1):
						neg_prompt_status = gr.TextArea('', label="Status", placeholder='Status')
						neg_prompt_submit_button = gr.Button('Save Negative Prompt')

				neg_prompt_submit_button.click(fn=ui_code.set_neg_prompt, inputs=neg_prompt_text,
											   outputs=neg_prompt_status)

			with gr.Tab('Rephrase instruction') as negative_prompt:
				with gr.Row():
					with gr.Column(scale=3):
						rephrase_instruction_text = gr.Textbox(g.settings_data['rephrase_instruction'],
															   label=f'Rephrase instruction',
															   lines=5)
					with gr.Column(scale=1):
						rephrase_instruction_status = gr.TextArea(lines=1, label="Status", placeholder='Status')
						rephrase_instruction_submit_button = gr.Button('Save Rephrase instruction')

				rephrase_instruction_submit_button.click(fn=ui_code.set_rephrase_instruction,
														 inputs=rephrase_instruction_text,
														 outputs=rephrase_instruction_status)

			with gr.Tab("Character") as Character:
				with gr.Row():
					with gr.Column(scale=3):
						prompt_template = gr.TextArea(
							g.settings_data["prompt_templates"][g.settings_data["selected_template"]], lines=20)
						prompt_template_select = gr.Dropdown(choices=g.settings_data["prompt_templates"].keys(),
															 value=g.settings_data["selected_template"],
															 label='Template',
															 interactive=True)

					with gr.Column(scale=1):
						prompt_template_status = gr.TextArea(lines=1, label="Status", placeholder='Status')
						prompt_template_submit_button = gr.Button('Save Character')

				gr.on(
					triggers=[Character.select],
					fn=ui_code.get_prompt_template,
					inputs=None,
					outputs=[prompt_template]
				)
				gr.on(
					triggers=[prompt_template_select.select],
					fn=ui_code.set_prompt_template_select,
					inputs=[prompt_template_select],
					outputs=[prompt_template]
				)

				prompt_template_submit_button.click(fn=ui_code.set_prompt_template,
													inputs=[prompt_template_select, prompt_template, ],
													outputs=[prompt_template_status])

	gr.on(
		triggers=[generator.select],
		fn=ui_code.all_get_last_prompt,
		inputs=None,
		outputs=[ui.hordeai_prompt_input,
				 ui.hordeai_negative_prompt_input,
				 ui.horde_api_key,
				 ui.horde_model,
				 ui.horde_sampler,
				 ui.horde_steps,
				 ui.horde_CFG,
				 ui.horde_width,
				 ui.horde_height,
				 ui.horde_clipskip,
				 automa_prompt_input,
				 automa_negative_prompt_input,
				 automa_sampler,
				 automa_steps,
				 automa_CFG,
				 automa_width,
				 automa_height,
				 automa_Batch,
				 automa_n_iter,
				 automa_url,
				 automa_save,
				 automa_save_on_api_host,
				 automa_checkpoint,
				 automa_vae,
				 automa_clip_skip
				 ]
	)
	gr.on(
		triggers=[automatic1111.select, generate.select],
		fn=ui_code.automa_get_last_prompt,
		inputs=None,
		outputs=[automa_prompt_input,
				 automa_negative_prompt_input,
				 automa_sampler,
				 automa_steps,
				 automa_CFG,
				 automa_width,
				 automa_height,
				 automa_Batch,
				 automa_n_iter,
				 automa_url,
				 automa_save,
				 automa_save_on_api_host,
				 automa_checkpoint]
	)

	gr.on(
		triggers=[chat.select,sailor.select,
				  generator.select,model_test.select,
				  deep_dive.select,batch_run.select],
		fn=ui_code.moon_unload,
		inputs=None,
		outputs=None
	)


if __name__ == "__main__":
	server_name = "localhost"
	if os.getenv("SERVER_NAME") is not None:
		server_name = os.getenv("SERVER_NAME")
	pq_ui.launch(favicon_path='logo/favicon32x32.ico', inbrowser=True, server_name=server_name,
				 server_port=49152)  # share=True
