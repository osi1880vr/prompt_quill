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
		gr.Markdown("**Prompt Quill**", elem_classes="app-title")  # Add unique ID for potential CSS styling

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
		input_file = gr.Files()
		with gr.Row():
			f2f_summary = gr.Checkbox(label="Summary", info="Create a summary from the LLM prompt?",
									  value=g.settings_data['summary'])
			finish = gr.Textbox(f"", label=f'Wait for its done',
								placeholder='Here you upload a file with your input prompts, it will then generate a new prompt based on each line of your file and write that to output file')

		file_submit_button = gr.Button('Run Batch')

		file_submit_button.click(ui_code.run_batch, input_file, finish)
		f2f_summary.change(ui_code.set_summary, f2f_summary, None)

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
								automa_Sampler = gr.Dropdown(
									choices=g.settings_data['automa_samplers'], value=g.settings_data['automa_Sampler'],
									label='Sampler')
							with gr.Column(scale=2):
								automa_Checkpoint = gr.Dropdown(
									choices=g.settings_data['automa_checkpoints'],
									value=g.settings_data['automa_Checkpoint'],
									label='Checkpoint')
						with gr.Row():
							with gr.Column(scale=3):
								automa_Steps = gr.Slider(1, 100, step=1, value=g.settings_data['automa_Steps'],
														 label="Steps",
														 info="Choose between 1 and 100")
							with gr.Column(scale=1):
								automa_CFG = gr.Slider(0, 20, step=0.1, value=g.settings_data['automa_CFG Scale'],
													   label="CFG Scale",
													   info="Choose between 1 and 20")
						with gr.Row():
							with gr.Column(scale=3):
								automa_Width = gr.Slider(1, 2048, step=1, value=g.settings_data['automa_Width'],
														 label="Width",
														 info="Choose between 1 and 2048")
								automa_Height = gr.Slider(1, 2048, step=1, value=g.settings_data['automa_Height'],
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
										 inputs=[automa_Width, automa_Height],
										 outputs=[automa_Width, automa_Height])

				automa_refresh_button.click(fn=ui_code.automa_refresh,
											inputs=None,
											outputs=[automa_Sampler, automa_Checkpoint])

				automa_start_button.click(fn=ui_code.run_automatics_generation,
										  inputs=[automa_prompt_input,
												  automa_negative_prompt_input,
												  automa_Sampler,
												  automa_Checkpoint,
												  automa_Steps,
												  automa_CFG,
												  automa_Width,
												  automa_Height,
												  automa_Batch,
												  automa_n_iter,
												  automa_url,
												  automa_save,
												  automa_save_on_api_host],
										  outputs=automa_result_images)

				gr.on(
					triggers=[automa_prompt_input.change,
							  automa_negative_prompt_input.change,
							  automa_Sampler.change,
							  automa_Steps.change,
							  automa_CFG.change,
							  automa_Width.change,
							  automa_Height.change,
							  automa_Batch.change,
							  automa_n_iter.change,
							  automa_url.change,
							  automa_save.change,
							  automa_save_on_api_host.change,
							  automa_Checkpoint.change],
					fn=ui_code.set_automa_settings,
					inputs=[automa_prompt_input,
							automa_negative_prompt_input,
							automa_Sampler,
							automa_Checkpoint,
							automa_Steps,
							automa_CFG,
							automa_Width,
							automa_Height,
							automa_Batch,
							automa_n_iter,
							automa_url,
							automa_save,
							automa_save_on_api_host],
					outputs=None)

			with gr.Tab('Interrogate') as interrogate:
				input_image = gr.Image(type='filepath', height=300)
				output_interrogation = gr.Textbox()
				interrogate_url = gr.TextArea(lines=1, label="API URL", value=g.settings_data['automa_url'])
				button_interrogate = gr.Button('Interrogate')
				button_interrogate.click(ui_code.run_automa_interrogation, [input_image, interrogate_url],
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

					automa_ad_use_inpaint_width_height = gr.Checkbox(label="Use inpaint with height",
																	 value=g.settings_data[
																		 'automa_ad_use_inpaint_width_height'])

					automa_ad_model = gr.Dropdown(
						choices=["face_yolov8n.pt",
								 "face_yolov8s.pt",
								 "hand_yolov8n.pt",
								 "person_yolov8n-seg.pt",
								 "person_yolov8s-seg.pt"], value=g.settings_data['automa_ad_model'], label='Model')

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
								  automa_ad_use_inpaint_width_height.change,
								  automa_ad_model.change,
								  automa_ad_denoising_strength.change,
								  automa_ad_clip_skip.change,
								  automa_ad_confidence.change],
						fn=ui_code.set_automa_adetailer,
						inputs=[automa_adetailer_enable,
								automa_ad_use_inpaint_width_height,
								automa_ad_model,
								automa_ad_denoising_strength,
								automa_ad_clip_skip,
								automa_ad_confidence],
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
						 ui.horde_Model,
						 ui.horde_Sampler,
						 ui.horde_Steps,
						 ui.horde_CFG,
						 ui.horde_Width,
						 ui.horde_Height,
						 ui.horde_Clipskip]
			)
			gr.Interface(
				ui_code.run_hordeai_generation,
				[
					ui.hordeai_prompt_input,
					ui.hordeai_negative_prompt_input,
					ui.horde_api_key,
					ui.horde_Model,
					ui.horde_Sampler,
					ui.horde_Steps,
					ui.horde_CFG,
					ui.horde_Width,
					ui.horde_Height,
					ui.horde_Clipskip,
				]
				, outputs=gr.Image(label="Generated Image"),  # "text",
				allow_flagging='never',
				flagging_options=None,
				# live=True
			)

	gr.on(
		triggers=[generator.select],
		fn=ui_code.all_get_last_prompt,
		inputs=None,
		outputs=[ui.hordeai_prompt_input,
				 ui.hordeai_negative_prompt_input,
				 ui.horde_api_key,
				 ui.horde_Model,
				 ui.horde_Sampler,
				 ui.horde_Steps,
				 ui.horde_CFG,
				 ui.horde_Width,
				 ui.horde_Height,
				 ui.horde_Clipskip,
				 automa_prompt_input,
				 automa_negative_prompt_input,
				 automa_Sampler,
				 automa_Steps,
				 automa_CFG,
				 automa_Width,
				 automa_Height,
				 automa_Batch,
				 automa_n_iter,
				 automa_url,
				 automa_save,
				 automa_save_on_api_host,
				 automa_Checkpoint
				 ]
	)
	gr.on(
		triggers=[automatic1111.select, generate.select],
		fn=ui_code.automa_get_last_prompt,
		inputs=None,
		outputs=[automa_prompt_input,
				 automa_negative_prompt_input,
				 automa_Sampler,
				 automa_Steps,
				 automa_CFG,
				 automa_Width,
				 automa_Height,
				 automa_Batch,
				 automa_n_iter,
				 automa_url,
				 automa_save,
				 automa_save_on_api_host,
				 automa_Checkpoint]
	)

	with gr.Tab("Sail the data ocean") as sailor:

		with gr.Tab('Sailing') as sailing:
			with gr.Tab('Main view'):
				with gr.Row():
					with gr.Column(scale=3):
						sail_text = gr.Textbox(g.settings_data['sail_text'], label=f'Start your journey with',
											   placeholder="Where do we set our sails", elem_id='sail-input-text')
						with gr.Row():
							sail_width = gr.Slider(1, 10000, step=1, value=g.settings_data['sail_width'],
											   label="Sail steps", info="Choose between 1 and 10000")
							sail_depth = gr.Slider(1, 10000, step=1, value=g.settings_data['sail_depth'],
											   label="Sail distance", info="Choose between 1 and 10000")
							sail_generate = gr.Checkbox(label="Generate with A1111",
														info="Do you want to directly generate the images?",
														value=g.settings_data['sail_generate'])
						with gr.Row():
							sail_sinus = gr.Checkbox(label="Add a sinus to the distance",
													 info="This will create a sinus wave based movement along the distance",
													 value=g.settings_data['sail_sinus'])
							sail_sinus_freq = gr.Slider(0.1, 10, step=0.1, value=g.settings_data['sail_sinus_freq'],
														label="Sinus Frequency", info="Choose between 0.1 and 10")
							sail_sinus_range = gr.Slider(1, 500, step=1, value=g.settings_data['sail_sinus_range'],
														 label="Sinus Multiplier", info="Choose between 1 and 500")
					with gr.Column(scale=1):
						with gr.Row():
							sail_submit_button = gr.Button('Start your journey')
							sail_stop_button = gr.Button('Interrupt your journey')
							sail_count_button = gr.Button('Count possible results')
							sail_check_connect_button = gr.Button('Check API Available')
						with gr.Row():
							sail_status = gr.Textbox('', label=f'Status', placeholder="Nothing yet")
						with gr.Row():
							sail_max_gallery_size = gr.Slider(1, 500, step=1, value=g.settings_data['sail_max_gallery_size'],
															  label="Max Gallery size",
															  info="Limit the number of images keept in the gallery choose between 1 and 500")
				with gr.Row():
					sail_result_images = gr.Gallery(label='output images',height=300,rows=1,columns=6)
				with gr.Row():
					sail_result = gr.Textbox("", label=f'Your journey journal', placeholder="Your journey logs")
			with gr.Tab('Filters'):
				with gr.Row():
					with gr.Column(scale=3):
						sail_filter_text = gr.Textbox(g.settings_data['sail_filter_text'],
													  label=f'List of negative words',
													  placeholder="Comma separated list of words you dont want in your prompt")
						sail_filter_not_text = gr.Textbox(g.settings_data['sail_filter_not_text'],
														  label=f'List of positive words',
														  placeholder="Comma separated list of words that must be part of the prompt")
					with gr.Column(scale=1):
						sail_filter_prompt = gr.Checkbox(label="Filter on prompt Level?",
														 info="With this you filter entries from the prompt generation. It may lead to long time until a prompt will match",
														 value=g.settings_data['sail_filter_prompt'])
						sail_filter_context = gr.Checkbox(label="Filter on context Level?",
														  info="With this you filter entries from the context prior to prompt generation. It may lead to empty context",
														  value=g.settings_data['sail_filter_context'])
				with gr.Row():
					sail_add_style = gr.Checkbox(label="Hard style specification", info="Add a text to each prompt",
												 value=g.settings_data['sail_add_style'])
					sail_style = gr.Textbox(g.settings_data['sail_style'], label=f'Style Spec',
											placeholder="Enter your hardcoded style")
				with gr.Row():
					sail_add_search = gr.Checkbox(label="Hard search specification",
												  info="Add a text to each search",
												  value=g.settings_data['sail_add_search'])
					sail_search = gr.Textbox(g.settings_data['sail_search'], label=f'Search Spec',
											 placeholder="Enter your hardcoded search")


			with gr.Tab('Prompt manipulation'):
				with gr.Row():
					sail_dyn_neg = gr.Checkbox(label="Use dynamic Negative Prompt",
											   info="Uses the negative if we find one, or the default. Be warned this can cause black images or other troubles.",
											   value=g.settings_data['sail_dyn_neg'])
					sail_add_neg = gr.Checkbox(label="Add to negative prompt",
											   info="Add a text to each negative prompt",
											   value=g.settings_data['sail_add_neg'])
				with gr.Row():
					sail_neg_prompt = gr.Textbox(g.settings_data['sail_neg_prompt'], label=f'Negative Prompt addon',
												 placeholder="Enter your negative prompt addon")
				with gr.Row():
					sail_summary = gr.Checkbox(label="Do summary of LLM prompt",
											   info="The prompt will get reduced to a summary",
											   value=g.settings_data['sail_summary'])
					sail_rephrase = gr.Checkbox(label="Rephrase LLM prompt",
												info="The prompt gets rephrased based on the rephrase prompt",
												value=g.settings_data['sail_rephrase'])
					sail_gen_rephrase = gr.Checkbox(label="Generate the input Prompt too",
													info="To see the effect of the rephrasing you can check here to get both prompts generated",
													value=g.settings_data['sail_gen_rephrase'])
				with gr.Row():
					sail_rephrase_prompt = gr.Textbox(g.settings_data['sail_rephrase_prompt'],
													  label=f'Rephrase Prompt',
													  placeholder="Enter your rephrase prompt")


			gr.on(
				triggers=[sail_text.change,
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
						  sail_filter_prompt.change
						  ],
				fn=ui_code.set_sailing_settings,
				inputs=[sail_text,
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
						sail_filter_prompt
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
						 sail_filter_prompt])

			start_sail = sail_submit_button.click(fn=ui_code.run_t2t_sail,
												  inputs=[],
												  outputs=[sail_result,
														   sail_result_images,
														   sail_status])
			sail_stop_button.click(fn=ui_code.stop_t2t_sail,
								   inputs=None,
								   outputs=None,
								   cancels=[start_sail])
			sail_check_connect_button.click(fn=ui_code.check_api_avail,
											inputs=None,
											outputs=sail_status)
			sail_count_button.click(fn=ui_code.count_context,
											inputs=None,
											outputs=sail_status)


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

	with gr.Tab('Image Scoring'):
		with gr.Tab('Single Image'):
			score_image = gr.Image(label='Image', type='pil')
			score_button = gr.Button('Score Image')
			score_result = gr.Textbox("", label=f'Image Score', placeholder="The Score of your Image", lines=1)
			score_button.click(image_score.get_single_aestetics_score, score_image, score_result)

		with gr.Tab('Image Folder'):
			score_images_button = gr.Button('Score Image')
			score_min_aestetics_level = gr.Slider(0, 10, step=0.1, value=7, label="Minimum Score",
												  info="Choose between 1 and 10")
			score_keep_structure = gr.Checkbox(label="Create new Folder", value=False)
			score_output_folder = gr.Textbox("", label=f'Where to store the scored images', lines=1)
			score_images_result = gr.Textbox("", label=f'Status', placeholder="Status", lines=1)
			score_images = gr.File(file_count='directory')
			score_images_button.click(image_score.run_aestetic_prediction,
									  [score_images, score_min_aestetics_level, score_keep_structure,
									   score_output_folder], score_images_result)

	with gr.Tab('Settings'):
		with gr.Tab("Character") as Character:
			gr.on(
				triggers=[Character.select],
				fn=ui_code.get_prompt_template,
				inputs=None,
				outputs=[ui.prompt_template]
			)
			gr.on(
				triggers=[ui.prompt_template_select.select],
				fn=ui_code.set_prompt_template_select,
				inputs=[ui.prompt_template_select],
				outputs=[ui.prompt_template]
			)
			gr.Interface(
				fn=ui_code.set_prompt_template,
				inputs=[ui.prompt_template_select, ui.prompt_template, ],
				outputs=[ui.prompt_template_status],
				allow_flagging='never',
				flagging_options=None
			)
		with gr.Tab('Rephrase instruction') as negative_prompt:
			rephrase_instruction_text = gr.Textbox(g.settings_data['rephrase_instruction'],
												   label=f'Rephrase instruction')
			rephrase_instruction_submit_button = gr.Button('Save Rephrase instruction')

			rephrase_instruction_submit_button.click(ui_code.set_rephrase_instruction, rephrase_instruction_text, None)

		with gr.Tab("Model Settings") as llm_settings:
			gr.on(
				triggers=[llm_settings.select],
				fn=ui_code.get_llm_settings,
				inputs=None,
				outputs=[ui.collection,
						 ui.LLM,
						 ui.embedding_model,
						 ui.Temperature,
						 ui.Context,
						 ui.GPU,
						 ui.max,
						 ui.top_k,
						 ui.Instruct
						 ]
			)

			gr.Interface(
				ui_code.set_model,
				[ui.collection,
				 ui.LLM,
				 ui.embedding_model,
				 ui.Temperature,
				 ui.Context,
				 ui.GPU,
				 ui.max,
				 ui.top_k,
				 ui.Instruct
				 ]
				, outputs="text",
				allow_flagging='never',
				flagging_options=None

			)

		with gr.Tab("Default") as defaults:
			with gr.Tab('Negative Prompt'):
				neg_prompt_text = gr.Textbox(g.settings_data['negative_prompt'], label=f'Default Negative Prompt')
				np_submit_button = gr.Button('Save Negative Prompt')

				np_submit_button.click(ui_code.set_neg_prompt, neg_prompt_text, None)

		with gr.Tab("Presets") as presets:
			with gr.Row():
				preset_select = gr.Dropdown(choices=g.settings_data['preset_list'],
											value=g.settings_data['selected_preset'], label='Preset')

				preset_load_button = gr.Button('Load preset')
				preset_save_button = gr.Button('Save preset')
				preset_reload_button = gr.Button('Reload presets')
			with gr.Row():
				preset_name = gr.TextArea('', lines=1, label="Filename", placeholder='Enter preset name')
				preset_create_button = gr.Button('Create new preset')
				preset_status = gr.TextArea('', lines=1, label="Status")
			gr.on(
				triggers=[presets.select],
				fn=ui_code.load_preset_list,
				inputs=None,
				outputs=[preset_select]
			)
			preset_load_button.click(ui_code.load_preset, preset_select, preset_status)
			preset_save_button.click(ui_code.save_preset, preset_select, preset_status)
			preset_create_button.click(ui_code.save_preset, preset_name, preset_status)
			preset_reload_button.click(ui_code.load_preset_list, None, preset_select)

if __name__ == "__main__":
	server_name = "localhost"
	if os.getenv("SERVER_NAME") is not None:
		server_name = os.getenv("SERVER_NAME")
	pq_ui.launch(favicon_path='logo/favicon32x32.ico', inbrowser=True, server_name=server_name,
				 server_port=49152)  # share=True
