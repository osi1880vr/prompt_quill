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

import gradio as gr
import llm_interface_qdrant
from generators.civitai.client import civitai_client
from generators.hordeai.client import hordeai_client
from generators.hordeai.client import hordeai_models
from horde_sdk import ANON_API_KEY




hordeai_model_list = hordeai_models().read_model_list()

import os

interface = llm_interface_qdrant.LLM_INTERFACE()

import model_list
def set_model(model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k, instruct):
	return interface.change_model(model,temperature,n_ctx,n_gpu_layers,max_tokens, top_k, instruct)

def get_last_prompt():
	return interface.last_prompt,interface.last_negative_prompt


def set_prompt(prompt_text):

	return_data = interface.set_prompt(prompt_text)

	return return_data

def run_civitai_generation(air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip):
	client = civitai_client()
	return client.request_generation(air, prompt, negative_prompt, steps, cfg, width, heigth, clipskip)
def run_hordeai_generation(api_key, prompt, negative_prompt, model, sampler, steps, cfg, width, heigth, clipskip):
	client = hordeai_client()
	return client.request_generation(api_key=api_key, prompt=prompt, negative_prompt=negative_prompt,
									 sampler=sampler, model=model, steps=steps, cfg=cfg, width=width, heigth=heigth, clipskip=clipskip)
def run_llm_response(query, history):
	return_data = interface.run_llm_response(query, history)
	return return_data


css = """
.gr-image {
  min-width: 60px !important;
  max-width: 60px !important;
  min-heigth: 65px !important;
  max-heigth: 65px !important;  
  
}
.app-title {
  font-size: 50px;
}
"""

civitai_prompt_input = gr.TextArea(interface.last_prompt, lines = 10, label="Prompt")
civitai_negative_prompt_input = gr.TextArea(interface.last_negative_prompt, lines = 5, label="Negative Prompt")
hordeai_prompt_input = gr.TextArea(interface.last_prompt, lines = 10, label="Prompt")
hordeai_negative_prompt_input = gr.TextArea(interface.last_negative_prompt, lines = 5, label="Negative Prompt")

with gr.Blocks(css=css) as pq_ui:

	with gr.Row():
		# Image element (adjust width as needed)
		gr.Image(os.path.join(os.getcwd(),"logo/pq_v_small.jpg"),width="20vw",show_label=False,show_download_button=False,container=False, elem_classes="gr-image",)

		# Title element (adjust font size and styling with CSS if needed)
		gr.Markdown("**Prompt Quill**", elem_classes="app-title")  # Add unique ID for potential CSS styling


	with gr.Tab("Chat"):



		gr.ChatInterface(
			interface.run_llm_response,
			chatbot=gr.Chatbot(height=500,render=False),
			textbox=gr.Textbox(placeholder="Make your prompts more creative", container=False, scale=7,render=False),
			theme="soft",
			examples=['A fishermans lake','night at cyberpunk city','living in a steampunk world'],
			cache_examples=True,
			retry_btn="🔄  Retry",
			undo_btn="↩️ Undo",
			clear_btn="Clear",

		)


	with gr.Tab("Character"):
		gr.Interface(
			set_prompt,
			[	gr.TextArea(interface.prompt_template,lines = 20),]
			,outputs=None,
			allow_flagging='never',
			flagging_options=None

		)

	with gr.Tab("Model Settings"):
		gr.Interface(
			set_model,
			[

				gr.Dropdown(
					model_list.model_list.keys(),value=list(model_list.model_list.keys())[0], label="LLM Model", info="Will add more LLMs later!"
				),
				gr.Slider(0, 1, step= 0.1, value=0.0, label="Temperature", info="Choose between 0 and 1"),
				gr.Slider(0, 8192, step= 1, value=3900, label="Context Length", info="Choose between 1 and 8192"),
				gr.Slider(0, 1024, step= 1, value=50, label="GPU Layers", info="Choose between 1 and 1024"),
				gr.Slider(0, 1024, step= 1, value=200, label="max output Tokens", info="Choose between 1 and 1024"),
				gr.Slider(0, 50, step= 1, value=5, label="how many entrys to be fetched from the vector store", info="Choose between 1 and 50 be careful not to overload the context window of the LLM"),
				gr.Checkbox(label='Instruct Model')
			]
			,outputs="text",
			allow_flagging='never',
			flagging_options=None

		)

	with gr.Tab("Generator") as generator:
		gr.on(
			triggers=[generator.select],
			fn=get_last_prompt,
			inputs=None,
			outputs=[civitai_prompt_input, civitai_negative_prompt_input],
		)
		with gr.Tab("Civitai") as civitai:

			gr.Interface(
				run_civitai_generation,
				[
					 gr.TextArea(lines = 1, label="Air",value='urn:air:sd1:checkpoint:civitai:4201@130072'),
					 civitai_prompt_input,
					 civitai_negative_prompt_input,
					 #gr.Dropdown(choices=["DPM++ 2M Karras", "Euler a", "Third Choice"]),
					 gr.Slider(0, 100, step= 1, value=20, label="Steps", info="Choose between 1 and 100"),
					 gr.Slider(0, 20, step= 0.1, value=7, label="CFG Scale", info="Choose between 1 and 20"),
					 gr.Slider(0, 1024, step= 1, value=512, label="Width", info="Choose between 1 and 1024"),
					 gr.Slider(0, 1024, step= 1, value=512, label="Height", info="Choose between 1 and 1024"),
					 gr.Slider(0, 10, step= 1, value=2, label="Clipskip", info="Choose between 1 and 10"),
					 ]
				,outputs=gr.Image(label="Generated Image"), #"text",
				allow_flagging='never',
				flagging_options=None,
				#live=True
			)
		with gr.Tab("HordeAI") as hordeai:
			gr.on(
				triggers=[generator.select],
				fn=get_last_prompt,
				inputs=None,
				outputs=[hordeai_prompt_input, hordeai_negative_prompt_input],
			)
			gr.Interface(
				run_hordeai_generation,
				[
					gr.TextArea(lines = 1, label="API Key",value=ANON_API_KEY),
					hordeai_prompt_input,
					hordeai_negative_prompt_input,
					gr.Dropdown(choices=hordeai_model_list.keys(), value='Deliberate 3.0', label='Model'),
					gr.Dropdown(choices=["k_dpmpp_2s_a", "k_lms", "k_heun", "k_heun", "k_euler", "k_euler_a",
										 "k_dpm_2", "k_dpm_2_a", "k_dpm_fast", "k_dpm_adaptive", "k_dpmpp_2s_a",
										 "k_dpmpp_2m", "dpmsolver", "k_dpmpp_sde", "lcm", "DDIM"
										 ], value="k_dpmpp_2s_a", label='Sampler'),


					gr.Slider(0, 100, step= 1, value=20, label="Steps", info="Choose between 1 and 100"),
					gr.Slider(0, 20, step= 0.1, value=7, label="CFG Scale", info="Choose between 1 and 20"),
					gr.Slider(0, 1024, step= 1, value=768, label="Width", info="Choose between 1 and 1024"),
					gr.Slider(0, 1024, step= 1, value=512, label="Height", info="Choose between 1 and 1024"),
					gr.Slider(0, 10, step= 1, value=2, label="Clipskip", info="Choose between 1 and 10"),
				]
				,outputs=gr.Image(label="Generated Image"), #"text",
				allow_flagging='never',
				flagging_options=None,
				#live=True
			)



if __name__ == "__main__":
	pq_ui.launch(inbrowser=True ) #share=True
