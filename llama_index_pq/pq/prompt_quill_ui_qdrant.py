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
import warnings
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics via env var
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=info, 1=warn, 2=error, 3=fatal

# Suppress TensorFlow deprecation warnings more aggressively
warnings.filterwarnings(
	"ignore",
	category=DeprecationWarning,
	module="tf_keras.*"  # Catch all tf_keras submodules
)

warnings.filterwarnings(
	"ignore",
	message="The value passed into gr.Dropdown.*",
	category=UserWarning,
	module="gradio.components.dropdown"
)

import globals
from settings.io import settings_io

g = globals.get_globals()
g.settings_data = settings_io().load_settings()
g.settings_data['automa']['automa_checkpoints'] = []  # bad hack for now, this should later be updateable via a button
g.settings_data['automa']['automa_samplers'] = []



from ui.ui_share import UiShare

from llm_fw.llama_cpp_hijack import llama_cpp_hijack

hijack = llama_cpp_hijack()

from generators.aesthetic import score
from style import style


import gradio as gr


import importlib

from extension_base import PromptQuillExtension  # Assuming this exists already



#import the UI stuff
from ui.ui_codes import ui_staff, ui_actions
from ui.sailing_manager import SailingManager  # New import
from ui.settings_manager import SettingsManager  # New import
from ui.model_testing_manager import ModelTestingManager  # New import
from ui.deep_dive_manager import DeepDiveManager  # New import
from ui.chat_manager import ChatManager  # New import
from ui.file2file_manager import File2FileManager  # New import
from ui.generator_manager import GeneratorManager  # New import
from ui.interrogation_manager import InterrogationManager  # New import
from ui.iti_manager import ITIManager  # New


from ui.ui_chat import setup_chat_tab
from ui.ui_sailing import setup_sailing_tab
from ui.ui_generator import setup_generator_tab
from ui.ui_interrogation import setup_interrogation_tab
from ui.ui_model_testing import setup_model_testing_tab
from ui.ui_deep_dive import setup_deep_dive_tab
from ui.ui_image_scoring import setup_image_scoring_tab
from ui.ui_file2file import setup_file2file_tab
from ui.ui_settings import setup_settings_tab
from ui.ui_wildcards import setup_wildcards_tab  # Updated import
from ui.ui_iti import setup_iti_tab  # New

css = style

ui = ui_staff()
ui_share = UiShare()
ui_code = ui_actions()
sailing_manager = SailingManager()  # New instantiation
settings_manager = SettingsManager()  # New instantiation
model_testing_manager = ModelTestingManager()  # New instantiation
deep_dive_manager = DeepDiveManager()  # New instantiation
chat_manager = ChatManager()  # New instantiation
file2file_manager = File2FileManager()  # New instantiation
generator_manager = GeneratorManager()  # New instantiation
interrogation_manager = InterrogationManager()  # New instantiation
iti_manager = ITIManager()  # New instantiation—no () in tab!

image_score = score.aestetic_score()

max_top_k = 50
textboxes = []

def load_extensions():
	extensions = {}
	extension_dir = "extensions"
	if not os.path.exists(extension_dir):
		os.makedirs(extension_dir)
		print(f"Created {extension_dir} directory")
	print(f"Looking for extensions in {extension_dir}...")
	for dir_name in os.listdir(extension_dir):
		dir_path = os.path.join(extension_dir, dir_name)
		if os.path.isdir(dir_path) and not dir_name.startswith('__') and 'examples' not in dir_name:
			print(f"Loading extension: {dir_name}")
			try:
				module = importlib.import_module(f"extensions.{dir_name}")
				for attr_name in dir(module):
					attr = getattr(module, attr_name)
					if (isinstance(attr, type) and
							issubclass(attr, PromptQuillExtension) and
							attr != PromptQuillExtension):
						extension_instance = attr()
						extensions[extension_instance.name] = extension_instance
						print(f"Loaded {extension_instance.name} from {dir_name}")
			except Exception as e:
				print(f"Error loading {dir_name}: {e}")
	print(f"Found extensions: {list(extensions.keys())}")
	return extensions

with gr.Blocks(css=css, title='Prompt Quill') as pq_ui:
	with gr.Row():
		# Image element (adjust width as needed)
		gr.Image(os.path.join(os.getcwd(), "logo/pq_v_small.jpg"), width="20vw", show_label=False,
				 show_download_button=False, container=False, elem_classes="gr-image", )
		# Title element (adjust font size and styling with CSS if needed)
		gr.Markdown("**Prompt Quill 2.0**", elem_classes="app-title")  # Add unique ID for potential CSS styling

	with gr.Tab("Chat") as chat:
		setup_chat_tab(chat, ui, chat_manager)  # Use chat_manager instead of ui_code
	with gr.Tab("Sail the data ocean") as sailor:
		setup_sailing_tab(sailor, sailing_manager)  # Pass sailing_manager instead of ui_code
	with gr.Tab("ItTtI") as iti:  # Typo fixed!
		iti_components = setup_iti_tab(iti, iti_manager, generator_manager, ui_code)  # No ui—lean!
	with gr.Tab("Generator") as generator:
		generator_components = setup_generator_tab(generator, ui, generator_manager, ui_share)  # Use generator_manager
	with gr.Tab("Interrogation") as interrogation:
		interrogation_components = setup_interrogation_tab(interrogation, interrogation_manager)  # Use interrogation_manager
	with gr.Tab("Wildcards"):
		setup_wildcards_tab()  # New tab
	with gr.Tab("Model testing") as model_test:
		model_test_components = setup_model_testing_tab(model_test, model_testing_manager)  # Use model_testing_manager
	with gr.Tab("Deep Dive") as deep_dive:
		deep_dive_components = setup_deep_dive_tab(deep_dive, deep_dive_manager, max_top_k)  # Use deep_dive_manager
	with gr.Tab("File2File") as batch_run:
		file2file_components = setup_file2file_tab(batch_run, file2file_manager)  # Use file2file_manager
	with gr.Tab("Image Scoring"):
		image_scoring_components = setup_image_scoring_tab(image_score)
	with gr.Tab("Settings"):
		settings_components = setup_settings_tab(settings_manager)  # Use settings_manager

	extensions = load_extensions()
	tab_map = {
		"Chat": chat,
		"Sail the data ocean": sailor,
		"ItTtI": iti,
		"Generator": generator,
		"Interrogation": interrogation,
		"Model testing": model_test,
		"Deep Dive": deep_dive,
		"File2File": batch_run,
		"Image Scoring": None,
		"Settings": None,
	}
	for ext_name, extension in extensions.items():
		print(f"Setting up {ext_name}...")
		if extension.tab_name and extension.tab_name in tab_map:
			tab = tab_map[extension.tab_name]
			if extension.tab_name == "Chat":
				extension.setup(tab, ui, chat_manager)
				print(f"Added {ext_name} to Chat tab")
			# ... add other tab cases as needed ...
		else:
			print(f"Building new tab for {ext_name}...")
			with gr.Tab(extension.name) as new_tab:
				extension.setup(new_tab)
			print(f"New tab {ext_name} is ready")


if __name__ == "__main__":
	print('If you see a message about : The name tf.losses.sparse_softmax_cross_entropy is deprecated..... you can ignore it, its out of my control and even the very latest version spill that warning.')

	server_name = "localhost"
	if os.getenv("SERVER_NAME") is not None:
		server_name = os.getenv("SERVER_NAME")

	pq_ui.launch(favicon_path='logo/favicon32x32.ico',
				 inbrowser=True,
				 server_name=server_name,
				 server_port=49152)  # share=True
