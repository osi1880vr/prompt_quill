# settings_manager.py
import globals
import gradio as gr
from settings.io import settings_io
from llm_fw import llm_interface_qdrant

class SettingsManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.settings_io = settings_io()
        self.interface = llm_interface_qdrant.get_interface()

    def load_preset_list(self):
        try:
            self.g.settings_data['preset_list'] = self.settings_io.load_preset_list()
            return gr.update(choices=self.g.settings_data['preset_list'], value=self.g.settings_data['selected_preset']), 'OK'
        except Exception as e:
            return gr.update(choices=[], value=''), str(e)

    def load_preset(self, name):
        try:
            self.g.settings_data = self.settings_io.load_preset(name)
            return 'OK'
        except Exception as e:
            return str(e)

    def save_preset(self, name):
        try:
            status = self.settings_io.save_preset(name, self.g.settings_data)
            return status
        except Exception as e:
            return str(e)

    def get_llm_settings(self):
        return (
            gr.update(choices=self.g.settings_data['collections_list'], value=self.g.settings_data['collection']),
            self.g.settings_data["LLM Model"],
            self.g.settings_data["embedding_model"],
            self.g.settings_data['Temperature'],
            self.g.settings_data['Context Length'],
            self.g.settings_data['GPU Layers'],
            self.g.settings_data['max output Tokens'],
            self.g.settings_data['top_k']
        )

    def set_model(self, collection, model, embedding_model, temperature, n_ctx, n_gpu_layers, max_tokens, top_k):
        self.g.settings_data['collection'] = collection
        self.g.settings_data['LLM Model'] = model
        self.g.settings_data['embedding_model'] = embedding_model
        self.g.settings_data['Temperature'] = temperature
        self.g.settings_data['Context Length'] = n_ctx
        self.g.settings_data['GPU Layers'] = n_gpu_layers
        self.g.settings_data['max output Tokens'] = max_tokens
        self.g.settings_data['top_k'] = top_k
        self.settings_io.write_settings(self.g.settings_data)
        return self.interface.change_model(self.g.settings_data['model_list'][model], temperature, n_ctx, max_tokens, n_gpu_layers, top_k)

    def set_neg_prompt(self, value):
        self.g.settings_data['negative_prompt'] = value
        self.settings_io.write_settings(self.g.settings_data)
        return 'Negative prompt saved'

    def set_rephrase_instruction(self, value):
        self.g.settings_data['rephrase_instruction'] = value
        self.settings_io.write_settings(self.g.settings_data)
        return 'Rephrase instruction saved'

    def get_prompt_template(self):
        self.interface.prompt_template = self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]
        return self.g.settings_data["prompt_templates"][self.g.settings_data["selected_template"]]

    def set_prompt_template_select(self, value):
        self.g.settings_data['selected_template'] = value
        self.settings_io.write_settings(self.g.settings_data)
        return self.g.settings_data["prompt_templates"][value]

    def set_prompt_template(self, selection, prompt_text):
        return_data = self.interface.set_prompt(prompt_text)
        self.g.settings_data["prompt_templates"][selection] = prompt_text
        self.settings_io.write_settings(self.g.settings_data)
        return return_data