import globals
import gradio as gr
from generators.automatics.client import automa_client
from settings.io import settings_io



adetailer_choices = ["face_yolov8n.pt",
                     "face_yolov8m.pt",
                     "face_yolov8n_v2.pt",
                     "face_yolov9c.pt",
                     "hand_yolov8n.pt",
                     "hand_yolov8s.pt",
                     "female-breast-v4.7.pt",
                     "vagina-v4.1.pt"]

class UiShare:


    def __init__(self):
        self.g = globals.get_globals()
        self.automa_client = automa_client()
        self.adetailer_checkpoints = []
        self.settings_io = settings_io()

    def get_automa_checkpoints(self):
        checkpoints = self.automa_client.get_checkpoints(self.g.settings_data['automa']['automa_url'])
        if checkpoints != -1:
            if self.g.settings_data['automa']['automa_checkpoint'] == '':
                self.g.settings_data['automa']['automa_checkpoint'] = checkpoints[0]
            return checkpoints
        else:
            return []


    def refresh_adetailer_checkpoints(self, chpt, number):
        checkpoints = self.get_automa_checkpoints()
        out_array = ['Same']
        out_array.extend(checkpoints)

        return gr.update(choices=out_array, value=self.g.settings_data['automa'][f'automa_ad_checkpoint_{number}'])


    def set_automa_adetailer(self,
                             number,
                             automa_adetailer_enable,
                             automa_ad_prompt,
                             automa_ad_negative_prompt,
                             automa_ad_checkpoint,
                             automa_ad_use_inpaint_width_height,
                             automa_ad_model,
                             automa_ad_denoising_strength,
                             automa_ad_clip_skip,
                             automa_ad_confidence):

        self.g.settings_data['automa'][f'automa_adetailer_enable_{number}'] = automa_adetailer_enable
        self.g.settings_data['automa'][f'automa_ad_prompt_{number}'] = automa_ad_prompt
        self.g.settings_data['automa'][f'automa_ad_negative_prompt_{number}'] = automa_ad_negative_prompt
        self.g.settings_data['automa'][f'automa_ad_checkpoint_{number}'] = automa_ad_checkpoint
        self.g.settings_data['automa'][f'automa_ad_use_inpaint_width_height_{number}'] = automa_ad_use_inpaint_width_height
        self.g.settings_data['automa'][f'automa_ad_model_{number}'] = automa_ad_model
        self.g.settings_data['automa'][f'automa_ad_denoising_strength_{number}'] = automa_ad_denoising_strength
        self.g.settings_data['automa'][f'automa_ad_clip_skip_{number}'] = automa_ad_clip_skip
        self.g.settings_data['automa'][f'automa_ad_confidence_{number}'] = automa_ad_confidence

        self.settings_io.write_settings(self.g.settings_data)

    def check_variables(self, number):
        if f'automa_adetailer_enable_{number}' not in self.g.settings_data:
            self.g.settings_data['automa'][f'automa_adetailer_enable_{number}'] = False
            self.g.settings_data['automa'][f'automa_ad_checkpoint_{number}'] = 'None'
            self.g.settings_data['automa'][f'automa_ad_use_inpaint_width_height_{number}'] = False
            self.g.settings_data['automa'][f'automa_ad_model_{number}'] = 'face_yolov8m.pt'
            self.g.settings_data['automa'][f'automa_ad_denoising_strength_{number}'] = 0.2
            self.g.settings_data['automa'][f'automa_ad_clip_skip_{number}'] = 1
            self.g.settings_data['automa'][f'automa_ad_confidence_{number}'] = 0.7
            self.settings_io.write_settings(self.g.settings_data)

    def generate_ad_block(self, number):
        self.check_variables(number)
        with (gr.Row()):
            with gr.Column(scale=3):

                automa_adetailer_enable = gr.Checkbox(label="Enable Adetailer",
                                                      value=self.g.settings_data['automa'][f'automa_adetailer_enable_{number}'])


                automa_ad_prompt = gr.TextArea(lines=1, label="Prompt", value=self.g.settings_data['automa'][f'automa_ad_prompt_{number}'])
                automa_ad_negative_prompt = gr.TextArea(lines=1, label="negative Prompt", value=self.g.settings_data['automa'][f'automa_ad_negative_prompt_{number}'])


                automa_ad_use_inpaint_width_height = gr.Checkbox(label="Use inpaint with height",
                                                                 value=self.g.settings_data['automa'][f'automa_ad_use_inpaint_width_height_{number}'])

                automa_ad_model = gr.Dropdown(
                    choices=adetailer_choices, value=self.g.settings_data['automa'][f'automa_ad_model_{number}'], label='Model', allow_custom_value=True)

                automa_ad_checkpoint = gr.Dropdown(
                    choices=self.g.settings_data['automa']['automa_checkpoints'],
                    value=self.g.settings_data['automa'][f'automa_ad_checkpoint_{number}'],
                    label='Checkpoint',
                    allow_custom_value=True)

                automa_ad_denoising_strength = gr.Slider(0, 1, step=0.1,
                                                         value=self.g.settings_data['automa'][f'automa_ad_denoising_strength_{number}'],
                                                         label="Denoising strength",
                                                         info="Denoising strength 0-1.")

                automa_ad_clip_skip = gr.Slider(1, 5, step=1, value=self.g.settings_data['automa'][f'automa_ad_clip_skip_{number}'],
                                                label="Clipskip",
                                                info="Clipskip 1-5.")

                automa_ad_confidence = gr.Slider(0, 1, step=0.1, value=self.g.settings_data['automa'][f'automa_ad_confidence_{number}'],
                                                 label="Confidence",
                                                 info="Level of confidence 0-1.")

            with gr.Column(scale=1):
                adetailer_refresh_button = gr.Button('Refresh')

                adetailer_refresh_button.click(lambda checkpoint: self.refresh_adetailer_checkpoints(checkpoint, number),
                                               inputs=None,
                                               outputs=[automa_ad_checkpoint])

            # Generate the gr.on block
            fn = lambda enable, ad_prompt, ad_negative_prompt, ad_checkpoint, use_inpaint, model, denoising, clip_skip, confidence, restore_face, steps: \
                self.set_automa_adetailer(number,            # Static number argument
                                          enable,            # automa_adetailer_enable from Gradio input
                                          ad_prompt,
                                          ad_negative_prompt,
                                          ad_checkpoint,     # automa_ad_checkpoint from Gradio input
                                          use_inpaint,       # automa_ad_use_inpaint_width_height from Gradio input
                                          model,             # automa_ad_model from Gradio input
                                          denoising,         # automa_ad_denoising_strength from Gradio input
                                          clip_skip,         # automa_ad_clip_skip from Gradio input
                                          confidence         # automa_ad_confidence from Gradio input
                                        )


            gr.on(
                triggers=[automa_adetailer_enable.change,
                          automa_ad_prompt.change,
                          automa_ad_negative_prompt.change,
                          automa_ad_checkpoint.change,
                          automa_ad_use_inpaint_width_height.change,
                          automa_ad_model.change,
                          automa_ad_denoising_strength.change,
                          automa_ad_clip_skip.change,
                          automa_ad_confidence.change],
                fn=fn,
                inputs=[automa_adetailer_enable,
                        automa_ad_prompt,
                        automa_ad_negative_prompt,
                        automa_ad_checkpoint,
                        automa_ad_use_inpaint_width_height,
                        automa_ad_model,
                        automa_ad_denoising_strength,
                        automa_ad_clip_skip,
                        automa_ad_confidence],
                outputs=None
            )


        #return #automa_adetailer_enable, automa_ad_use_inpaint_width_height, automa_ad_model, automa_ad_checkpoint, automa_ad_denoising_strength, automa_ad_clip_skip, automa_ad_confidence

