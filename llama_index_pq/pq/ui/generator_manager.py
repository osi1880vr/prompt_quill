# generator_manager.py
import globals
import gradio as gr
import base64
from PIL import Image
from io import BytesIO
import time
import os
import shared
from settings.io import settings_io
from generators.hordeai.client import hordeai_client
from generators.automatics.client import automa_client
from llm_fw import llm_interface_qdrant

class GeneratorManager:
    def __init__(self):
        self.g = globals.get_globals()
        self.settings_io = settings_io()
        self.interface = llm_interface_qdrant.get_interface()
        self.automa_client = automa_client()
        self.out_dir_t2i = os.path.join('../api_out', 'txt2img')
        os.makedirs(self.out_dir_t2i, exist_ok=True)

    def set_automa_layerdiffuse(self, automa_layerdiffuse_enable):
        self.g.settings_data['automa']['automa_layerdiffuse_enable'] = automa_layerdiffuse_enable
        self.settings_io.write_settings(self.g.settings_data)

    def run_automa_interrogation_batch(self, image_filenames,url, save):

        all_response = ''
        output = ''
        n = 0
        for file in image_filenames:
            response = self.run_automa_interrogation(file[0],url)
            if all_response == '':
                all_response = response
            else:
                all_response = f'{all_response}\n{response}'
            output = f'{output}{response}\n{n} ---------\n'
            n += 1
            yield output

        if save:
            import time
            filename = f'{time.strftime("%Y%m%d-%H%M%S")}.txt'
            outfile = os.path.join(out_dir_i2t,filename)
            f = open(outfile,'a',encoding='utf8',errors='ignore')
            f.write(f'{all_response}\n')
            f.close()



    # Automatic 1111 / Forge methods
    def automa_switch_size(self, automa_width, automa_height):
        self.g.settings_data['automa']['automa_width'] = automa_height
        self.g.settings_data['automa']['automa_height'] = automa_width
        return automa_height, automa_width

    def automa_refresh(self):
        self.g.settings_data['automa']['automa_checkpoints'] = self.automa_client.get_checkpoints(self.g.settings_data['automa']['automa_url'])
        self.g.settings_data['automa']['automa_samplers'] = self.automa_client.get_samplers(self.g.settings_data['automa']['automa_url'])
        self.g.settings_data['automa']['automa_schedulers'] = self.automa_client.get_schedulers(self.g.settings_data['automa']['automa_url'])
        self.g.settings_data['automa']['automa_vaes'] = self.automa_client.get_vaes(self.g.settings_data['automa']['automa_url'])
        return (
            gr.update(choices=self.g.settings_data['automa']['automa_samplers'], value=self.g.settings_data['automa']['automa_sampler']),
            gr.update(choices=self.g.settings_data['automa']['automa_checkpoints'], value=self.g.settings_data['automa']['automa_checkpoint']),
            gr.update(choices=self.g.settings_data['automa']['automa_vaes'], value=self.g.settings_data['automa']['automa_vae']),
            gr.update(choices=self.g.settings_data['automa']['automa_schedulers'], value=self.g.settings_data['automa']['automa_scheduler'])
        )

    def run_automatics_generation(self, prompt, negative_prompt, sampler, checkpoint, steps, cfg, width, heigth, batch, n_iter, url, save, save_api, vae, clip_skip, automa_scheduler):
        self.g.running = True
        self.set_automa_settings(prompt, negative_prompt, sampler, checkpoint, steps, cfg, width, heigth, batch, n_iter, url, save, save_api, vae, clip_skip, self.g.settings_data['automa']['automa_new_forge'], automa_scheduler)
        self.g.last_prompt = prompt
        self.g.last_negative_prompt = negative_prompt

        response = self.automa_client.request_generation(prompt, negative_prompt, self.g.settings_data)
        images = []
        if response != '':
            for index, image in enumerate(response.get('images')):
                img = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
                if save:
                    save_path = os.path.join(self.out_dir_t2i, f'txt2img-{self.timestamp()}-{index}.png')
                    self.automa_client.decode_and_save_base64(image, save_path)
                images.append(img)
        yield images

    def set_automa_settings(self, prompt, negative_prompt, sampler, checkpoint, steps, cfg, width, heigth, batch, n_iter, url, save, save_api, vae, clip_skip, automa_new_forge, automa_scheduler):
        self.g.last_prompt = prompt
        self.g.last_negative_prompt = negative_prompt
        self.g.settings_data['automa']['automa_sampler'] = sampler
        self.g.settings_data['automa']['automa_steps'] = steps
        self.g.settings_data['automa']['automa_cfg_scale'] = cfg
        self.g.settings_data['automa']['automa_width'] = width
        self.g.settings_data['automa']['automa_height'] = heigth
        self.g.settings_data['automa']['automa_batch'] = batch
        self.g.settings_data['automa']['automa_n_iter'] = n_iter
        self.g.settings_data['automa']['automa_url'] = url
        self.g.settings_data['automa']['automa_save'] = save
        self.g.settings_data['automa']['automa_save_on_api_host'] = save_api
        self.g.settings_data['automa']['automa_checkpoint'] = checkpoint
        self.g.settings_data['automa']['automa_vae'] = vae
        self.g.settings_data['automa']['automa_clip_skip'] = clip_skip
        self.g.settings_data['automa']['automa_new_forge'] = automa_new_forge
        self.g.settings_data['automa']['automa_scheduler'] = automa_scheduler
        self.settings_io.write_settings(self.g.settings_data)

    def run_automa_interrogation(self, image_filename, url):
        with open(image_filename, mode='rb') as fp:
            base64_image = base64.b64encode(fp.read()).decode('utf-8')
        response = self.automa_client.request_interrogation(base64_image, url)
        self.g.context_prompt = response
        return response

    def automa_get_last_prompt(self):
        return (
            self.g.last_prompt,
            self.g.last_negative_prompt,
            gr.update(choices=self.g.settings_data['automa']['automa_samplers'], value=self.g.settings_data['automa']['automa_sampler']),
            self.g.settings_data['automa']['automa_steps'],
            self.g.settings_data['automa']['automa_cfg_scale'],
            self.g.settings_data['automa']['automa_width'],
            self.g.settings_data['automa']['automa_height'],
            self.g.settings_data['automa']['automa_batch'],
            self.g.settings_data['automa']['automa_n_iter'],
            self.g.settings_data['automa']['automa_url'],
            self.g.settings_data['automa']['automa_save'],
            self.g.settings_data['automa']['automa_save_on_api_host'],
            gr.update(choices=self.g.settings_data['automa']['automa_checkpoints'], value=self.g.settings_data['automa']['automa_checkpoint'])
        )

    # HordeAI methods
    def hordeai_get_last_prompt(self):
        return (
            self.g.last_prompt,
            self.g.last_negative_prompt,
            self.g.settings_data['horde']['horde_api_key'],
            self.g.settings_data['horde']['horde_model'],
            self.g.settings_data['horde']['horde_sampler'],
            self.g.settings_data['horde']['horde_steps'],
            self.g.settings_data['horde']['horde_cfg_scale'],
            self.g.settings_data['horde']['horde_width'],
            self.g.settings_data['horde']['horde_height'],
            self.g.settings_data['horde']['horde_clipskip']
        )

    def run_hordeai_generation(self, prompt, negative_prompt, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
        self.set_hordeai_settings(api_key, model, sampler, steps, cfg, width, heigth, clipskip)
        client = hordeai_client()
        return client.request_generation(api_key=api_key, prompt=prompt, negative_prompt=negative_prompt,
                                         sampler=sampler, model=model, steps=steps, cfg=cfg, width=width, heigth=heigth,
                                         clipskip=clipskip)

    def set_hordeai_settings(self, api_key, model, sampler, steps, cfg, width, heigth, clipskip):
        self.g.settings_data['horde']['horde_api_key'] = api_key
        self.g.settings_data['horde']['horde_model'] = model
        self.g.settings_data['horde']['horde_sampler'] = sampler
        self.g.settings_data['horde']['horde_steps'] = steps
        self.g.settings_data['horde']['horde_cfg_scale'] = cfg
        self.g.settings_data['horde']['horde_width'] = width
        self.g.settings_data['horde']['horde_height'] = heigth
        self.g.settings_data['horde']['horde_clipskip'] = clipskip
        self.settings_io.write_settings(self.g.settings_data)

    # Top-level method
    def all_get_last_prompt(self):
        if not self.g.settings_data['automa']['automa_checkpoints']:
            self.g.settings_data['automa']['automa_checkpoints'] = self.automa_client.get_checkpoints(self.g.settings_data['automa']['automa_url'])
            self.g.settings_data['automa']['automa_samplers'] = self.automa_client.get_samplers(self.g.settings_data['automa']['automa_url'])
            self.g.settings_data['automa']['automa_vaes'] = self.automa_client.get_vaes(self.g.settings_data['automa']['automa_url'])
            if self.g.settings_data['automa']['automa_new_forge']:
                self.g.settings_data['automa']['automa_schedulers'] = self.automa_client.get_schedulers(self.g.settings_data['automa']['automa_url'])
        return (
            self.g.last_prompt,
            self.g.last_negative_prompt,
            self.g.settings_data['horde']['horde_api_key'],
            self.g.settings_data['horde']['horde_model'],
            self.g.settings_data['horde']['horde_sampler'],
            self.g.settings_data['horde']['horde_steps'],
            self.g.settings_data['horde']['horde_cfg_scale'],
            self.g.settings_data['horde']['horde_width'],
            self.g.settings_data['horde']['horde_height'],
            self.g.settings_data['horde']['horde_clipskip'],
            self.g.last_prompt,
            self.g.last_negative_prompt,
            gr.update(choices=self.g.settings_data['automa']['automa_samplers'], value=self.g.settings_data['automa']['automa_sampler']),
            self.g.settings_data['automa']['automa_steps'],
            self.g.settings_data['automa']['automa_cfg_scale'],
            self.g.settings_data['automa']['automa_width'],
            self.g.settings_data['automa']['automa_height'],
            self.g.settings_data['automa']['automa_batch'],
            self.g.settings_data['automa']['automa_n_iter'],
            self.g.settings_data['automa']['automa_url'],
            self.g.settings_data['automa']['automa_save'],
            self.g.settings_data['automa']['automa_save_on_api_host'],
            gr.update(choices=self.g.settings_data['automa']['automa_checkpoints'], value=self.g.settings_data['automa']['automa_checkpoint']),
            gr.update(choices=self.g.settings_data['automa']['automa_vaes'], value=self.g.settings_data['automa']['automa_vae']),
            self.g.settings_data['automa']['automa_clip_skip'],
            gr.update(choices=self.g.settings_data['automa']['automa_schedulers'], value=self.g.settings_data['automa']['automa_scheduler'])
        )

    # Helper method
    def timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")