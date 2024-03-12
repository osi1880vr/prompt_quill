
from horde_sdk import ANON_API_KEY
from horde_sdk.ai_horde_api import KNOWN_SAMPLERS
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPISimpleClient
from horde_sdk.ai_horde_api.apimodels import ImageGenerateAsyncRequest, ImageGenerationInputPayload, LorasPayloadEntry
import json
import os


class hordeai_models:
    def read_model_list(self):

        f = open(os.path.join(os.getcwd(),'generators','hordeai','stable_diffusion.json'),'r')
        json_string = f.read()
        f.close()
        return json.loads(json_string)


class hordeai_client:

    def __init__(self):
        self.samplers = {
            'k_lms': KNOWN_SAMPLERS.k_lms ,
            'k_heun': KNOWN_SAMPLERS.k_heun,
            'k_euler': KNOWN_SAMPLERS.k_euler,
            'k_euler_a': KNOWN_SAMPLERS.k_euler_a,
            'k_dpm_2': KNOWN_SAMPLERS.k_dpm_2,
            'k_dpm_2_a': KNOWN_SAMPLERS.k_dpm_2_a,
            'k_dpm_fast': KNOWN_SAMPLERS.k_dpm_fast,
            'k_dpm_adaptive': KNOWN_SAMPLERS.k_dpm_adaptive,
            'k_dpmpp_2s_a': KNOWN_SAMPLERS.k_dpmpp_2s_a,
            'k_dpmpp_2m': KNOWN_SAMPLERS.k_dpmpp_2m,
            'dpmsolver': KNOWN_SAMPLERS.dpmsolver,
            'k_dpmpp_sde': KNOWN_SAMPLERS.k_dpmpp_sde,
            'lcm': KNOWN_SAMPLERS.lcm,
        'DDIM' : "DDIM"}



    def get_annon_api_key(self):
        return ANON_API_KEY

    def get_generation_dict(self, api_key, prompt, negative_prompt, sampler, model, steps, cfg, width, heigth, clipskip):

        sampler = self.samplers[sampler]

        prompt = f'{prompt}###{negative_prompt}'

        return ImageGenerateAsyncRequest(
            apikey=api_key,
            params=ImageGenerationInputPayload(
                sampler_name=sampler,
                cfg_scale=cfg,
                width=width,
                height=heigth,
                karras=False,
                hires_fix=False,
                clip_skip=clipskip,
                steps=steps,
                nsfw=True,
                # loras=[
                #     LorasPayloadEntry(
                #         name="GlowingRunesAI",
                #         model=1,
                #         clip=1,
                #         inject_trigger="any",  # Get a random color trigger
                #     ),
                #],
                n=1,
            ),
            prompt=prompt,
            models=[model],
        )

    def request_generation(self, api_key, prompt, negative_prompt, sampler, model, steps, cfg, width, heigth, clipskip):
        simple_client = AIHordeAPISimpleClient()
        status_response, job_id = simple_client.image_generate_request(self.get_generation_dict(api_key, prompt, negative_prompt, sampler, model, steps, cfg, width, heigth, clipskip))


        if len(status_response.generations) == 0:
            return -1



        if len(status_response.generations) == 1:
            image = simple_client.download_image_from_generation(status_response.generations[0])
            return image
        else:
            for i, generation in enumerate(status_response.generations):
                image = simple_client.download_image_from_generation(generation)
                return image
