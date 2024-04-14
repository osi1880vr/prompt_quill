default = {

    'translate': False,
    'batch': False,
    "LLM Model": 'TheBloke/Panda-7B-v0.1-GGUF',
    "Temperature": 0.0,
    "Context Length": 3900,
    "GPU Layers": 50,
    "max output Tokens": 200,
    "top_k": 5,
    'Instruct Model': False,

    'civitai_Air': 'urn:air:sd1:checkpoint:civitai:4201@130072',
    "civitai_Steps": 20,
    "civitai_CFG Scale": 7,
    "civitai_Width": 512,
    "civitai_Height": 512,
    "civitai_Clipskip": 2,

    'horde_api_key': "0000000000",
    'horde_Model': 'Deliberate 3.0',
    'horde_Sampler': "k_dpmpp_2s_a",
    "horde_Steps": 20,
    "horde_CFG Scale": 7,
    "horde_Width": 768,
    "horde_Height": 512,
    "horde_Clipskip": 2,

    'automa_Sampler': "DPM++ 2M Karras",
    "automa_Steps": 20,
    "automa_CFG Scale": 7,
    "automa_Width": 768,
    "automa_Height": 512,
    "automa_url": "http://localhost:7860",
    "automa_save": True,

    'selected_template': 'prompt_template_b',
    'model_list': {

        'thebloke/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf':
            {
                'name': 'thebloke/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf',
                'path': 'https://huggingface.co/TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF/resolve/main/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf'
            },
        'TheBloke/openchat-3.5-0106-GGUF':
            {
                'name': 'TheBloke/openchat-3.5-0106-GGUF',
                'path': 'https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q8_0.gguf'
            },

        'thebloke/llama-2-13b-chat.Q5_K_M.gguf':
            {
                'name': 'thebloke/llama-2-13b-chat.Q5_K_M.gguf',
                'path': 'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf'
            },
        'TheBloke/WestLake-7B-v2-GGUF':
            {
                'name': 'TheBloke/WestLake-7B-v2-GGUF',
                'path': 'https://huggingface.co/TheBloke/WestLake-7B-v2-GGUF/resolve/main/westlake-7b-v2.Q4_K_M.gguf'
            },
        'TheBloke/Rosa_v2_7B-GGUF':
            {
                'name': 'TheBloke/Rosa_v2_7B-GGUF',
                'path': 'https://huggingface.co/TheBloke/Rosa_v2_7B-GGUF/resolve/main/rosa_v2_7b.Q4_K_M.gguf'
            },
        'TheBloke/Panda-7B-v0.1-GGUF':
            {
                'name': 'TheBloke/Panda-7B-v0.1-GGUF',
                'path': 'https://huggingface.co/TheBloke/Panda-7B-v0.1-GGUF/resolve/main/panda-7b-v0.1.Q4_K_M.gguf'
            },
        'TheBloke/Sonya-7B-GGUF':
            {
                'name': 'TheBloke/Sonya-7B-GGUF',
                'path': 'https://huggingface.co/TheBloke/Sonya-7B-GGUF/resolve/main/sonya-7b.Q4_K_M.gguf'
            },
        'TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF':
            {
                'name': 'TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF',
                'path': 'https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF/resolve/main/dolphin-2.6-mistral-7b-dpo.Q4_K_M.gguf'
            },
        'TheBloke/Lelantos-7B-GGUF':
            {
                'name': 'TheBloke/Lelantos-7B-GGUF',
                'path': 'https://huggingface.co/TheBloke/Lelantos-7B-GGUF/resolve/main/lelantos-7b.Q4_K_M.gguf'
            },


    },

    'prompt_templates': {

        'prompt_template_a': """Context information is below.        
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
create a text to image prompt based on the context and the Query, don't mind if the context does not match the Query, still try to create a wonderfull text to image prompt.
You also take care of describing the scene, the lighting as well as the quality improving keywords. the length of the prompt may vary depending on the complexity of the context.
Query: beautiful female wizard in forest
Answer: cinematic photo, masterpiece, in the style of picasso, ((beautiful female wizard)), at dusk, standing in a mystical forest, surrounded by fireflies, wearing a long flowing dress with a starry pattern, and holding a glowing wand, magical and enchanting, on eye level, scenic, masterpiece,
Query: beautiful female scientist on frozen lake
Answer: ultra high res, detailed, perfect face, ((beautiful female scientist)), in winter, wearing a warm fur coat, standing on a frozen lake with snow-capped mountains in the background, casting a spell with her hands, the ice cracking beneath her feet, stunning and majestic, on eye level, scenic, masterpiece
Query: beautiful female princess in a meadow
Answer: Best quality, masterpiece, realistic, ((beautiful female princess)), in spring, wearing a flower crown, standing in a blooming meadow, surrounded by butterflies, holding a staff with a crystal on top, the sun shining down on her, a symbol of nature's beauty and power, on eye level, scenic, masterpiece
Query: beautiful female witch on a castle rooftop
Answer: photorealistic, Professional photo, analog style, ((beautiful female witch)), at midnight, wearing a black cloak with a hood, standing on the rooftop of a castle, surrounded by stars and the moon, casting a spell with a dark aura, mysterious and powerful, on eye level, scenic, masterpiece
Query: {query_str}
Answer:""",

        'prompt_template_b': """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, 
create a text to image prompt based on the context and the Query, don't mind if the context does not match the Query, still try to create a wonderfull text to image prompt.
You also take care of describing the scene, the lighting as well as the quality improving keywords
Query: translation in major world languages, machinery of translation in various specializations, cyberpunk style
Answer: Cyberpunk-style illustration, featuring a futuristic translation device in various specializations, set against a backdrop of neon-lit cityscape. The device, adorned with glowing circuits and cybernetic enhancements, showcases its capabilities in translating languages such as English, Mandarin, Spanish, French, and Arabic. The scene is illuminated by the warm glow of streetlights and the pulsing neon signs, casting intricate shadows on the surrounding machinery. The artwork is rendered in high-quality, vivid colors, with detailed textures and sharp lines, evoking the gritty yet mesmerizing atmosphere of the cyberpunk world.
Query: a man walking moon
Answer: cinematic photo, high resolution, masterpiece, ((man walking on the moon)), in a surrealistic setting, with the moon's surface featuring vivid colors and abstract patterns, the man wearing a spacesuit with an astronaut helmet, the American flag planted on the moon's surface in the background, the Earth visible in the distance, the scene illuminated by the moon's glow, on eye level, scenic, masterpiece.
Query: a female witch
Answer: The scene unfolds with the beautiful female witch standing on the rooftop of an ancient castle, her black cloak billowing in the wind as she gazes out at the breathtaking view below. The midnight sky above is filled with stars and the full moon casts an eerie glow on the witch's face, highlighting her enchanting beauty. She stands tall, her hood framing her face, casting a spell with her outstretched hand, her dark aura swirling around her. The castle walls, adorned with intricate carvings and gargoyles, stand tall behind her, adding to the mystical atmosphere of the scene. The wind whispers through the rooftop's crenellations, creating an eerie yet captivating soundtrack for this magical moment. The quality of the photo is exceptional, with every detail of the witch's cloak, the castle's architecture, and the night sky captured in stunning clarity. This cinematic masterpiece invites the viewer to step into the world of magic and mystery, leaving them in awe of the beautiful female witch standing on the castle rooftop under the starry sky.
Query: artifical intelligence and human
Answer: High-quality digital art, blending fantasy and reality, ((artificial intelligence)) and (((human))), in a futuristic cityscape, an AI robot with glowing circuits standing alongside a confident, well-dressed human, both exuding intelligence and grace, the AI with a sleek metal body and the human with impeccable style, the cityscape filled with advanced technology and vibrant colors, dynamic lighting, surreal and thought-provoking, on eye level, scenic, masterpiece.
Query: futuristic combat zone
Answer: cinematic photo, masterpiece, in the style of Blade Runner, futuristic combat zone, at dusk, showcasing a high-tech battlefield with neon lights illuminating the scene, filled with advanced mechs and soldiers engaged in an intense fight, the air filled with stunning lighting effects, on eye level, dramatic, masterpiece, ultra high resolution, dynamic anime-style fight scene, with a focus on the sleek design of the combat gear and the fluidity of the movements, capturing the essence of sci-fi action in a visually stunning manner.
Query: {query_str}
Answer: "
""",

        'custom_template': ''

    },
    'negative_prompt': """out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"""

}
