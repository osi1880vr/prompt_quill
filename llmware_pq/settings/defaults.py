default = {
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

    'selected_template': 'prompt_template_a',
    'model_list': {
        'TheBloke/toxicqa-Llama2-7B-GGUF':
            {
                'type': 'deep_link',
                'name': 'TheBloke/toxicqa-Llama2-7B-GGUF',
                'repo_name': 'TheBloke/toxicqa-Llama2-7B-GGUF',
                'file': 'toxicqa-llama2-7b.Q5_K_M.gguf'
            },
        'thebloke/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf':
            {
                'type': 'deep_link',
                'name': 'thebloke/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf',
                'repo_name': 'TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF',
                'file': 'speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf'
            },

        'TheBloke/WestLake-7B-v2-GGUF':
            {
                'type': 'deep_link',
                'name': 'TheBloke/WestLake-7B-v2-GGUF',
                'repo_name': 'TheBloke/WestLake-7B-v2-GGUF',
                'file': 'westlake-7b-v2.Q4_K_M.gguf'
            },
        'TheBloke/Rosa_v2_7B-GGUF':
            {
                'type': 'deep_link',
                'name': 'TheBloke/Rosa_v2_7B-GGUF',
                'repo_name': 'TheBloke/Rosa_v2_7B-GGUF',
                'file': 'rosa_v2_7b.Q4_K_M.gguf'
            },
        'TheBloke/Panda-7B-v0.1-GGUF':
            {
                'type': 'deep_link',
                'name': 'TheBloke/Panda-7B-v0.1-GGUF',
                'repo_name': 'TheBloke/Panda-7B-v0.1-GGUF',
                'file': 'panda-7b-v0.1.Q4_K_M.gguf'
            },

        'thebloke/llama-2-13b-chat.Q5_K_M.gguf':
            {
                'type': 'deep_link',
                'name': 'thebloke/llama-2-13b-chat.Q5_K_M.gguf',
                'repo_name': 'TheBloke/Llama-2-13B-chat-GGUF',
                'file': 'llama-2-13b-chat.Q5_K_M.gguf'
            },
        'thebloke/solar-10.7b-instruct-v1.0-uncensored.Q5_K_M.gguf':
            {
                'type': 'deep_link',
                'name': 'thebloke/solar-10.7b-instruct-v1.0-uncensored.Q5_K_M.gguf',
                'repo_name': 'TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF',
                'file': 'solar-10.7b-instruct-v1.0-uncensored.Q5_K_M.gguf'
            },
        'thebloke/yarn-mistral-7b-128k.Q5_K_M.gguf':
            {
                'type': 'deep_link',
                'name': 'thebloke/yarn-mistral-7b-128k.Q5_K_M.gguf',
                'repo_name': 'TheBloke/Yarn-Mistral-7B-128k-GGUF',
                'file': 'yarn-mistral-7b-128k.Q5_K_M.gguf'
            },
        'thebloke/neural-chat-7b-v3-1.Q5_K_M.gguf':
            {
                'type': 'deep_link',
                'name': 'thebloke/neural-chat-7b-v3-1.Q5_K_M.gguf',
                'repo_name': 'TheBloke/neural-chat-7B-v3-1-GGUF',
                'file': 'neural-chat-7b-v3-1.Q5_K_M.gguf'
            },
        'thebloke/mistral-7b-instruct-v0.1.Q5_K_M.gguf':
            {
                'type': 'deep_link',
                'name': 'thebloke/mistral-7b-instruct-v0.1.Q5_K_M.gguf',
                'repo_name': 'TheBloke/neural-chat-7B-v3-1-GGUF',
                'file': 'mistral-7b-instruct-v0.1.Q5_K_M.gguf'
            },
        'llmware/dragon-mistral-7b-gguf':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-mistral-7b-gguf',
            },
        'llmware/dragon-yi-6b-v0':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-yi-6b-v0',
            },
        'llmware/dragon-red-pajama-7b-v0':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-red-pajama-7b-v0',
            },
        'llmware/dragon-deci-6b-v0':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-deci-6b-v0',
            },
        'llmware/dragon-mistral-7b-v0':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-mistral-7b-v0',
            },
        'llmware/dragon-falcon-7b-v0':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-falcon-7b-v0',
            },
        'llmware/dragon-llama-7b-v0':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-llama-7b-v0',
            },
        'llmware/dragon-deci-7b-v0':
            {
                'type': 'llmware',
                'name': 'llmware/dragon-deci-7b-v0',
            },
        'llmware/bling-sheared-llama-1.3b-0.1':
            {
                'type': 'llmware',
                'name': 'llmware/bling-sheared-llama-1.3b-0.1',
            },
        'llmware/bling-stable-lm-3b-4e1t-v0':
            {
                'type': 'llmware',
                'name': 'llmware/bling-stable-lm-3b-4e1t-v0',
            },
        'llmware/bling-sheared-llama-2.7b-0.1':
            {
                'type': 'llmware',
                'name': 'llmware/bling-sheared-llama-2.7b-0.1',
            },
        'llmware/bling-falcon-1b-0.1':
            {
                'type': 'llmware',
                'name': 'llmware/bling-falcon-1b-0.1',
            },
        'llmware/bling-red-pajamas-3b-0.1':
            {
                'type': 'llmware',
                'name': 'llmware/bling-red-pajamas-3b-0.1',
            },
        'llmware/bling-1b-0.1':
            {
                'type': 'llmware',
                'name': 'llmware/bling-1b-0.1',
            },
        'llmware/bling-cerebras-1.3b-0.1':
            {
                'type': 'llmware',
                'name': 'llmware/bling-cerebras-1.3b-0.1',
            },
        'llmware/bling-1.4b-0.1':
            {
                'type': 'llmware',
                'name': 'llmware/bling-1.4b-0.1',
            },
        'llmware/bling-phi-1_5-v0':
            {
                'type': 'llmware',
                'name': 'llmware/bling-phi-1_5-v0',
            },
        'llmware/bling-phi-2-v0':
            {
                'type': 'llmware',
                'name': 'llmware/bling-phi-2-v0',
            },
        'llmware/bling-tiny-llama-v0':
            {
                'type': 'llmware',
                'name': 'llmware/bling-tiny-llama-v0',
            }

    },

    'prompt_templates': {
        'prompt_template_a': {"blurb1": """Given the context information below and not prior knowledge, create a text to image prompt 
based on the context and the Query, don't mind if the context does not match the Query, 
still try to create a wonderful text to image prompt.You also take care of describing the 
scene, the lighting as well as the quality improving keywords
Query: beautiful female wizard in forest
Answer: cinematic photo, masterpiece, in the style of picasso, ((beautiful female wizard)), at dusk, standing in a mystical forest, surrounded by fireflies, wearing a long flowing dress with a starry pattern, and holding a glowing wand, magical and enchanting, on eye level, scenic, masterpiece
Query: beautiful female scientist on frozen lake
Answer: ultra high res, detailed, perfect face, ((beautiful female scientist)), in winter, wearing a warm fur coat, standing on a frozen lake with snow-capped mountains in the background, casting a spell with her hands, the ice cracking beneath her feet, stunning and majestic, on eye level, scenic, masterpiece
Query: beautiful female princess in a meadow
Answer: Best quality, masterpiece, realistic, ((beautiful female princess)), in spring, wearing a flower crown, standing in a blooming meadow, surrounded by butterflies, holding a staff with a crystal on top, the sun shining down on her, a symbol of nature's beauty and power, on eye level, scenic, masterpiece                               
Query: beautiful female witch on a castle rooftop
Answer: photorealistic, Professional photo, analog style, ((beautiful female witch)) at midnight, wearing a black cloak with a hood, standing on the rooftop of a castle, surrounded by stars and the moon, casting a spell with a dark aura, mysterious and powerful, on eye level, scenic, masterpiece""",
"blurb2": "Query: ",
"instruction": "\nAnswer: "
        }

    }

}
