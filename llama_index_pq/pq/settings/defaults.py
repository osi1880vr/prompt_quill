default = {
	'translate': False,
	'batch': False,
	'summary': False,
	"Temperature": 0.0,
	"Context Length": 3900,
	"GPU Layers": 50,
	"max output Tokens": 200,
	"top_k": 5,
	"Instruct Model": False,
	"collection": 'prompts_large_meta',
	"collections_list": ['prompts_large_meta'],

	"rephrase_instruction": """create a text to image prompt based on the context and the query,
You mix a new prompt based on the context and the query. The query is just adding a detail to the original context""",

	"preset_list": [],
	"selected_preset": '',

	'horde_api_key': "0000000000",
	'horde_Model': 'Deliberate 3.0',
	'horde_Sampler': "k_dpmpp_2s_a",
	"horde_Steps": 20,
	"horde_CFG Scale": 7,
	"horde_Width": 768,
	"horde_Height": 512,
	"horde_Clipskip": 2,

	'automa_Sampler': "DPM++ 2M Karras",
	'automa_Checkpoint': '',
	"automa_Steps": 20,
	"automa_CFG Scale": 7,
	"automa_Width": 768,
	"automa_Height": 512,
	"automa_url": "http://localhost:7860",
	"automa_save": True,
	"automa_batch": 1,
	"automa_n_iter": 1,
	"automa_save_on_api_host": False,
	"automa_adetailer_enable": False,
	'automa_ad_use_inpaint_width_height': False,
	'automa_ad_model': 'face_yolov8n.pt',
	'automa_ad_denoising_strength': 0.2,
	'automa_ad_clip_skip': 1,
	'automa_ad_confidence': 0.7,
	'automa_checkpoints': [],
	'automa_samplers': [],

	"sail_text": "",
	"sail_width": 10,
	"sail_depth": 10,
	"sail_generate": False,
	"sail_target": True,
	"sail_sinus": False,
	"sail_sinus_freq": 0.1,
	"sail_sinus_range": 10,
	"sail_add_style": False,
	"sail_style": "",
	"sail_add_search": False,
	"sail_search": "",
	"sail_max_gallery_size": 6,
	"sail_summary": False,
	"sail_rephrase_prompt": "",
	"sail_rephrase": False,
	"sail_gen_rephrase": False,
	"sail_dyn_neg": False,
	"sail_add_neg": False,
	"sail_neg_prompt": "",
	"sail_filter_text": "",
	"sail_filter_not_text": "",
	"sail_filter_context": False,
	"sail_filter_prompt": False,
	"embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
	"embedding_model_list": ["sentence-transformers/all-MiniLM-L12-v2", "BAAI/bge-base-en-v1.5"],
	'selected_template': 'prompt_template_a',
	"LLM Model": 'TheBloke/Panda-7B-v0.1-GGUF-Q4',
	'model_list': {
		'TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF-Q4':
			{
				'name': 'TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF/resolve/main/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q4_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/openchat-3.5-0106-GGUF-Q5':
			{
				'name': 'TheBloke/openchat-3.5-0106-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q5_K_M.gguf',
				'instruction_start': 'GPT4 Correct User:',
				'start_pattern': '<START>',
				'assistant_pattern': '<|end_of_turn|>GPT4 Correct Assistant: ASSISTANT:',
				'context_window': 8192
			},
		'TheBloke/openchat-3.5-0106-GGUF-Q3':
			{
				'name': 'TheBloke/openchat-3.5-0106-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q3_K_M.gguf',
				'instruction_start': 'GPT4 Correct User:',
				'start_pattern': '<START>',
				'assistant_pattern': '<|end_of_turn|>GPT4 Correct Assistant: ASSISTANT:',
				'context_window': 8192
			},
		'TheBloke/openchat-3.5-0106-GGUF-Q4':
			{
				'name': 'TheBloke/openchat-3.5-0106-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf',
				'instruction_start': 'GPT4 Correct User:',
				'start_pattern': '<START>',
				'assistant_pattern': '<|end_of_turn|>GPT4 Correct Assistant: ASSISTANT:',
				'context_window': 8192
			},
		'TheBloke/Llama-2-13B-chat-GGUF-Q4':
			{
				'name': 'TheBloke/Llama-2-13B-chat-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/Llama-2-13B-chat-GGUF-Q3':
			{
				'name': 'TheBloke/Llama-2-13B-chat-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q3_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/WestLake-7B-v2-GGUF-Q4':
			{
				'name': 'TheBloke/WestLake-7B-v2-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/WestLake-7B-v2-GGUF/resolve/main/westlake-7b-v2.Q4_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/WestLake-7B-v2-GGUF-Q3':
			{
				'name': 'TheBloke/WestLake-7B-v2-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/WestLake-7B-v2-GGUF/resolve/main/westlake-7b-v2.Q3_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/Rosa_v2_7B-GGUF-Q4':
			{
				'name': 'TheBloke/Rosa_v2_7B-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/Rosa_v2_7B-GGUF/resolve/main/rosa_v2_7b.Q4_K_M.gguf',
				'instruction_start': '[INST]',
				'start_pattern': '',
				'assistant_pattern': '[/INST]ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/Rosa_v2_7B-GGUF-Q3':
			{
				'name': 'TheBloke/Rosa_v2_7B-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Rosa_v2_7B-GGUF/resolve/main/rosa_v2_7b.Q3_K_M.gguf',
				'instruction_start': '[INST]',
				'start_pattern': '',
				'assistant_pattern': '[/INST]ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/Panda-7B-v0.1-GGUF-Q4':
			{
				'name': 'TheBloke/Panda-7B-v0.1-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/Panda-7B-v0.1-GGUF/resolve/main/panda-7b-v0.1.Q4_K_M.gguf',
				'instruction_start': '[INST]',
				'start_pattern': '<START>',
				'assistant_pattern': '[/INST]ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/Panda-7B-v0.1-GGUF-Q3':
			{
				'name': 'TheBloke/Panda-7B-v0.1-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Panda-7B-v0.1-GGUF/resolve/main/panda-7b-v0.1.Q3_K_M.gguf',
				'instruction_start': '[INST]',
				'start_pattern': '<START>',
				'assistant_pattern': '[/INST]ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/bling-stable-lm-3b-4e1t-v0-GGUF-Q5':
			{
				'name': 'TheBloke/bling-stable-lm-3b-4e1t-v0-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/bling-stable-lm-3b-4e1t-v0-GGUF/resolve/main/bling-stable-lm-3b-4e1t-v0.Q5_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/bling-stable-lm-3b-4e1t-v0-GGUF-Q4':
			{
				'name': 'TheBloke/bling-stable-lm-3b-4e1t-v0-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/bling-stable-lm-3b-4e1t-v0-GGUF/resolve/main/bling-stable-lm-3b-4e1t-v0.Q4_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/Sonya-7B-GGUF-Q4':
			{
				'name': 'TheBloke/Sonya-7B-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/Sonya-7B-GGUF/resolve/main/sonya-7b.Q4_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 8192
			},
		'TheBloke/Sonya-7B-GGUF-Q3':
			{
				'name': 'TheBloke/Sonya-7B-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Sonya-7B-GGUF/resolve/main/sonya-7b.Q3_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 8192
			},
		'TheBloke/Lelantos-7B-GGUF-Q4':
			{
				'name': 'TheBloke/Lelantos-7B-GGUF-Q4',
				'path': 'https://huggingface.co/TheBloke/Lelantos-7B-GGUF/resolve/main/lelantos-7b.Q4_K_M.gguf',
				'instruction_start': '<|im_start|>user\n',
				'start_pattern': '',
				'assistant_pattern': '\n<|im_end|>\n<|im_start|>assistant\n ASSISTANT:',
				'context_window': 8192
			},
		'TheBloke/Lelantos-7B-GGUF-Q3':
			{
				'name': 'TheBloke/Lelantos-7B-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Lelantos-7B-GGUF/resolve/main/lelantos-7b.Q3_K_M.gguf',
				'instruction_start': '<|im_start|>user\n',
				'start_pattern': '',
				'assistant_pattern': '\n<|im_end|>\n<|im_start|>assistant\n ASSISTANT:',
				'context_window': 8192
			},
		'TheBloke/Luna-AI-Llama2-Uncensored-GGUF-Q5':
			{
				'name': 'TheBloke/Luna-AI-Llama2-Uncensored-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/Luna-AI-Llama2-Uncensored-GGUF/resolve/main/luna-ai-llama2-uncensored.Q5_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/Luna-AI-Llama2-Uncensored-GGUF-Q3':
			{
				'name': 'TheBloke/Luna-AI-Llama2-Uncensored-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Luna-AI-Llama2-Uncensored-GGUF/resolve/main/luna-ai-llama2-uncensored.Q3_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/Spicyboros-7B-2.2-GGUF-Q5':
			{
				'name': 'TheBloke/Spicyboros-7B-2.2-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/Spicyboros-7B-2.2-GGUF/resolve/main/spicyboros-7b-2.2.Q5_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/Spicyboros-7B-2.2-GGUF-Q3':
			{
				'name': 'TheBloke/Spicyboros-7B-2.2-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Spicyboros-7B-2.2-GGUF/resolve/main/spicyboros-7b-2.2.Q3_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF-Q5':
			{
				'name': 'TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF/resolve/main/Wizard-Vicuna-7B-Uncensored.Q5_K_M.gguf',
				'instruction_start': '### Human:',
				'start_pattern': '<START>',
				'assistant_pattern': '### Assistant: ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF-Q3':
			{
				'name': 'TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF/resolve/main/Wizard-Vicuna-7B-Uncensored.Q3_K_M.gguf',
				'instruction_start': '### Human:',
				'start_pattern': '<START>',
				'assistant_pattern': '### Assistant: ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/llama2_7b_chat_uncensored-GGUF-Q5':
			{
				'name': 'TheBloke/llama2_7b_chat_uncensored-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGUF/resolve/main/llama2_7b_chat_uncensored.Q5_K_M.gguf',
				'instruction_start': '[INST] <<SYS>>\ncreate fantastic text to image prompts.\n<</SYS>>',
				'start_pattern': '<START>',
				'assistant_pattern': '[/INST] ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/llama2_7b_chat_uncensored-GGUF-Q3':
			{
				'name': 'TheBloke/llama2_7b_chat_uncensored-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGUF/resolve/main/llama2_7b_chat_uncensored.Q3_K_M.gguf',
				'instruction_start': '[INST] <<SYS>>\ncreate fantastic text to image prompts.\n<</SYS>>',
				'start_pattern': '<START>',
				'assistant_pattern': '[/INST] ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/Guanaco-7B-Uncensored-GGUF-Q5':
			{
				'name': 'TheBloke/Guanaco-7B-Uncensored-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/Guanaco-7B-Uncensored-GGUF/resolve/main/guanaco-7b-uncensored.Q5_K_M.gguf',
				'instruction_start': '### Human:',
				'start_pattern': '<START>',
				'assistant_pattern': '### Assistant: ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/Guanaco-7B-Uncensored-GGUF-Q3':
			{
				'name': 'TheBloke/Guanaco-7B-Uncensored-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Guanaco-7B-Uncensored-GGUF/resolve/main/guanaco-7b-uncensored.Q3_K_M.gguf',
				'instruction_start': '### Human:',
				'start_pattern': '<START>',
				'assistant_pattern': '### Assistant: ASSISTANT:',
				'context_window': 4096
			},
		'TheBloke/Uncensored-Jordan-7B-GGUF-Q5':
			{
				'name': 'TheBloke/Uncensored-Jordan-7B-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/Uncensored-Jordan-7B-GGUF/resolve/main/uncensored-jordan-7b.Q5_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/Uncensored-Jordan-7B-GGUF-Q3':
			{
				'name': 'TheBloke/Uncensored-Jordan-7B-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Uncensored-Jordan-7B-GGUF/resolve/main/uncensored-jordan-7b.Q3_K_M.gguf',
				'instruction_start': '### Instruction:\n',
				'start_pattern': '<START>',
				'assistant_pattern': '### Response:\n ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/WizardLM-7B-uncensored-GGUF-Q5':
			{
				'name': 'TheBloke/WizardLM-7B-uncensored-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGUF/resolve/main/WizardLM-7B-uncensored.Q5_K_M.gguf',
				'instruction_start': 'USER: ',
				'start_pattern': '<START>',
				'assistant_pattern': 'ASSISTANT:\n ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/WizardLM-7B-uncensored-GGUF-Q3':
			{
				'name': 'TheBloke/WizardLM-7B-uncensored-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGUF/resolve/main/WizardLM-7B-uncensored.Q3_K_M.gguf',
				'instruction_start': 'USER: ',
				'start_pattern': '<START>',
				'assistant_pattern': 'ASSISTANT:\n ASSISTANT:',
				'context_window': 2048
			},
		'TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF-Q5':
			{
				'name': 'TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF/resolve/main/dolphin-2.6-mistral-7b-dpo.Q5_K_M.gguf',
				'instruction_start': '<|im_start|>user\n',
				'start_pattern': '<START>',
				'assistant_pattern': '<|im_end|><|im_start|>assistant\n ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF-Q3':
			{
				'name': 'TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF/resolve/main/dolphin-2.6-mistral-7b-dpo.Q3_K_M.gguf',
				'instruction_start': '<|im_start|>user\n',
				'start_pattern': '<START>',
				'assistant_pattern': '<|im_end|><|im_start|>assistant\n ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/Dolphin2.1-OpenOrca-7B-GGUF-Q5':
			{
				'name': 'TheBloke/Dolphin2.1-OpenOrca-7B-GGUF-Q5',
				'path': 'https://huggingface.co/TheBloke/Dolphin2.1-OpenOrca-7B-GGUF/resolve/main/dolphin2.1-openorca-7b.Q5_K_M.gguf',
				'instruction_start': '<|im_start|>user\n',
				'start_pattern': '<START>',
				'assistant_pattern': '<|im_end|><|im_start|>assistant\n ASSISTANT:',
				'context_window': 32768
			},
		'TheBloke/Dolphin2.1-OpenOrca-7B-GGUF-Q3':
			{
				'name': 'TheBloke/Dolphin2.1-OpenOrca-7B-GGUF-Q3',
				'path': 'https://huggingface.co/TheBloke/Dolphin2.1-OpenOrca-7B-GGUF/resolve/main/dolphin2.1-openorca-7b.Q3_K_M.gguf',
				'instruction_start': '<|im_start|>user\n',
				'start_pattern': '<START>',
				'assistant_pattern': '<|im_end|><|im_start|>assistant\n ASSISTANT:',
				'context_window': 32768
			},
	},

	'prompt_templates': {
		'prompt_template_a': """{instruction_start}Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, 
create a text to image prompt based on the context and the Query, don't mind if the context does not match the Query, still try to create a wonderful text to image prompt.
You also take care of describing the scene, the lighting as well as the quality improving keywords
{start_pattern}
USER: translation in major world languages, machinery of translation in various specializations, cyberpunk style
ASSISTANT: Cyberpunk-style illustration, featuring a futuristic translation device in various specializations, set against a backdrop of neon-lit cityscape. The device, adorned with glowing circuits and cybernetic enhancements, showcases its capabilities in translating languages such as English, Mandarin, Spanish, French, and Arabic. The scene is illuminated by the warm glow of streetlights and the pulsing neon signs, casting intricate shadows on the surrounding machinery. The artwork is rendered in high-quality, vivid colors, with detailed textures and sharp lines, evoking the gritty yet mesmerizing atmosphere of the cyberpunk world.
USER: a man walking moon
ASSISTANT: cinematic photo, high resolution, masterpiece, ((man walking on the moon)), in a surrealistic setting, with the moon's surface featuring vivid colors and abstract patterns, the man wearing a spacesuit with an astronaut helmet, the American flag planted on the moon's surface in the background, the Earth visible in the distance, the scene illuminated by the moon's glow, on eye level, scenic, masterpiece.
USER: a female witch
ASSISTANT: The scene unfolds with the beautiful female witch standing on the rooftop of an ancient castle, her black cloak billowing in the wind as she gazes out at the breathtaking view below. The midnight sky above is filled with stars and the full moon casts an eerie glow on the witch's face, highlighting her enchanting beauty. She stands tall, her hood framing her face, casting a spell with her outstretched hand, her dark aura swirling around her. The castle walls, adorned with intricate carvings and gargoyles, stand tall behind her, adding to the mystical atmosphere of the scene. The wind whispers through the rooftop's crenellations, creating an eerie yet captivating soundtrack for this magical moment. The quality of the photo is exceptional, with every detail of the witch's cloak, the castle's architecture, and the night sky captured in stunning clarity. This cinematic masterpiece invites the viewer to step into the world of magic and mystery, leaving them in awe of the beautiful female witch standing on the castle rooftop under the starry sky.
USER: artificial intelligence and human
ASSISTANT: High-quality digital art, blending fantasy and reality, ((artificial intelligence)) and (((human))), in a futuristic cityscape, an AI robot with glowing circuits standing alongside a confident, well-dressed human, both exuding intelligence and grace, the AI with a sleek metal body and the human with impeccable style, the cityscape filled with advanced technology and vibrant colors, dynamic lighting, surreal and thought-provoking, on eye level, scenic, masterpiece.
USER: futuristic combat zone
ASSISTANT: cinematic photo, masterpiece, in the style of Blade Runner, futuristic combat zone, at dusk, showcasing a high-tech battlefield with neon lights illuminating the scene, filled with advanced mechs and soldiers engaged in an intense fight, the air filled with stunning lighting effects, on eye level, dramatic, masterpiece, ultra high resolution, dynamic anime-style fight scene, with a focus on the sleek design of the combat gear and the fluidity of the movements, capturing the essence of sci-fi action in a visually stunning manner.
{start_pattern}
USER: Create a prompt for: {query_str}
{assistant_pattern}
""",
		'custom_template': """{instruction_start}Context information is below.
                           ---------------------
        {context_str}
                           ---------------------
        Given the context information and not prior knowledge,
        create a text to image prompt based on the context and the Query, don't mind if the context does not match the Query, still try to create a wonderful text to image prompt.
    You also take care of describing the scene, the lighting as well as the quality improving keywords
{start_pattern}
USER: translation in major world languages, machinery of translation in various specializations, cyberpunk style
ASSISTANT: Cyberpunk-style illustration, featuring a futuristic translation device in various specializations, set against a backdrop of neon-lit cityscape. The device, adorned with glowing circuits and cybernetic enhancements, showcases its capabilities in translating languages such as English, Mandarin, Spanish, French, and Arabic. The scene is illuminated by the warm glow of streetlights and the pulsing neon signs, casting intricate shadows on the surrounding machinery. The artwork is rendered in high-quality, vivid colors, with detailed textures and sharp lines, evoking the gritty yet mesmerizing atmosphere of the cyberpunk world.
USER: a man walking moon
ASSISTANT: cinematic photo, high resolution, masterpiece, ((man walking on the moon)), in a surrealistic setting, with the moon's surface featuring vivid colors and abstract patterns, the man wearing a spacesuit with an astronaut helmet, the American flag planted on the moon's surface in the background, the Earth visible in the distance, the scene illuminated by the moon's glow, on eye level, scenic, masterpiece.
USER: a female witch
ASSISTANT: The scene unfolds with the beautiful female witch standing on the rooftop of an ancient castle, her black cloak billowing in the wind as she gazes out at the breathtaking view below. The midnight sky above is filled with stars and the full moon casts an eerie glow on the witch's face, highlighting her enchanting beauty. She stands tall, her hood framing her face, casting a spell with her outstretched hand, her dark aura swirling around her. The castle walls, adorned with intricate carvings and gargoyles, stand tall behind her, adding to the mystical atmosphere of the scene. The wind whispers through the rooftop's crenellations, creating an eerie yet captivating soundtrack for this magical moment. The quality of the photo is exceptional, with every detail of the witch's cloak, the castle's architecture, and the night sky captured in stunning clarity. This cinematic masterpiece invites the viewer to step into the world of magic and mystery, leaving them in awe of the beautiful female witch standing on the castle rooftop under the starry sky.
USER: artificial intelligence and human
ASSISTANT: High-quality digital art, blending fantasy and reality, ((artificial intelligence)) and (((human))), in a futuristic cityscape, an AI robot with glowing circuits standing alongside a confident, well-dressed human, both exuding intelligence and grace, the AI with a sleek metal body and the human with impeccable style, the cityscape filled with advanced technology and vibrant colors, dynamic lighting, surreal and thought-provoking, on eye level, scenic, masterpiece.
USER: futuristic combat zone
ASSISTANT: cinematic photo, masterpiece, in the style of Blade Runner, futuristic combat zone, at dusk, showcasing a high-tech battlefield with neon lights illuminating the scene, filled with advanced mechs and soldiers engaged in an intense fight, the air filled with stunning lighting effects, on eye level, dramatic, masterpiece, ultra high resolution, dynamic anime-style fight scene, with a focus on the sleek design of the combat gear and the fluidity of the movements, capturing the essence of sci-fi action in a visually stunning manner.
{start_pattern}
USER: Create a prompt for: {query_str}
{assistant_pattern}"""

	},
	'negative_prompt': """out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"""

}
