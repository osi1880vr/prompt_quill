import gradio as gr

import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate

from llama_index.core import ServiceContext
import os, re


import qdrant_client
from llama_index.vector_stores.milvus import MilvusVectorStore
vector_store = MilvusVectorStore(
	uri = "http://192.168.0.127:19530",
	port = 19530   ,
	collection_name = 'llama_index_prompts_all',
	dim = 384,
	similarity_metric = "L2",
	#   text_key="paragraph"
)


from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

llm = LlamaCPP(
	#model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf",
	#model_url="https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF/resolve/main/solar-10.7b-instruct-v1.0-uncensored.Q5_K_M.gguf",
	#model_url="https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-GGUF/resolve/main/yarn-mistral-7b-128k.Q5_K_M.gguf",   # strange answers
	#model_url="https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q5_K_M.gguf",
	model_url="https://huggingface.co/TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-GGUF/resolve/main/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf",

	#model_url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
	#model_url="https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF/resolve/main/phind-codellama-34b-v2.Q4_K_M.gguf",
	# optionally, you can set the path to a pre-downloaded model instead of model_url
	model_path=None,

	temperature=0.3,
	max_new_tokens=1024,

	# llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
	context_window=3900,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.

	# kwargs to pass to __call__()
	generate_kwargs={},

	# kwargs to pass to __init__()
	# set to at least 1 to use GPU
	model_kwargs={"n_gpu_layers": 50}, # I need to play with this and see if it actually helps

	# transform inputs into Llama2 format
	messages_to_prompt=messages_to_prompt,
	completion_to_prompt=completion_to_prompt,
	verbose=True,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
#from llama_index.llms import Ollama
#service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)


from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

#query_engine = vector_index.as_query_engine(similarity_top_k=10,)

query_engine = vector_index.as_chat_engine(similarity_top_k=10,)


qa_prompt_tmpl_str = (
	"Context information is below.\n"
	"---------------------\n"
	"{context_str}\n"
	"---------------------\n"
	"Given the context information and not prior knowledge, "
	"create a text to image prompt based on the context and the Query, don't mind if the context does not match the Query, still try to create a wonderfull text to image prompt.\n"
	"You also take care of describing the scene, the lighting as well as the quality improving keywords\n"
	"Query: translation in major world languages, machinery of translation in various specializations, cyberpunk style"
	"Answer: Cyberpunk-style illustration, featuring a futuristic translation device in various specializations, set against a backdrop of neon-lit cityscape. The device, adorned with glowing circuits and cybernetic enhancements, showcases its capabilities in translating languages such as English, Mandarin, Spanish, French, and Arabic. The scene is illuminated by the warm glow of streetlights and the pulsing neon signs, casting intricate shadows on the surrounding machinery. The artwork is rendered in high-quality, vivid colors, with detailed textures and sharp lines, evoking the gritty yet mesmerizing atmosphere of the cyberpunk world."
	"Query: a man walking moon"
	"Answer: cinematic photo, high resolution, masterpiece, ((man walking on the moon)), in a surrealistic setting, with the moon's surface featuring vivid colors and abstract patterns, the man wearing a spacesuit with an astronaut helmet, the American flag planted on the moon's surface in the background, the Earth visible in the distance, the scene illuminated by the moon's glow, on eye level, scenic, masterpiece."
	"Query: a female witch"
	"Answer: The scene unfolds with the beautiful female witch standing on the rooftop of an ancient castle, her black cloak billowing in the wind as she gazes out at the breathtaking view below. The midnight sky above is filled with stars and the full moon casts an eerie glow on the witch's face, highlighting her enchanting beauty. She stands tall, her hood framing her face, casting a spell with her outstretched hand, her dark aura swirling around her. The castle walls, adorned with intricate carvings and gargoyles, stand tall behind her, adding to the mystical atmosphere of the scene. The wind whispers through the rooftop's crenellations, creating an eerie yet captivating soundtrack for this magical moment. The quality of the photo is exceptional, with every detail of the witch's cloak, the castle's architecture, and the night sky captured in stunning clarity. This cinematic masterpiece invites the viewer to step into the world of magic and mystery, leaving them in awe of the beautiful female witch standing on the castle rooftop under the starry sky."
	"Query: artifical intelligence and human"
	"Answer: High-quality digital art, blending fantasy and reality, ((artificial intelligence)) and (((human))), in a futuristic cityscape, an AI robot with glowing circuits standing alongside a confident, well-dressed human, both exuding intelligence and grace, the AI with a sleek metal body and the human with impeccable style, the cityscape filled with advanced technology and vibrant colors, dynamic lighting, surreal and thought-provoking, on eye level, scenic, masterpiece."
	"Query: futuristic combat zone"
	"Answer: cinematic photo, masterpiece, in the style of Blade Runner, futuristic combat zone, at dusk, showcasing a high-tech battlefield with neon lights illuminating the scene, filled with advanced mechs and soldiers engaged in an intense fight, the air filled with stunning lighting effects, on eye level, dramatic, masterpiece, ultra high resolution, dynamic anime-style fight scene, with a focus on the sleek design of the combat gear and the fluidity of the movements, capturing the essence of sci-fi action in a visually stunning manner."
	"Query: {query_str}\n"
	"Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

query_engine.update_prompts(
	{"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)



def yes_man(query, history):

	f = open('logfile.txt', 'a')
	f.write(f"QUERY: {query} \n")
	f.close()


	# if len(history) > 0:
	# 	hist_query=''
	# 	round = 0
	# 	for hist in history:
	# 		if round == 0:
	# 			round = 1
	# 			hist_query = hist_query + f'{hist[0]}/nAnswer: {hist[1]}/n'
	# 		else:
	# 			hist_query = hist_query + f'Query: {hist[0]}/nAnswer: {hist[1]}/n'
	#
	# 	hist_query = hist_query + f'Query: {query}'
	# 	response = query_engine.query(hist_query)
	# else:
	# 	response = query_engine.query(query)





	f = open('logfile.txt', 'a')
	f.write(f'RESPONSE: {response.response} \n')
	f.close()
	return response.response

gr.ChatInterface(
	yes_man,
	chatbot=gr.Chatbot(height=300,label='magic prompts'),
	textbox=gr.Textbox(placeholder="Make your prompts more magical", container=False, scale=7),
	title="Prompt Magic v0.0.1 build with llmware",
	description="Enter your prompt to work with",
	theme="soft",
	examples=['A fishermans lake','night at cyberpunk city','living in a steampunk world'],
	cache_examples=True,
	retry_btn=None,
	undo_btn="Delete Previous",
	clear_btn="Clear",
).launch(server_name='0.0.0.0')