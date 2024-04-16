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

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
from settings import io
from deep_translator import GoogleTranslator
from generators.automatics import client as auto_client

import gc
import os
import time

settings_io = io.settings_io()
out_dir = 'api_out'
out_dir_t2t = os.path.join(out_dir, 'txt2txt')
automa_client = auto_client.automa_client()



url = "http://localhost:6333"

if os.getenv("QDRANT_URL") is not None:
    url = os.environ["QDRANT_URL"]

class LLM_INTERFACE:


    def __init__(self):

        self.index='prompts_large_meta'
        self.last_prompt = ''
        self.last_negative_prompt = ''
        self.last_context = []
        self.settings_data = settings_io.load_settings()
        self.model_path = self.settings_data['model_list'][self.settings_data['LLM Model']]['path']

        self.document_store = qdrant_client.QdrantClient(
            # you can use :memory: mode for fast and light-weight experiments,
            # it does not require to have Qdrant deployed anywhere
            # but requires qdrant-client >= 1.1.1
            #location=":memory:"
            # otherwise set Qdrant instance address with:
            url=url
            # set API KEY for Qdrant Cloud
            # api_key="<qdrant-api-key>",
        )

        self.instruct = False
        self.n_ctx=3900
        self.n_batch=128
        self.n_gpu_layers=50
        self.max_tokens=200
        self.temperature=0.0
        self.top_k=10

        self.set_llm()

        self.prompt_template = self.settings_data['prompt_templates']['prompt_template_b']

        self.set_pipeline()



    def set_llm(self):

        self.llm = LlamaCPP(

            model_url=self.model_path,

            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=None,

            temperature=self.temperature,
            max_new_tokens=self.max_tokens,

            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=self.n_ctx,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.

            # kwargs to pass to __call__()
            generate_kwargs={},

            # kwargs to pass to __init__()
            # set to at least 1 to use GPU, check with your model the number need to fully run on GPU might be way higher than 1
            model_kwargs={"n_gpu_layers": self.n_gpu_layers}, # I need to play with this and see if it actually helps

            # transform inputs into Llama2 format
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )

    def set_pipeline(self):

        if hasattr(self,'query_engine'):
            del self.vector_store
            del self.vector_index
            del self.query_engine


        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
        self.vector_store = QdrantVectorStore(client=self.document_store, collection_name=self.index)
        self.vector_index = VectorStoreIndex.from_vector_store( vector_store=self.vector_store, embed_model=self.embed_model)

        self.retriever = self.vector_index.as_retriever(similarity_top_k=self.settings_data['top_k'])

        self.query_engine = self.vector_index.as_query_engine(similarity_top_k=self.settings_data['top_k'],llm=self.llm)

        self.qa_prompt_tmpl = PromptTemplate(self.prompt_template)

        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": self.qa_prompt_tmpl}
        )


    def log(self,logfile, text):
        f = open(logfile, 'a')
        try:
            f.write(f"QUERY: {text} \n")
        except:
            pass
        f.close()
    def log_raw(self,logfile, text):
        f = open(logfile, 'a')
        try:
            f.write(f"{text}\n")
        except:
            pass
        f.close()

    def retrieve_context(self, prompt):
        nodes = self.retriever.retrieve(prompt)
        self.last_context = [s.node.get_text() for s in nodes]
        return self.last_context


    def set_top_k(self, top_k):
        self.settings_data['top_k'] = top_k
        self.set_pipeline()

    def get_context_details(self):
        return self.last_context

    def reload_settings(self):
        self.settings_data = settings_io.load_settings()


    def translate(self, query):
        tanslated = GoogleTranslator(source='auto', target='en').translate(query)
        return tanslated


    def run_batch_response(self,context):
        output = ''
        n = 1
        for query in context:
            if query != '':
                response = self.query_engine.query(query)
                output = f'{output}\n\n\nPrompt {str(n)}:\n{response.response.lstrip(" ")}'
                self.log(os.path.join(out_dir_t2t,'WildcardReady.txt'),f'{response.response.lstrip(" ")}\n')
                n += 1

        return output



    def run_llm_response_batch(self, query):

        if self.settings_data['translate']:
            query = self.translate(query)

        if self.instruct is True:
            query = f'[INST]{query}[/INST]'

        response = self.query_engine.query(query)

        output = response.response.lstrip(' ')
        output = output.replace('\n','')

        return output

    def get_next_target(self, nodes, sail_target):
        target_dict = {}

        for node in nodes:
            if node.text not in self.sail_history:
                target_dict[node.score] = node.text

        if len(target_dict.keys()) > 0:

            if sail_target:
                out =  target_dict[min(target_dict.keys())]
                self.sail_history.append(out)
                return out
            else:
                out =  target_dict[max(target_dict.keys())]
                self.sail_history.append(out)
                return out
        else:
            return -1


    def sail_automa_gen(self, query):
        return automa_client.request_generation(query,
                                         self.settings_data['negative_prompt'],
                                         self.settings_data['automa_Sampler'],
                                         self.settings_data['automa_Steps'],
                                         self.settings_data['automa_CFG Scale'],
                                         self.settings_data['automa_Width'],
                                         self.settings_data['automa_Height'],
                                         self.settings_data['automa_url'], True)

    def run_t2t_sail(self,query,sail_width,sail_depth,sail_target,sail_generate):
        self.sail_history = []
        filename = os.path.join(out_dir_t2t, f'Journey_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')
        sail_log = ''
        sail_retriever = self.vector_index.as_retriever(similarity_top_k=sail_depth)
        if self.settings_data['translate']:
            query = self.translate(query)

        images = []

        for n in range(sail_width):
            response = self.query_engine.query(query)
            self.log_raw(filename,f'{response.response.lstrip(" ")}')
            self.log_raw(filename,f'{n} ----------')
            sail_log = sail_log + f'{response.response.lstrip(" ")}\n'
            sail_log = sail_log + f'{n} ----------\n'
            nodes = sail_retriever.retrieve(query)
            if sail_generate:
                img = self.sail_automa_gen(response.response.lstrip(" "))
                images.append(img)
            query = self.get_next_target(nodes,sail_target)
            if query == -1:
                self.log_raw(filename,f'{n} sail is finished early due to rotating context')
                break

        return sail_log,images


    def run_llm_response(self, query, history):

        if self.settings_data['translate']:
            query = self.translate(query)

        self.log('logfile.txt',f"QUERY: {query} \n-------------\n")

        if self.instruct is True:
            query = f'[INST]{query}[/INST]'

        response = self.query_engine.query(query)

        self.log('logfile.txt',f"RESPONSE: {response.response} \n-------------\n")
        self.log(os.path.join(out_dir_t2t,'WildcardReady.txt'),f'{response.response.lstrip(" ")}\n')

        self.last_context = [s.node.get_text() for s in response.source_nodes]

        output = response.response.lstrip(' ')
        self.last_prompt = output

        if self.settings_data['translate']:
            output = f'Your prompt was translated to: {query}\n\n\n{output}'

        if self.settings_data['batch']:
            batch_result = self.run_batch_response(self.last_context)
            output = f'Prompt 0:\n{output}\n\n\n{batch_result}'



        negative_prompts = []
        models = []

        for key in response.metadata.keys():
            if 'negative_prompt' in response.metadata[key]:
                negative_prompts = negative_prompts + response.metadata[key]['negative_prompt'].split(',')
                models.append(f'{response.metadata[key]["model_name"]}')

        if len(negative_prompts) > 0:
            negative_prompts = set(negative_prompts)
            self.last_negative_prompt = ",".join(negative_prompts).lstrip(' ')
            if len(self.last_negative_prompt) < 30:
                self.last_negative_prompt = self.settings_data['negative_prompt']
            if self.last_negative_prompt != '':
                output = f'{output} \n\nMaybe helpful negative prompt:\n\n{self.last_negative_prompt}'
        else:
            self.last_negative_prompt = self.settings_data['negative_prompt']
            output = f'{output} \n\nMaybe helpful negative prompt:\n\n{self.last_negative_prompt}'

        if len(models) > 0:
            models_out = "\n".join(models)
            if models_out != '':
                output = f'{output} \n\nMaybe helpful models:\n\n{models_out}'


        return output


    def change_model(self,model,temperature,n_ctx,n_gpu_layers,max_tokens,top_k, instruct):

        self.n_ctx=n_ctx
        self.n_gpu_layers=n_gpu_layers
        self.max_tokens=max_tokens
        self.temperature=float(temperature)
        self.top_k=top_k
        self.instruct = instruct

        self.model_path = model['path']

        self.llm._model = None
        del self.llm

        # delete the model from Ram
        gc.collect()

        self.set_llm()

        self.set_pipeline()
        return f'Model set to {model["name"]}'

    def set_prompt(self,prompt_text):
        self.prompt_template = prompt_text

        self.log('magic_prompt_logfile.txt',f"Magic Prompt: \n{prompt_text} \n")

        self.llm._model = None
        del self.llm

        # delete the model from Ram
        gc.collect()

        self.set_llm()

        self.set_pipeline()
        return f'Magic Prompt set to:\n {prompt_text}'