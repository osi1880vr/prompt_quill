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

import globals
from llama_index_interface import adapter

from settings import io
from deep_translator import GoogleTranslator
from generators.automatics import client as auto_client

import gc
import os
import time
import math


settings_io = io.settings_io()
out_dir = 'api_out'
out_dir_t2t = os.path.join(out_dir, 'txt2txt')
automa_client = auto_client.automa_client()





class LLM_INTERFACE:

    def __init__(self):
        self.g = globals.get_globals()
        self.g.settings_data = settings_io.load_settings()
        self.adapter = adapter()
        self.document_store = self.adapter.get_document_store()



    async def log(self,logfile, text):
        f = open(logfile, 'a')
        try:
            f.write(f"QUERY: {text} \n")
        except:
            pass
        f.close()


    async def log_raw(self,logfile, text):
        f = open(logfile, 'a')
        try:
            f.write(f"{text}\n")
        except:
            pass
        f.close()

    async def retrieve_context(self, prompt):
        nodes = await self.adapter.retrieve_context(prompt)
        self.g.last_context = [s.node.get_text() for s in nodes]
        return self.g.last_context


    async def set_top_k(self, top_k):
        self.g.settings_data['top_k'] = top_k
        await self.adapter.set_pipeline()

    async def get_context_details(self):
        return self.g.last_context

    async def reload_settings(self):
        self.g.settings_data = await settings_io.load_settings()

    async def change_model(self,model,temperature,n_ctx,n_gpu_layers,max_tokens,top_k, instruct):
        return await self.adapter.change_model(model,temperature,n_ctx,n_gpu_layers,max_tokens,top_k, instruct)

    async def set_prompt(self,prompt_text):
        return await self.adapter.set_prompt(prompt_text)

    async def translate(self, query):
        tanslated = GoogleTranslator(source='auto', target='en').translate(query)
        return tanslated


    async def run_batch_response(self,context):
        output = ''
        n = 1
        for query in context:
            if query != '':
                response = await self.adapter.retrieve_query(query)
                output = f'{output}\n\n\nPrompt {str(n)}:\n{response.response.lstrip(" ")}'
                self.log(os.path.join(out_dir_t2t,'WildcardReady.txt'),f'{response.response.lstrip(" ")}\n')
                n += 1

        return output



    async def run_llm_response_batch(self, query):

        if self.g.settings_data['translate']:
            query = self.translate(query)

        if self.g.settings_data['Instruct Model'] is True:
            query = f'[INST]{query}[/INST]'

        response = await self.adapter.retrieve_query(query)

        output = response.response.lstrip(' ')
        output = output.replace('\n','')

        return output

    async def get_next_target(self, nodes, sail_target,sail_sinus,sail_sinus_range,sail_sinus_freq):
        target_dict = {}

        for node in nodes:
            if node.text not in self.sail_history:
                target_dict[node.score] = node.text

        if len(target_dict.keys()) < self.sail_depth:
            self.sail_depth = self.sail_depth_start + len(self.sail_history)

        if sail_sinus:
            sinus = int(math.sin(self.sail_sinus_count/10.0)*sail_sinus_range)
            self.sail_sinus_count += sail_sinus_freq
            self.sail_depth += sinus
            if self.sail_depth < 0:
                self.sail_depth = 1

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


    async def sail_automa_gen(self, query):
        return await automa_client.request_generation(query,
                                         self.g.settings_data['negative_prompt'],
                                         self.g.settings_data['automa_Sampler'],
                                         self.g.settings_data['automa_Steps'],
                                         self.g.settings_data['automa_CFG Scale'],
                                         self.g.settings_data['automa_Width'],
                                         self.g.settings_data['automa_Height'],
                                         self.g.settings_data['automa_url'], True)

    async def run_t2t_sail(self,query,sail_width,sail_depth,sail_target,sail_generate,sail_sinus,sail_sinus_range,sail_sinus_freq,sail_add_style,sail_style,sail_add_search,sail_search):
        self.sail_history = []
        self.sail_depth = sail_depth
        self.sail_depth_start = sail_depth
        self.sail_sinus_count = 1.0
        filename = os.path.join(out_dir_t2t, f'Journey_log_{time.strftime("%Y%m%d-%H%M%S")}.txt')
        sail_log = ''

        if self.g.settings_data['translate']:
            query = self.translate(query)

        images = []

        for n in range(sail_width):
            if self.g.sailing_run == False:
                break
            sail_retriever = await self.adapter.get_retriever(similarity_top_k=self.sail_depth)
            if sail_add_search:
                query = f'{sail_search}, {query}'
            response = await self.adapter.retrieve_query(query)
            prompt = response.response.lstrip(" ")
            if sail_add_style:
                prompt = f'{sail_style}, {prompt}'

            self.log_raw(filename,f'{prompt}')
            self.log_raw(filename,f'{n} ----------')
            sail_log = sail_log + f'{prompt}\n'
            sail_log = sail_log + f'{n} ----------\n'
            nodes = sail_retriever.retrieve(query)
            if sail_generate:
                img = await self.sail_automa_gen(prompt)
                images.append(img)
            query = await self.get_next_target(nodes,sail_target,sail_sinus,sail_sinus_range,sail_sinus_freq)
            if query == -1:
                self.log_raw(filename,f'{n} sail is finished early due to rotating context')
                break

        return sail_log,images


    async def run_llm_response(self, query, history):

        if self.g.settings_data['translate']:
            query = await self.translate(query)

        self.log('logfile.txt',f"QUERY: {query} \n-------------\n")

        if self.g.settings_data['Instruct Model'] is True:
            query = f'[INST]{query}[/INST]'

        response = await self.adapter.retrieve_query(query)

        self.log('logfile.txt',f"RESPONSE: {response.response} \n-------------\n")
        self.log(os.path.join(out_dir_t2t,'WildcardReady.txt'),f'{response.response.lstrip(" ")}\n')

        self.g.last_context = [s.node.get_text() for s in response.source_nodes]

        output = response.response.lstrip(' ')
        self.g.last_prompt = output

        if self.g.settings_data['translate']:
            output = f'Your prompt was translated to: {query}\n\n\n{output}'

        if self.g.settings_data['batch']:
            batch_result = await self.run_batch_response(self.g.last_context)
            output = f'Prompt 0:\n{output}\n\n\n{batch_result}'



        negative_prompts = []
        models = []

        for key in response.metadata.keys():
            if 'negative_prompt' in response.metadata[key]:
                negative_prompts = negative_prompts + response.metadata[key]['negative_prompt'].split(',')
                models.append(f'{response.metadata[key]["model_name"]}')

        if len(negative_prompts) > 0:
            negative_prompts = set(negative_prompts)
            self.g.last_negative_prompt = ",".join(negative_prompts).lstrip(' ')
            if len(self.g.last_negative_prompt) < 30:
                self.g.last_negative_prompt = self.g.settings_data['negative_prompt']
            if self.g.last_negative_prompt != '':
                output = f'{output} \n\nMaybe helpful negative prompt:\n\n{self.g.last_negative_prompt}'
        else:
            self.g.last_negative_prompt = self.g.settings_data['negative_prompt']
            output = f'{output} \n\nMaybe helpful negative prompt:\n\n{self.g.last_negative_prompt}'

        if len(models) > 0:
            models_out = "\n".join(models)
            if models_out != '':
                output = f'{output} \n\nMaybe helpful models:\n\n{models_out}'


        return output



