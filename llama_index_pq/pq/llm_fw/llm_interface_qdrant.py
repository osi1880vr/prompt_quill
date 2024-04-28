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

import globals
from llm_fw.llama_index_interface import adapter

from post_process.summary import extractive_summary
from deep_translator import GoogleTranslator
import os

out_dir = 'api_out'
out_dir_t2t = os.path.join(out_dir, 'txt2txt')





class LLM_INTERFACE:

    def __init__(self):

        self.g = globals.get_globals()
        self.adapter = adapter()
        self.g.negative_prompt_list = []
        self.g.models_list = []


    def change_model(self,model,temperature,n_ctx,max_tokens,n_gpu_layers, top_k, instruct):
        return self.adapter.change_model(model,temperature,n_ctx,n_gpu_layers,max_tokens,top_k, instruct)

    def set_prompt(self,prompt_text):
        return self.adapter.set_prompt(prompt_text)
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

    def retrieve_context(self, query):
        self.last_context = self.adapter.get_context_text(query)
        return self.last_context


    def set_top_k(self, top_k):
        self.g.settings_data['top_k'] = top_k
        self.adapter.set_pipeline()


    def get_context_details(self):
        return self.g.last_context

    def translate(self, query):
        tanslated = GoogleTranslator(source='auto', target='en').translate(query)
        return tanslated


    def run_batch_response(self,context):
        output = ''
        n = 1
        for query in context:
            if query != '':
                response = self.adapter.retrieve_query(query)
                output = f'{output}\n\n\nPrompt {str(n)}:\n{response}'
                self.log(os.path.join(out_dir_t2t,'WildcardReady.txt'),f'{response}\n')
                n += 1

        return output



    def run_llm_response_batch(self, query):

        if self.g.settings_data['translate']:
            query = self.translate(query)

        if self.g.settings_data['Instruct Model'] is True:
            query = f'[INST]{query}[/INST]'

        response = self.adapter.retrieve_query(query)

        output = response

        if self.g.settings_data['summary']:
            output = extractive_summary(output)

        output = output.replace('\n',' ')
        return output


    def get_retriever(self,similarity_top_k):
        return  self.adapter.get_retriever(similarity_top_k=similarity_top_k)


    def retrieve_top_k_query(self, query, similarity_top_k):
        retriever = self.get_retriever(similarity_top_k)
        return retriever.retrieve(query)


    def get_query_texts(self, nodes):
        target_dict = {}
        for node in nodes:
            if node.text not in self.g.sail_history:
                target_dict[node.score] = node.text
        return target_dict


    def retrieve_query(self, query):
        return self.adapter.retrieve_query(query)

    def rephrase(self,prompt, query):

        return self.adapter.retrieve_rephrase_query(query, prompt)

    def run_llm_response(self, query, history):

        if self.g.settings_data['translate']:
            query = self.translate(query)

        self.log('logfile.txt',f"QUERY: {query} \n-------------\n")

        if self.g.settings_data['Instruct Model'] is True:
            query = f'[INST]{query}[/INST]'

        response = self.adapter.retrieve_query(query)

        self.g.last_prompt = response

        self.log('logfile.txt',f"RESPONSE: {response} \n-------------\n")
        self.log(os.path.join(out_dir_t2t,'WildcardReady.txt'),f'{response}\n')

        output = response

        if self.g.settings_data['summary']:
            output = extractive_summary(output)

        if self.g.settings_data['translate']:
            output = f'Your prompt was translated to: {query}\n\n\n{output}'

        if self.g.settings_data['batch']:
            batch_result = self.run_batch_response(self.g.last_context)
            output = f'Prompt 0:\n{output}\n\n\n{batch_result}'


        if len(self.g.negative_prompt_list) > 0:
            self.g.last_negative_prompt = ",".join(self.g.negative_prompt_list).lstrip(' ')
            if len(self.g.last_negative_prompt) < 30:
                self.g.last_negative_prompt = self.g.settings_data['negative_prompt']
            if self.g.last_negative_prompt != '':
                output = f'{output} \n\nMaybe helpful negative prompt:\n\n{self.g.last_negative_prompt}'
        else:
            self.g.last_negative_prompt = self.g.settings_data['negative_prompt']
            output = f'{output} \n\nMaybe helpful negative prompt:\n\n{self.g.last_negative_prompt}'

        if len(self.g.models_list) > 0:
            models_out = "\n".join(self.g.models_list)
            if models_out != '':
                output = f'{output} \n\nMaybe helpful models:\n\n{models_out}'


        return output



