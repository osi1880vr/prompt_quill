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
import gc
import os
import json
import torch
import random
import re
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_path" has conflict')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_url" has conflict')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_kwargs" has conflict')

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core import VectorStoreIndex, PromptHelper
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from qdrant_client.http.models import Filter, FieldCondition, MatchText, SearchParams
import qdrant_client
from settings.io import settings_io
import shared





url = "http://localhost:6333"

if os.getenv("QDRANT_URL") is not None:
    url = os.environ["QDRANT_URL"]

class adapter:

    def __init__(self):
        self.qa_prompt_tmpl = None
        self.query_engine = None
        self.retriever = None
        self.embed_model = None
        self.vector_index = None
        self.vector_store = None
        self.g = globals.get_globals()
        self.set_document_store()
        self.llm_state = None
        self.llm = None #self.set_llm()
        #self.set_pipeline()
        self.last_context = []
        self.prompt_array_index = {}



    def get_document_store(self):
        return self.document_store


    def get_all_collections(self):
        try:
            collections_list = []
            collections = self.document_store.get_collections()
            for collection in collections:
                for c in list(collection[1]):
                    collections_list.append(c.name)
            self.g.settings_data['collections_list'] = collections_list
        except Exception as e:
            print(f"Error fetching collections from Qdrant: {e}")

    def set_document_store(self):
        self.document_store = qdrant_client.QdrantClient(url=url,
                                                         timeout=60)
        self.get_all_collections()

    def get_llm(self):
        return self.llm

    def set_llm(self):

        self.llm = LlamaCPP(

            model_url=self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['path'],

            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=None,

            temperature=self.g.settings_data["Temperature"],
            max_new_tokens=self.g.settings_data["max output Tokens"],

            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=self.g.settings_data["Context Length"],  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.

            # kwargs to pass to __call__()
            generate_kwargs={},

            # kwargs to pass to __init__()
            # set to at least 1 to use GPU, check with your model the number need to fully run on GPU might be way higher than 1
            model_kwargs={"n_gpu_layers": self.g.settings_data["GPU Layers"]}, # I need to play with this and see if it actually helps

            # transform inputs into Llama2 format
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )

        if not self.embed_model:
            self.set_pipeline()
        return self.llm



    def get_retriever(self, similarity_top_k):
        self.check_llm_loaded()
        return self.vector_index.as_retriever(similarity_top_k=similarity_top_k)

    def set_pipeline(self):

        if hasattr(self,'query_engine'):
            del self.vector_store
            del self.vector_index
            del self.query_engine

        self.embed_model = HuggingFaceEmbedding(model_name=self.g.settings_data['embedding_model'])
        self.vector_store = QdrantVectorStore(client=self.document_store, collection_name=self.g.settings_data['collection'])
        self.vector_index = VectorStoreIndex.from_vector_store( vector_store=self.vector_store, embed_model=self.embed_model)

        self.retriever = self.get_retriever(similarity_top_k=self.g.settings_data['top_k'])

        self.query_engine = self.vector_index.as_query_engine(similarity_top_k=self.g.settings_data['top_k'],llm=self.llm)

        self.qa_prompt_tmpl = PromptTemplate(self.g.settings_data['prompt_templates'][self.g.settings_data['selected_template']])

        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": self.qa_prompt_tmpl}
        )


    def filter_context(self, nodes,context_retrieve):
        out_nodes = []
        n = 0
        for node in nodes:
            payload = json.loads(node.payload['_node_content'])
            if not shared.check_filtered(payload['text']):
                out_nodes.append(node)
                n += 1
                if context_retrieve:
                    if n == self.g.settings_data['top_k']:
                        break
            if self.g.job_running is False:
                break
        return out_nodes


    def get_context_filter(self):

        must = []
        must_not = []
        if self.g.settings_data['sailing']['sail_filter_context']:
            if len(self.g.settings_data['sailing']['sail_filter_not_text']) > 0:
                for word in self.g.settings_data['sailing']['sail_filter_not_text'].split(','):
                    must.append(
                        FieldCondition(
                            key="search",
                            match=MatchText(text=f"{word.strip()}"),
                        )
                    )

            if len(self.g.settings_data['sailing']['sail_filter_text']) > 0:
                for word in self.g.settings_data['sailing']['sail_filter_text'].split(','):
                    must_not.append(
                        FieldCondition(
                            key="search",
                            match=MatchText(text=f"{word.strip()}"),
                        )
                    )

        if self.g.settings_data['sailing']['sail_neg_filter_context']:
            if len(self.g.settings_data['sailing']['sail_neg_filter_not_text']) > 0:
                for word in self.g.settings_data['sailing']['sail_neg_filter_not_text'].split(','):
                    must.append(
                        FieldCondition(
                            key="negative_prompt",
                            match=MatchText(text=f"{word.strip()}"),
                        )
                    )

            if len(self.g.settings_data['sailing']['sail_neg_filter_text']) > 0:
                for word in self.g.settings_data['sailing']['sail_neg_filter_text'].split(','):
                    must_not.append(
                        FieldCondition(
                            key="negative_prompt",
                            match=MatchText(text=f"{word.strip()}"),
                        )
                    )

        if len(must) < 1:
            must = None
        if len(must_not) < 1:
            must_not = None
        filter = Filter(must=must,
                        must_not=must_not)
        return filter

    def count_context(self):
        self.check_llm_loaded()
        filter = self.get_context_filter()
        result = self.document_store.count(collection_name=self.g.settings_data['collection'],
                                           count_filter=filter,
                                            )
        return result

    def direct_search(self,query,limit,offset,context_retrieve=False):
        self.check_llm_loaded()

        vector = self.embed_model.get_text_embedding(query)

        filter = self.get_context_filter()

        if self.g.settings_data['sailing']['sail_filter_context'] or self.g.settings_data['sailing']['sail_neg_filter_context']:
            result = self.document_store.search(collection_name=self.g.settings_data['collection'],
                                       query_vector=vector,
                                       limit=limit,
                                       offset=self.g.settings_data['sailing']['sail_depth_preset']+((offset+1)*limit),
                                       query_filter=filter,
                                       search_params=SearchParams(hnsw_ef=128, exact=False),
                                       )

        else:
            result = self.document_store.search(collection_name=self.g.settings_data['collection'],
                                                query_vector=vector,
                                                limit=limit,
                                                offset=self.g.settings_data['sailing']['sail_depth_preset']+((offset+1)*limit)
                                                )
        return result


    def set_rephrase_pipeline(self, context):
        if hasattr(self,'query_rephrase_engine'):
            del self.query_rephrase_engine


        node1 = TextNode(text=context, id_="<node_id>")

        nodes = [node1]
        index = VectorStoreIndex(nodes,embed_model=self.embed_model)

        self.query_rephrase_engine = index.as_query_engine(similarity_top_k=1,llm=self.llm)
        rephrase_prompt = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,\n""" + self.g.settings_data['rephrase_instruction'] + "\nQuery: {query_str}\nAnswer: "

        qa_prompt_tmpl = PromptTemplate(rephrase_prompt)

        self.query_rephrase_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )



    def retrieve_context(self, prompt):
        return self.direct_search(prompt,self.g.settings_data['top_k'],0,True)


    def get_context_text(self, query):
        self.check_llm_loaded()
        nodes = self.retrieve_context(query)
        self.prepare_meta_data_from_nodes(nodes)
        self.g.last_context_list = []
        for node in nodes:
            payload = json.loads(node.payload['_node_content'])
            self.g.last_context_list.append(payload['text'])
        return "\n".join(list(set(self.g.last_context_list)))


    def prepare_meta_data_from_nodes(self, nodes):
        self.g.negative_prompt_list = []
        self.g.models_list = []
        negative_prompts = []
        for node in nodes:
            if hasattr(node, 'metadata'):
                if 'negative_prompt' in node.metadata:
                    negative_prompts = negative_prompts + node.metadata['negative_prompt'].split(',')
                if 'model_name' in node.metadata:
                    self.g.models_list.append(f'{node.metadata["model_name"]}')
            if hasattr(node, 'payload'):
                if 'negative_prompt' in node.payload:
                    if node.payload['negative_prompt'] is not None:
                        negative_prompts = negative_prompts + node.payload['negative_prompt'].split(',')
                if 'model_name' in node.payload:
                    self.g.models_list.append(f'{node.payload["model_name"]}')
            if len(negative_prompts) > 0:
                self.g.negative_prompt_list = set(negative_prompts)


    def prepare_meta_data(self, response):
        self.g.negative_prompt_list = []
        self.g.models_list = []
        negative_prompts = []
        for key in response.metadata.keys():
            if 'negative_prompt' in response.metadata[key]:
                negative_prompts = negative_prompts + response.metadata[key]['negative_prompt'].split(',')
            if 'model_name' in response.metadata[key]:
                self.g.models_list.append(f'{response.metadata[key]["model_name"]}')

            if len(negative_prompts) > 0:
                self.g.negative_prompt_list = set(negative_prompts)

    def process_prompt_arrays(self, input_string):
        processor = shared.WildcardResolver()
        result = processor.resolve_prompt(input_string)
        return result

    def prepare_prompt(self,prompt,context):

        prompt = self.process_prompt_arrays(prompt)

        meta_prompt = self.g.settings_data['prompt_templates'][self.g.settings_data['selected_template']]
        meta_prompt = f"{meta_prompt}"
        instruction_start = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['instruction_start']
        start_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['start_pattern']
        user_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['user_pattern']
        assistant_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['assistant_pattern']
        return meta_prompt.format(query_str=prompt, context_str=context, instruction_start=instruction_start,start_pattern=start_pattern,user_pattern=user_pattern,assistant_pattern=assistant_pattern)


    def prepare_meta_prompt(self,meta_prompt, context):

        instruction_start = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['instruction_start']
        start_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['start_pattern']
        user_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['user_pattern']
        assistant_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['assistant_pattern']
        return meta_prompt.format(context=context, instruction_start=instruction_start,start_pattern=start_pattern,user_pattern=user_pattern,assistant_pattern=assistant_pattern)



    def prepare_model_test_prompt(self,prompt,context):
        meta_prompt = self.g.settings_data['prompt_templates']['model_test_instruction']
        meta_prompt = f"{meta_prompt}"
        instruction_start = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['instruction_start']
        start_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['start_pattern']
        user_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['user_pattern']
        assistant_pattern = self.g.settings_data['model_list'][self.g.settings_data['LLM Model']]['assistant_pattern']
        return meta_prompt.format(query_str=prompt, context_str=context, instruction_start=instruction_start,start_pattern=start_pattern,user_pattern=user_pattern,assistant_pattern=assistant_pattern)



    def save_state(self):
        if self.llm:
            self.llm_state = self.llm._model.save_state()

    def reload_state(self):
        if self.llm_state:
            self.llm._model.load_state(state=self.llm_state)

    def create_completion(self, prompt):

        self.check_llm_loaded()
        if self.g.settings_data["sailing"]["reset_model"]:
            self.reset_model()

        completion_chunks = self.llm._model.create_completion(
            prompt=prompt,
            temperature=self.g.settings_data["Temperature"],
            max_tokens=self.g.settings_data["max output Tokens"],
            top_p=self.g.settings_data["llm_settings"]["top_p"],
            min_p=self.g.settings_data["llm_settings"]["min_p"],
            typical_p=self.g.settings_data["llm_settings"]["typical_p"],
            frequency_penalty=self.g.settings_data["llm_settings"]["frequency_penalty"],
            presence_penalty=self.g.settings_data["llm_settings"]["presence_penalty"],
            repeat_penalty=self.g.settings_data["llm_settings"]["repeat_penalty"],
            top_k=self.g.settings_data["llm_settings"]["top_k"],
            stream=False,
            seed=None,
            tfs_z=1,
            mirostat_mode=0,
            mirostat_tau=5,
            mirostat_eta=0.1,
            grammar=None
        )
        output = completion_chunks['choices'][0]['text']

        gc.collect()
        torch.cuda.empty_cache()

        if '</think>' in output:
            output = re.sub(r".*?</think>", "", output, flags=re.DOTALL)

        return output.strip()

    def check_llm_loaded(self):
        if not hasattr(self, 'llm') or not self.llm:
            self.llm = self.set_llm()
            self.reload_state()

    def retrieve_model_test_llm_completion(self, prompt):
        self.reset_model()

        context = self.get_context_text(prompt)
        self.g.last_context = context
        prompt = self.prepare_model_test_prompt(prompt,context)

        return self.create_completion(prompt)

    def retrieve_llm_completion(self, prompt, sail_keep_text=False):

        self.check_llm_loaded()

        self.reset_model()

        context = self.get_context_text(prompt)
        self.g.last_context = context

        if sail_keep_text:
            prompt = self.prepare_prompt(self.g.settings_data['sailing']['sail_text'],context)
        else:
            prompt = self.prepare_prompt(prompt,context)

        result = self.create_completion(prompt)

        if self.g.settings_data['unload_llm']:
            self.del_llm_model()

        return result

    def retrieve_query(self, query):
        self.check_llm_loaded()

        try:
            self.reset_model()
            answer = self.retrieve_llm_completion(query)
            gc.collect()
            torch.cuda.empty_cache()
            return answer

        except Exception as e:
            return 'something went wrong:' + str(e)

    def retrieve_rephrase_query(self, query, context):

        self.check_llm_loaded()
        self.set_rephrase_pipeline(context)
        self.reset_model()
        response =  self.query_rephrase_engine.query(query)
        gc.collect()
        torch.cuda.empty_cache()
        response = response.response.strip(" ")
        return response.replace('"','')

    def del_llm_model(self):
        if hasattr(self, 'llm'):
            #self.save_state()
            if hasattr(self.llm, '_model'):
                self.llm._model = None
            del self.llm
        # delete the model from Ram
        gc.collect()
        torch.cuda.empty_cache()

    def reset_model(self):
        if self.llm:
            self.llm._model.reset()
            #self.reload_state()

    def change_model(self,model,temperature,n_ctx,n_gpu_layers,max_tokens,top_k):

        self.g.settings_data["Context Length"] = n_ctx
        self.g.settings_data["GPU Layers"] = n_gpu_layers
        self.g.settings_data["max output Tokens"] = max_tokens
        self.g.settings_data["Temperature"] = float(temperature)
        self.g.settings_data["top_k"] = top_k
        self.g.settings_data['LLM Model'] = model["name"]

        self.del_llm_model()

        self.llm = self.set_llm()

        self.set_pipeline()
        settings_io().write_settings(self.g.settings_data)
        return f'Model set to {model["name"]}'

    def log(self,logfile, text):
        f = open(logfile, 'a')
        try:
            f.write(f"{text}\n")
        except:
            pass
        f.close()

    def set_prompt(self,prompt_text):

        self.g.settings_data['prompt_templates']['prompt_template_b'] = prompt_text

        self.log('magic_prompt_logfile.txt',f"Magic Prompt: \n{prompt_text} \n")

        self.del_llm_model()

        self.llm = self.set_llm()

        self.set_pipeline()
        return f'Magic Prompt set'