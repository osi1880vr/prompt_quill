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


from llama_index.core.prompts import PromptTemplate
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
import qdrant_client
from settings.io import settings_io



url = "http://localhost:6333"

if os.getenv("QDRANT_URL") is not None:
    url = os.environ["QDRANT_URL"]

class adapter:

    def __init__(self):
        self.g = globals.get_globals()
        self.set_document_store()
        self.llm = self.set_llm()
        self.set_pipeline()
        self.last_context = []

    def get_instruct(self):
        return self.g.settings_data['Instruct Model']
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
        self.document_store = qdrant_client.QdrantClient(url=url)
        self.get_all_collections()

    def get_llm(self):
        return self.llm
    def set_llm(self):

        return LlamaCPP(

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

    def get_retriever(self, similarity_top_k):
        return self.vector_index.as_retriever(similarity_top_k=similarity_top_k)

    def set_pipeline(self):

        if hasattr(self,'query_engine'):
            del self.vector_store
            del self.vector_index
            del self.query_engine

        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
        self.vector_store = QdrantVectorStore(client=self.document_store, collection_name=self.g.settings_data['collection'])
        self.vector_index = VectorStoreIndex.from_vector_store( vector_store=self.vector_store, embed_model=self.embed_model)

        self.retriever = self.get_retriever(similarity_top_k=self.g.settings_data['top_k'])

        self.query_engine = self.vector_index.as_query_engine(similarity_top_k=self.g.settings_data['top_k'],llm=self.llm)

        self.qa_prompt_tmpl = PromptTemplate(self.g.settings_data['prompt_templates']['prompt_template_b'])

        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": self.qa_prompt_tmpl}
        )

    def direct_search(self,query,limit,offset):

        vector = self.embed_model.get_text_embedding(query)
        result = self.document_store.search(collection_name=self.g.settings_data['collection'],
                                   query_vector=vector,
                                   limit=limit,
                                   offset=offset
                                   )
        return result


    def set_rephrase_pipeline(self, context):
        if hasattr(self,'query_rephrase_engine'):
            del self.query_rephrase_engine


        node1 = TextNode(text=context, id_="<node_id>")

        nodes = [node1]
        index = VectorStoreIndex(nodes,embed_model=self.embed_model)

        test = index.as_retriever(similarity_top_k=1)

        check = test.retrieve('hello world')

        print(check)


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
        return self.retriever.retrieve(prompt)


    def get_context_text(self, query):
        nodes = self.retrieve_context(query)
        return [s.node.get_text() for s in nodes]


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



    def retrieve_query(self, query):
        try:
            self.llm._model.reset()
            response =  self.query_engine.query(query)
            self.prepare_meta_data(response)
            self.g.last_context = [s.node.get_text() for s in response.source_nodes]
            return response.response.lstrip(" ")
        except Exception as e:
            return 'something went wrong:' + str(e)

    def retrieve_rephrase_query(self, query, context):
        self.set_rephrase_pipeline(context)
        self.llm._model.reset()
        response =  self.query_rephrase_engine.query(query)
        response = response.response.lstrip(" ")
        return response.replace('"','')

    def change_model(self,model,temperature,n_ctx,n_gpu_layers,max_tokens,top_k, instruct):

        self.g.settings_data["Context Length"] = n_ctx
        self.g.settings_data["GPU Layers"] = n_gpu_layers
        self.g.settings_data["max output Tokens"] = max_tokens
        self.g.settings_data["Temperature"] = float(temperature)
        self.g.settings_data["top_k"] = top_k
        self.g.settings_data['Instruct Model'] = instruct
        self.g.settings_data['LLM Model'] = model["name"]

        self.llm._model = None
        del self.llm

        self.llm = self.set_llm()

        # delete the model from Ram
        gc.collect()

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

        self.llm._model = None
        del self.llm

        # delete the model from Ram
        gc.collect()

        self.llm = self.set_llm()

        self.set_pipeline()
        return f'Magic Prompt set'