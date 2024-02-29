import gradio as gr

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

import prompt_templates
import model_list
import torch
import gc

url = "http://192.168.0.127:6333"


class LLM_INTERFACE:


    def __init__(self):

        self.index='prompts_large'


        self.model_path = model_list.model_list['thebloke/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_M.gguf']['path']

        self.document_store = qdrant_client.QdrantClient(
            # you can use :memory: mode for fast and light-weight experiments,
            # it does not require to have Qdrant deployed anywhere
            # but requires qdrant-client >= 1.1.1
            #location=":memory:"
            # otherwise set Qdrant instance address with:
            url="http://192.168.0.127:6333"
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

        self.prompt_template = prompt_templates.prompt_template_b

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
            # set to at least 1 to use GPU
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

        self.query_engine = self.vector_index.as_query_engine(similarity_top_k=5,llm=self.llm)

        qa_prompt_tmpl_str = (self.prompt_template)
        self.qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": self.qa_prompt_tmpl}
        )


    def log(self,logfile, text):
        f = open(logfile, 'a')
        f.write(f"QUERY: {text} \n")
        f.close()

    def run_llm_response(self, query, history):

        self.log('logfile.txt',f"QUERY: {query} \n")

        if self.instruct is True:
            query = f'[INST]{query}[/INST]'

        response = self.query_engine.query(query)

        self.log('logfile.txt',f"RESPONSE: {response.response} \n")

        return response.response


    def change_model(self,model,temperature,n_ctx,n_gpu_layers,max_tokens,top_k, instruct):

        self.n_ctx=n_ctx
        self.n_gpu_layers=n_gpu_layers
        self.max_tokens=max_tokens
        self.temperature=float(temperature)
        self.top_k=top_k
        self.instruct = instruct

        self.model_path = model_list.model_list[model]['path']

        self.llm._model = None
        del self.llm

        # delete the model from Ram
        gc.collect()

        self.set_llm()

        self.set_pipeline()
        return f'Model set to {model}'

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