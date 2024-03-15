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

from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
import prompt_templates
import model_list
import gc
import os




url = "http://localhost:6333"

if os.getenv("QDRANT_URL")  is not None:
    url = os.getenv("QDRANT_URL")

index = 'haystack_large_meta'

class LLM_INTERFACE:

    def __init__(self):

        self.index=index
        self.url = url
        self.last_prompt = ''
        self.last_negative_prompt = ''

        self.model_path = model_list.model_list[list(model_list.model_list.keys())[0]]['path']

        self.document_store = QdrantDocumentStore(
            url=self.url,
            index=self.index,
            return_embedding=True,
            wait_result_from_api=True,
        )

        self.n_ctx=3900
        self.n_batch=128
        self.n_gpu_layers=50
        self.max_tokens=200
        self.temperature=0.0
        self.top_k=10


        self.generator = LlamaCppGenerator(
            model=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            model_kwargs={"n_gpu_layers": self.n_gpu_layers},
            generation_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
        )

        self.prompt_template = prompt_templates.prompt_template_b
        self.set_pipeline()

    def set_pipeline(self):

        if hasattr(self,'rag_pipeline'):
            del self.text_embedder
            del self.retriever
            del self.llm
            del self.rag_pipeline


        self.text_embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-base-en-v1.5")
        self.retriever = QdrantEmbeddingRetriever(document_store=self.document_store,top_k=self.top_k)
        self.prompt_builder = PromptBuilder(template=self.prompt_template)
        self.llm = self.generator
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("text_embedder", self.text_embedder)
        self.rag_pipeline.add_component("retriever", self.retriever)
        self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.rag_pipeline.add_component("llm", self.llm)
        self.rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
        self.rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder", "llm")
        self.rag_pipeline.connect("llm.replies", "answer_builder.replies")
        self.rag_pipeline.connect("llm.meta", "answer_builder.meta")
        self.rag_pipeline.connect("retriever", "answer_builder.documents")


    def run_llm_response(self, query, history):

        f = open('logfile.txt', 'a')
        f.write(f"QUERY: {query} \n")
        f.close()

        results = self.rag_pipeline.run(
            {
                "text_embedder": {"text": query},
                "prompt_builder": {"question": query},
                "answer_builder": {"query": query},
            }
        )

        output = results['answer_builder']['answers'][0].data.lstrip(' ')
        self.last_prompt = output


        if 'answer_builder' in results:
            docs = results['answer_builder']['answers'][0].documents
            negative_prompts = []
            models = []
            for doc in docs:
                negative_prompts += doc.meta['negative_prompt'].split(',')
                models.append(doc.meta['model_name'])


            if len(negative_prompts) > 0:
                set(negative_prompts)
                self.last_negative_prompt = ",".join(negative_prompts).lstrip(' ')
                output = f'{output} \n\nMaybe helpful negative prompt:\n\n{self.last_negative_prompt}'

            if len(models) > 0:
                models_out = "\n".join(models)

                output = f'{output} \n\nMaybe helpful models:\n\n{models_out}'



        f = open('logfile.txt', 'a')
        f.write(f'RESPONSE: {output} \n')
        f.close()

        return output

    def change_model(self,model,temperature,n_ctx,n_batch,n_gpu_layers,max_tokens,top_k):
        self.n_ctx=n_ctx
        self.n_batch=n_batch
        self.n_gpu_layers=n_gpu_layers
        self.max_tokens=max_tokens
        self.temperature=float(temperature)
        self.top_k=top_k

        self.model_path = model_list.model_list[model]['path']

        del self.generator.model
        del self.generator

        # delete the model from Ram
        gc.collect()

        self.generator = LlamaCppGenerator(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            model_kwargs={"n_gpu_layers": self.n_gpu_layers},
            generation_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
        )

        self.llm = self.generator
        self.set_pipeline()
        return f'Model set to {model}'

    def set_prompt(self,prompt_text):
        self.prompt_template = prompt_text

        f = open('magic_prompt_logfile.txt', 'a')
        f.write(f"Magic Prompt: \n{prompt_text} \n\n\n")
        f.close()

        del self.generator.model
        del self.generator

        # delete the model from Ram
        gc.collect()

        self.generator = LlamaCppGenerator(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            model_kwargs={"n_gpu_layers": self.n_gpu_layers},
            generation_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
        )

        self.llm = self.generator
        self.set_pipeline()
        return f'Magic Prompt set to:\n {prompt_text}'