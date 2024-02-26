import datetime
import os

from llama_index import GPTVectorStoreIndex,VectorStoreIndex, StorageContext, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding

import torch

from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate



vector_store = MilvusVectorStore(
    uri = "http://192.168.0.127:19530",
    port = 19530   ,
    collection_name = 'llama_index_prompts_all',
    dim = 384,
    similarity_metric = "L2",
    #text_key="paragraph",
    #overwrite=True
)


sample_files_path = "E:\prompt_sources\lexica_split"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)

for subdir, dirs, files in os.walk(sample_files_path):
    if len(files) > 0:
        now = datetime.datetime.now()
        print(f'{now.strftime("%H:%M:%S")} adding folder: {subdir}')

        documents = SimpleDirectoryReader(subdir).load_data()

        docs = []
        for doc in documents:
            doc.excluded_llm_metadata_keys.append("file_path")
            doc.excluded_embed_metadata_keys.append("file_path")
            if doc.text != '':
                docs = docs + [doc]

        del documents

        vector_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, service_context=service_context, show_progress=True)

        vector_store.collection.flush()
        vector_store.collection.compact()




