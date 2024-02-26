import datetime
import logging
import sys
import os
from llama_index.embeddings import HuggingFaceEmbedding
import torch
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
import qdrant_client

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore


client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    #location=":memory:"
    # otherwise set Qdrant instance address with:
    url="http://192.168.0.127:6333"
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

sample_files_path = "E:\prompt_sources\lexica_split"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
service_context = ServiceContext.from_defaults(llm=None,embed_model=embed_model)
vector_store = QdrantVectorStore(client=client, collection_name="prompts_all")
storage_context = StorageContext.from_defaults(vector_store=vector_store)



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


        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, service_context=service_context, show_progress=True
        )


