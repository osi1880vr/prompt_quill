import datetime
import logging
import sys
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import qdrant_client

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.storage.storage_context import StorageContext
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



sample_files_path = "E:\short_large"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")

vector_store = QdrantVectorStore(client=client, collection_name="prompts_large",)
storage_context = StorageContext.from_defaults(vector_store=vector_store)



for subdir, dirs, files in os.walk(sample_files_path):
    if len(files) > 0:
        now = datetime.datetime.now()
        print(f'{now.strftime("%H:%M:%S")} adding folder: {subdir}')

        documents = SimpleDirectoryReader(subdir).load_data()

        # here we set the file_path to become no part of the embedding, its not for this usecase
        # also we check if a doc has zero content then we dont try to embedd it as it would result in an error
        docs = []
        for doc in documents:
            doc.excluded_llm_metadata_keys.append("file_path")
            doc.excluded_embed_metadata_keys.append("file_path")
            if doc.text != '':
                docs = docs + [doc]

        del documents


        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, embed_model=embed_model, show_progress=True
        )


