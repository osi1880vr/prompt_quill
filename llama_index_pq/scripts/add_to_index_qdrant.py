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

import datetime
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

sample_files_path = "E:\prompt_sources\horde_split"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")

vector_store = QdrantVectorStore(client=client, collection_name="prompts_large_meta",)
storage_context = StorageContext.from_defaults(vector_store=vector_store)



for subdir, dirs, files in os.walk(sample_files_path):
    if len(files) > 0:
        now = datetime.datetime.now()
        print(f'{now.strftime("%H:%M:%S")} adding folder: {subdir}')

        documents = SimpleDirectoryReader(subdir).load_data()

        docs = []
        for doc in documents:
            if doc.text != '':
                doc.excluded_llm_metadata_keys.append("file_path")
                doc.excluded_embed_metadata_keys.append("file_path")
                doc.excluded_llm_metadata_keys.append("negative_prompt")
                doc.excluded_embed_metadata_keys.append("negative_prompt")
                doc.excluded_llm_metadata_keys.append("model_name")
                doc.excluded_embed_metadata_keys.append("model_name")

                if '##superspacer##' in doc.text:

                    meta_array = doc.text.split('##superspacer##')
                    doc.text = meta_array[1]

                    doc.metadata['negative_prompt'] = meta_array[2]
                    doc.metadata['model_name'] = f'https://civitai.com/models/{meta_array[3]}'
                else:
                    doc.metadata['negative_prompt'] = ""
                    doc.metadata['model_name'] = ""


                docs = docs + [doc]

        del documents


        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, embed_model=embed_model, show_progress=True
        )


