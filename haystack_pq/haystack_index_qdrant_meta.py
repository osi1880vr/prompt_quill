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

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice
from haystack import Document


document_store = QdrantDocumentStore(
    url="http://192.168.0.127:6333",
    index='haystack_large_meta',
    recreate_index=True,
    return_embedding=True,
    wait_result_from_api=True,

)


p = Pipeline()
p.add_component(instance=DocumentCleaner(), name="cleaner")
p.add_component(instance=DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30), name="splitter")
p.add_component(
    instance=SentenceTransformersDocumentEmbedder(model="BAAI/bge-base-en-v1.5",
                                                  device=ComponentDevice.from_str("cuda:0")), name="embedder"
)

p.add_component("writer", DocumentWriter(document_store=document_store))
p.connect("cleaner", "splitter")
p.connect("splitter", "embedder")
p.connect("embedder", "writer")


import os
sample_files_path = 'X:\\csv2'

for subdir, dirs, files in os.walk(sample_files_path):


    try:
        if len(files) > 0:
            raw_docs = []
            now = datetime.datetime.now()
            print(f'\n{now.strftime("%H:%M:%S")} start adding documents: {subdir}')
            for file in files:
                file_path = os.path.join(subdir,file)
                f = open(file_path, 'r', encoding='utf8', errors='ignore')
                text = f.read()
                f.close()

                meta_array = text.split('##superspacer##')

                text = meta_array[1]

                negative_prompt = meta_array[2]
                model_name = f'https://civitai.com/models/{meta_array[3]}'


                doc = Document(content=text, meta={"negative_prompt": negative_prompt, "model_name": model_name})
                raw_docs.append(doc)




            now = datetime.datetime.now()
            print(f'\n{now.strftime("%H:%M:%S")} adding embedding: {subdir}')
            result = p.run({"cleaner": {"documents": raw_docs}},debug=True)
    except Exception as e:
        print(e)


