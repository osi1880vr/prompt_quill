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


import os
import time

from llmware.library import Library
from llmware.status import Status
import datetime

os.environ['COLLECTION_DB_URI'] = 'mongodb://localhost:27017/'
os.environ['MILVUS_HOST'] = 'localhost'
os.environ['MILVUS_PORT'] = '19530'
def rag (library_name):

    # Step 0 - Configuration - we will use these in Step 4 to install the embeddings
    #embedding_model = "industry-bert-contracts"
    embedding_model = 'mini-lm-sbert'
    vector_db = "milvus"

    # Step 1 - Create library which is the main 'organizing construct' in llmware
    print ("\nupdate: Step 1 - Creating library: {}".format(library_name))

    library = Library().create_new_library(library_name)

    # Step 3 - point ".add_files" method to the folder of documents that was just created
    #   this method parses all of the documents, text chunks, and captures in MongoDB
    print("update: Step 2 - Parsing and Text Indexing Files")

    sample_files_path = 'E:\short_large'

    t0 = time.time()
    for subdir, dirs, files in os.walk(sample_files_path):
        if len(files) > 0:
            t2 = time.time()
            now = datetime.datetime.now()
            print(f'{now.strftime("%H:%M:%S")} adding folder: {subdir}')
            library.add_files(input_folder_path=subdir)
            print(f"done - parsing time - {time.time()-t2}")
            t1 = time.time()
            library.install_new_embedding(embedding_model_name=embedding_model, vector_db=vector_db)
            print(f"done - embedding time - {time.time()-t1}")


    # Step 4 - Install the embeddings
    print("\nupdate: Step 3 - Generating Embeddings in {} db - with Model- {}".format(vector_db, embedding_model))


    # note: for using llmware as part of a larger application, you can check the real-time status by polling Status()
    #   --both the EmbeddingHandler and Parsers write to Status() at intervals while processing
    update = Status().get_embedding_status(library_name, embedding_model)
    print("update: Embeddings Complete - Status() check at end of embedding - ", update)

    print(f"done - total processing time - {time.time()-t0}")


if __name__ == "__main__":

    # pick any name for the library
    user_selected_name = "llmware_prompts_large"
    rag(user_selected_name)

